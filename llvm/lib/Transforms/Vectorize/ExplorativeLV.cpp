//===- ExplorativeLV.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass is inserted in the middle end pipeline immediately before the
// loop vectorizer. When enabled, it retrieves the innermost loops of the
// function like it is done for vectorization.
// Each loop is copied into a plain function in a new module; for each
// vectorization and interleaving factor within range (range can be adapted in
// lines 659,660) the loop is annotated with pragmas to force the LV to choose
// this VF and IF. Then it is sent through the compilation pipeline, stopping
// at assembly file generation where MachineCodeExplorer.cpp computes a cost
// estimate based on the machine code.
// The VF-IF combination with the lowest cost is selected and forced onto the
// LV using annotations.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/ExplorativeLV.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/LoopIterator.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Transforms/ObjCARC.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <cstdint>

using namespace llvm;
#define DEBUG_TYPE "explorative-lv"

namespace llvm {

cl::opt<ExplorativeLVMetric> ExplorativeLV(
    "explorative-lv", cl::Hidden,
    cl::desc(
        "Which metric to use for machine code evaluation in loop exploration"),
    cl::init(ExplorativeLVMetric::Disable),
    cl::values(clEnumValN(ExplorativeLVMetric::Disable, "disable",
                          "Disable explorative loop vectorization"),
               clEnumValN(ExplorativeLVMetric::InstCount, "inst-count",
                          "Count instructions"),
               clEnumValN(ExplorativeLVMetric::MCA, "mca", "Use llvm-mca"),
               clEnumValN(ExplorativeLVMetric::Benchmark, "benchmark",
                          "Benchmark")));

} // namespace llvm

static cl::opt<bool>
    ExplorePlain("explore-plain", cl::Hidden, cl::init(false),
                 cl::desc("If loop exploration is enabled, use "
                          "entire function for cost calculation"));

static cl::opt<bool>
    ExploreDivVF("explore-divide-by-vf", cl::Hidden, cl::init(false),
                 cl::desc("If loop exploration is enabled, aggregate "
                          "the cost results by the VF"));

// Function from CodeExtractor.cpp:
/// definedInRegion - Return true if the specified value is defined in the
/// extracted region.
static bool
definedInRegion(const llvm::SmallPtrSetImpl<const llvm::BasicBlock *> &Blocks,
                Value *V) {
  if (Instruction *I = dyn_cast<Instruction>(V))
    return Blocks.contains(I->getParent());
  return false;
}

// Function from CloneModule.cpp
static void copyComdat(GlobalObject *Dst, const GlobalObject *Src) {
  const Comdat *SC = Src->getComdat();
  if (!SC)
    return;
  Comdat *DC = Dst->getParent()->getOrInsertComdat(SC->getName());
  DC->setSelectionKind(SC->getSelectionKind());
  Dst->setComdat(DC);
}
namespace {

class ExplorativeLVCtx {
  const Function &OrigFunc;
  Loop &L;

  Module *M;
  ValueToValueMapTy VMap;
  SmallVector<Value *> Inputs;
  SmallVector<Instruction *> Outputs;
  FunctionType *FuncTy;
  SmallVector<Function *> LoopFuncs;

  void cloneModuleDeclarations(bool CloneDefinitions);
  void determineIO();

public:
  ExplorativeLVCtx(const Function &OrigFunc, Loop &L,
                   bool CloneDefinitions = false);
  ~ExplorativeLVCtx() { delete M; }

  Module &getModule() { return *M; }

  void buildMainFunctionBody();

  Function *buildLoopFunction(unsigned VF, unsigned IF);
  void clearLoopFuncs() {
    for (Function *LoopFunc : LoopFuncs) {
      LoopFunc->eraseFromParent();
    }
    LoopFuncs.clear();
  }
};

// This method uses code of Utils/CloneModule.cpp
// We do not clone any definitions but declarations only.
void ExplorativeLVCtx::cloneModuleDeclarations(bool CloneDefinitions) {
  const Module *OrigM = OrigFunc.getParent();

  // First off, we need to create the new module.
  M = new Module(OrigM->getModuleIdentifier(), OrigM->getContext());
  M->setSourceFileName(OrigM->getSourceFileName());
  M->setDataLayout(OrigM->getDataLayout());
  M->setTargetTriple(OrigM->getTargetTriple());
  M->setModuleInlineAsm(OrigM->getModuleInlineAsm());

  // Loop over all of the global variables, making corresponding globals in the
  // new module.  Here we add them to the VMap and to the new Module.  We
  // don't worry about attributes or initializers, they will come later.
  //
  for (const GlobalVariable &I : OrigM->globals()) {
    // TODO: Only copy globals used in loop?
    GlobalVariable *GV = new GlobalVariable(
        *M, I.getValueType(), I.isConstant(), I.getLinkage(),
        (Constant *)nullptr, I.getName(), (GlobalVariable *)nullptr,
        I.getThreadLocalMode(), I.getType()->getAddressSpace());
    GV->copyAttributesFrom(&I);
    VMap[&I] = GV;
  }

  // Loop over the functions in the module, making external functions as before
  //
  // TODO: when using the benchmarking mode, we would either have to copy
  // function definitions or refrain from analyzing loops with call instructions
  // (at least non-intrinsic ones).
  for (const Function &I : OrigM->functions()) {
    Function *NF = Function::Create(cast<FunctionType>(I.getValueType()),
                                    I.getLinkage(), I.getName(), M);
    NF->copyAttributesFrom(&I);
    VMap[&I] = NF;
  }

  // Loop over the aliases in the module
  for (const GlobalAlias &I : OrigM->aliases()) {
    // An alias cannot act as an external reference, so we need to create
    // either a function or a global variable depending on the value type.
    // FIXME: Once pointee types are gone we can probably pick one or the
    // other.
    GlobalValue *GV;
    if (I.getValueType()->isFunctionTy())
      GV = Function::Create(cast<FunctionType>(I.getValueType()),
                            GlobalValue::ExternalLinkage, I.getAddressSpace(),
                            I.getName(), M);
    else
      GV = new GlobalVariable(*M, I.getValueType(), false,
                              GlobalValue::ExternalLinkage, nullptr,
                              I.getName(), nullptr, I.getThreadLocalMode(),
                              I.getType()->getAddressSpace());
    VMap[&I] = GV;
    // We do not copy attributes (mainly because copying between different
    // kinds of globals is forbidden), but this is generally not required for
    // correctness.
  }

  // Now that all of the things that global variable initializer can refer to
  // have been created, loop through and copy the global variable referrers
  // over...  We also set the attributes on the global now.
  //
  for (const GlobalVariable &G : OrigM->globals()) {
    GlobalVariable *GV = cast<GlobalVariable>(VMap[&G]);

    SmallVector<std::pair<unsigned, MDNode *>, 1> MDs;
    G.getAllMetadata(MDs);
    for (auto MD : MDs)
      GV->addMetadata(MD.first, *MapMetadata(MD.second, VMap));

    if (G.isDeclaration())
      continue;

    if (!CloneDefinitions) {
      // Skip after setting the correct linkage for an external reference.
      GV->setLinkage(GlobalValue::ExternalLinkage);
      continue;
    }
    if (G.hasInitializer())
      GV->setInitializer(MapValue(G.getInitializer(), VMap));

    copyComdat(GV, &G);
  }

  // And named metadata....
  for (const NamedMDNode &NMD : OrigM->named_metadata()) {
    NamedMDNode *NewNMD = M->getOrInsertNamedMetadata(NMD.getName());
    for (const MDNode *Operand : NMD.operands())
      NewNMD->addOperand(MapMetadata(Operand, VMap));
  }
}

void ExplorativeLVCtx::determineIO() {
  // The core idea is to generate a function containing the basic blocks of the
  // loop.  Values used inside the loop and defined outside are parameters to
  // the function.  For each values defined inside the loop and used outside we
  // generate an output parameter to produce an observable side-effect such that
  // the code is not optimized away.

  SmallPtrSet<Value *, 32> ArgSet;

  for (BasicBlock *BB : L.getBlocks()) {
    for (Instruction &I : *BB) {
      // Used inside loop and defined outside?
      for (Value *Op : I.operand_values()) {
        Instruction *OpInst = dyn_cast<Instruction>(Op);
        if (OpInst) {
          if (L.getBlocksSet().contains(OpInst->getParent()))
            continue;
        } else if (!isa<Argument>(Op) || ArgSet.contains(Op)) {
          continue;
        }
        ArgSet.insert(Op);
        Inputs.push_back(Op);
      }

      // Defined inside loop and used outside?
      for (User *U : I.users())
        if (!definedInRegion(L.getBlocksSet(), U)) {
          Outputs.push_back(&I);
          break;
        }
    }
  }
}

ExplorativeLVCtx::ExplorativeLVCtx(const Function &OrigFunc, Loop &L,
                                   bool CloneDefinitions)
    : OrigFunc(OrigFunc), L(L) {
  cloneModuleDeclarations(CloneDefinitions);
  determineIO();

  // Build the loop function type
  SmallVector<Type *> Params;
  Params.reserve(Inputs.size() + Outputs.size());
  for (Value *V : Inputs)
    Params.push_back(V->getType());
  for (Instruction *I : Outputs)
    Params.push_back(PointerType::getUnqual(I->getType()));

  // Returns void, no varargs
  FuncTy = FunctionType::get(Type::getVoidTy(M->getContext()), Params, false);
}

// This method uses a lot of code (and comments) from Utils/CloneFunction.cpp,
// but also from IPO/LoopExtractor.cpp resp. Utils/CodeExtractor.cpp
Function *ExplorativeLVCtx::buildLoopFunction(unsigned VF, unsigned IF) {
  // Note that we actually modify the original loop here.  According to the docs
  // the metadata is attached to the branch instructions in the latches, so it
  // might be a bit difficult to directly add the metadata to the new
  // function.  Hence, we do it here.  Since we finally choose the values
  // explicitly for the original loop, it does not matter much anyway.
  llvm::addStringMetadataToLoop(&L, "llvm.loop.vectorize.width", VF);
  llvm::addStringMetadataToLoop(&L, "llvm.loop.interleave.count", IF);

  Function *LoopFunc = Function::Create(
      FuncTy, GlobalValue::ExternalLinkage, OrigFunc.getAddressSpace(),
      OrigFunc.getName() + "." + std::to_string(VF) + "." + std::to_string(IF),
      M);

  /* Insert the generated function arguments into the VMap, add names */ {
    auto *ArgPtr = LoopFunc->arg_begin();
    for (Value *OldV : Inputs) {
      VMap[OldV] = ArgPtr;
      VMap[ArgPtr] = ArgPtr;
      if (OldV->hasName())
        ArgPtr->setName(OldV->getName() + ".i");
      ++ArgPtr;
    }
    for (Instruction *I : Outputs) {
      VMap[ArgPtr] = ArgPtr;
      if (I->hasName())
        ArgPtr->setName(I->getName() + ".o");
      ++ArgPtr;
    }
  }

  // Introduce empty label at the beginning of the function, and at the
  // end of the function, so we can refer to "before" and "after"
  BasicBlock *BeginBB =
      BasicBlock::Create(LoopFunc->getContext(), "loop_begin", LoopFunc);
  VMap[BeginBB] = BeginBB;

  // Copy argument attributes (~~> CloneFunctionInto)
  AttributeList LoopAttrs = LoopFunc->getAttributes();
  // We do not clone function attributes, since the original function might
  // fulfill properties that the new one does not.
  // However, we add a NoInline attribute, otherwise a lot of code might be
  // eliminated in benchmarking mode.
  LoopAttrs =
      LoopAttrs.addFnAttribute(LoopFunc->getContext(), Attribute::NoInline);

  SmallVector<AttributeSet, 4> LoopArgAttrs(LoopFunc->arg_size());
  AttributeList OldAttrs = OrigFunc.getAttributes();
  for (const Argument &OldArg : OrigFunc.args()) {
    if (VMap.count(&OldArg) == 0)
      continue;
    if (Argument *NewArg = dyn_cast<Argument>(VMap[&OldArg])) {
      LoopArgAttrs[NewArg->getArgNo()] =
          OldAttrs.getParamAttrs(OldArg.getArgNo());
    }
  }

  LoopFunc->setAttributes(
      AttributeList::get(LoopFunc->getContext(), LoopAttrs.getFnAttrs(),
                         LoopAttrs.getRetAttrs(), LoopArgAttrs));

  DebugInfoFinder DIFinder;

  // From here on we need to deviate from CloneFunctionInto to only clone
  // the loop, not a whole function.

  // Loop over all of the basic blocks in the loop, cloning them as
  // appropriate.
  BasicBlock *NewLoopHeader = nullptr;
  for (const BasicBlock *BB : L.getBlocks()) {

    // Create a new basic block and copy instructions into it!
    BasicBlock *NewBB =
        CloneBasicBlock(BB, VMap, ".l", LoopFunc, nullptr, &DIFinder);
    if (!NewLoopHeader)
      NewLoopHeader = NewBB;

    // Add basic block mapping.
    VMap[BB] = NewBB;
    // And map block to itself because the remapping step
    // will try to retrieve it
    VMap[NewBB] = NewBB;

    // Need to map each new instruction clone to itself, so VMap doesn't
    // break when later stuff is looking at instructions we generated, not
    // cloned
    for (Instruction &I : *NewBB)
      VMap[&I] = &I;

    // It is only legal to clone a function if a block address within that
    // function is never referenced outside of the function.  Given that, we
    // want to map block addresses from the old function to block addresses in
    // the clone.  (This is different from the generic ValueMapper
    // implementation, which generates an invalid BlockAddress when
    // cloning a function.)
    if (BB->hasAddressTaken()) {
      Constant *OldBBAddr = BlockAddress::get(const_cast<Function *>(&OrigFunc),
                                              const_cast<BasicBlock *>(BB));
      VMap[OldBBAddr] = BlockAddress::get(LoopFunc, NewBB);
    }
  }

  BasicBlock *EndBB =
      BasicBlock::Create(LoopFunc->getContext(), "loop_end", LoopFunc, nullptr);
  ReturnInst *Ret = ReturnInst::Create(LoopFunc->getContext(), EndBB);
  VMap[EndBB] = EndBB;
  VMap[Ret] = Ret;

  // For each edge that exits the loop: create store block that stores all live
  // variables in the output parameters.
  Function::arg_iterator OutputArgsBegin = LoopFunc->arg_begin();
  std::advance(OutputArgsBegin, Inputs.size());
  for (BasicBlock *BB : L.blocks()) {
    if (!L.isLoopExiting(BB))
      continue;

    Instruction *Term = BB->getTerminator();
    BasicBlock *NewBB = dyn_cast<BasicBlock>(VMap[BB]);
    assert(NewBB && "block mapping missing");
    Instruction *NewTerm = NewBB->getTerminator();

    for (unsigned Idx = 0, Num = Term->getNumSuccessors(); Idx < Num; ++Idx) {
      BasicBlock *SuccBB = Term->getSuccessor(Idx);
      if (L.getBlocksSet().contains(SuccBB))
        continue; // Not an exiting edge
      assert(SuccBB->getParent() == &OrigFunc);

      BasicBlock *StoreBB = BasicBlock::Create(LoopFunc->getContext(),
                                               "store_block", LoopFunc, EndBB);
      VMap[StoreBB] = StoreBB;
      NewTerm->setSuccessor(Idx, StoreBB);

      Function::arg_iterator OutputArg = OutputArgsBegin;
      for (Instruction *I : Outputs) {
        assert(I->getParent()->getParent() == &OrigFunc);
        for (User *U : I->users()) {
          Instruction *UI = dyn_cast<Instruction>(U);
          // A user might be in the new function as we have not remapped the
          // operands yet.  Just ignore these users.
          if (!UI || UI->getFunction() != &OrigFunc ||
              !isPotentiallyReachable(SuccBB, UI->getParent()))
            continue;

          // Use a phi to preserve LCSSA form
          PHINode *LCSSAPhi =
              PHINode::Create(UI->getType(), 1, "lcssa_phi", StoreBB);
          LCSSAPhi->addIncoming(VMap[I], NewBB);
          StoreInst *Store = new llvm::StoreInst(LCSSAPhi, OutputArg, StoreBB);
          VMap[LCSSAPhi] = LCSSAPhi;
          VMap[Store] = Store;

          break;
        }
        ++OutputArg;
      }

      BranchInst *EndBranch = BranchInst::Create(EndBB, StoreBB);
      VMap[EndBranch] = EndBranch;
    }
  }

  BranchInst *Branch = BranchInst::Create(NewLoopHeader, BeginBB);
  VMap[Branch] = Branch;

  // The first basic block might have had multiple predecessors that were
  // not part of the loop.  In our new setup, they are all replaced by
  // BeginBB.  Update the PHINodes accordingly.  Since we have a (natural)
  // loop, no other basic block than the loop header has predecedecessors
  // outside the loop.
  for (PHINode &P : NewLoopHeader->phis()) {
    for (BasicBlock **It = P.block_begin(), **End = P.block_end(); It != End;
         It++) {
      if (VMap.count(*It) == 0)
        *It = BeginBB;
    }
  }

  // Duplicate the metadata that is attached to the cloned function.
  // Subprograms/CUs/types that were already mapped to themselves won't be
  // duplicated.
  SmallVector<std::pair<unsigned, MDNode *>, 1> MDs;
  OrigFunc.getAllMetadata(MDs);
  for (auto MD : MDs) {
    LoopFunc->addMetadata(MD.first, *MapMetadata(MD.second, VMap));
  }

  // Loop over all of the instructions in the new function, fixing up operand
  // references as we go. This uses VMap to do all the hard work.
  for (BasicBlock &BB : *LoopFunc) {
    // Loop over all instructions, fixing each one as we find it...
    for (Instruction &II : BB)
      RemapInstruction(&II, VMap);
  }

  LoopFuncs.push_back(LoopFunc);
  return LoopFunc;
}

void ExplorativeLVCtx::buildMainFunctionBody() {
  LLVMContext &C = M->getContext();

  Function *MainFunc = M->getFunction("main");
  if (!MainFunc) {
    MainFunc = Function::Create(FunctionType::get(Type::getInt32Ty(C), false),
                                GlobalValue::ExternalLinkage, "main", M);
  }

  // TODO: what happens if this function is already declared?
  // -> use M->getOrInsertFunction?
  static_assert(sizeof(clock_t) == 8, "clock_t is assumed to be 64 bit");
  FunctionType *ClockFuncTy = FunctionType::get(Type::getInt64Ty(C), false);
  Function *ClockFunc =
      Function::Create(ClockFuncTy, GlobalValue::ExternalLinkage,
                       OrigFunc.getAddressSpace(), "clock", M);

  FunctionType *PrintfFuncTy =
      FunctionType::get(Type::getInt32Ty(C), {Type::getInt8PtrTy(C)}, true);
  Function *PrintfFunc =
      Function::Create(PrintfFuncTy, GlobalValue::ExternalLinkage,
                       OrigFunc.getAddressSpace(), "printf", M);

  assert(LoopFuncs.size() > 0 && "No loop functions generated yet");
  FunctionType *LoopFuncTy = LoopFuncs[0]->getFunctionType();

  BasicBlock *BB = BasicBlock::Create(C, "main", MainFunc);

  SmallVector<Value *> Args;
  /* Create arguments */ {
    assert(Inputs.size() <= LoopFuncTy->getNumParams());

    auto *It = LoopFuncTy->param_begin();
    auto *OutputParamsBegin = It;
    std::advance(OutputParamsBegin, Inputs.size());

    // Create input arguments
    for (; It != OutputParamsBegin; ++It) {
      // FIXME: we should use more meaningful values
      Args.push_back(Constant::getNullValue(*It));
    }

    // Alloc locations for output parameters
    for (auto *End = LoopFuncTy->param_end(); It != End; ++It) {
      Args.push_back(
          new AllocaInst(cast<PointerType>(*It)->getPointerElementType(),
                         MainFunc->getAddressSpace(), "outarg", BB));
    }
  }

  // Create format string
  static_assert(sizeof(unsigned long long) == 8,
                "unsigned long long is expected to be 64 bit");
  Constant *FormatStr = ConstantDataArray::getString(C, "%llu\n\0", false);
  GlobalVariable *FormatStrGV =
      new GlobalVariable(*M, FormatStr->getType(), true,
                         GlobalValue::InternalLinkage, FormatStr, "fstr");
  Constant *I640 = Constant::getNullValue(Type::getInt64Ty(C));
  Constant *FormatStrPtr = ConstantExpr::getInBoundsGetElementPtr(
      FormatStrGV->getValueType(), FormatStrGV,
      ArrayRef<Constant *>{I640, I640});

  for (Function *LoopFunc : LoopFuncs) {
    // One run that we do not count
    CallInst::Create(LoopFuncTy, LoopFunc, Args, "", BB);

    CallInst *TStart = CallInst::Create(ClockFuncTy, ClockFunc, "ts", BB);

    // Call the function many times to get stable results (hopefully)
    for (unsigned I = 0; I < 1024; ++I) // TODO: which N should we choose?
      CallInst::Create(LoopFuncTy, LoopFunc, Args, "", BB);

    CallInst *TEnd = CallInst::Create(ClockFuncTy, ClockFunc, "te", BB);
    BinaryOperator *TDiff = BinaryOperator::CreateSub(TEnd, TStart, "dt", BB);
    CallInst::Create(PrintfFuncTy, PrintfFunc, {FormatStrPtr, TDiff}, "", BB);
  }

  ReturnInst::Create(C, Constant::getNullValue(MainFunc->getReturnType()), BB);
}

} // end anonymous namespace

// From ClangLinkerWrapper
static std::string getMainExecutable(const char *Name) {
  void *Ptr = (void *)(intptr_t)&getMainExecutable;
  auto COWPath = sys::fs::getMainExecutable(Name, Ptr);
  return sys::path::parent_path(COWPath).str();
}

// The more complex approach, using the runtime estimation mechanisms
// already present in LLVM: llvm-mca
// (To be called from the pass that performs the exploration)
static unsigned performMCACostCalc(StringRef FileName, StringRef TargetCPU,
                                   StringRef TargetTriple) {
  // Before running the analyser, reduce the code we look at to the actual
  // loop code. The script will try to find a vector loop first, but fall back
  // to the for/while loop if no vector loop is found
  // FIXME: This won't work on any architecture other than x86
  // Uncomment and change exec call if you really want to use it!
  /*std::string Command = "python ../llvm/lib/CodeGen/ExtractAssemblyLoop.py ";
  std::string ReducedFileName = "loop.tmp";
  Command.append(FileName);
  Command.append(" ");
  Command.append(ReducedFileName);*/

  // Find llvm-mca executable
  // FIXME: do this only once
  auto MCAExecPath =
      sys::findProgramByName("llvm-mca", {getMainExecutable("llvm-mca")});
  if (!MCAExecPath) {
    errs() << "Explorative LV: Could not find llvm-mca executable\n";
    return UINT_MAX;
  }

  // Build the most specific command line we can generate from the info in TM
  SmallString<32> MCAOutPath;
  sys::fs::createTemporaryFile("llvm-mca-out", "json", MCAOutPath);
  FileRemover MCAOutRemover(MCAOutPath);
  if (sys::ExecuteAndWait(
          MCAExecPath.get(),
          {"llvm-mca", "--mcpu=" + TargetCPU.str(),
           "--mtriple=" + TargetTriple.str(), "--instruction-info=false",
           "--resource-pressure=false", "--json", FileName},
          None, {StringRef(""), MCAOutPath.str(), StringRef("")}) != 0) {
    errs() << "Explorative LV: executing llvm-mca failed";
    return UINT_MAX;
  }

  // Read JSON
  auto MCAOutBuf = MemoryBuffer::getFile(MCAOutPath);
  if (!MCAOutBuf) {
    errs() << "Explorative LV: could not read llvm-mca output\n";
    return UINT_MAX;
  }
  StringRef MCAOut = MCAOutBuf.get()->getBuffer();
  auto MCAJSON = json::parse(MCAOut);
  if (!MCAJSON) {
    errs() << "Explorative LV: could not parse llvm-mca JSON\n";
    return UINT_MAX;
  }

  // Extract CodeRegions[0].SummaryView.TotalCycles from JSON
  do {
    json::Object *Root = MCAJSON.get().getAsObject();
    if (!Root)
      break;
    json::Array *CodeRegions = Root->getArray("CodeRegions");
    if (!CodeRegions || CodeRegions->size() == 0)
      break;
    json::Object *CodeRegion = (*CodeRegions)[0].getAsObject();
    if (!CodeRegion)
      break;
    json::Object *SummaryView = CodeRegion->getObject("SummaryView");
    if (!SummaryView)
      break;
    int64_t TotalCycles = SummaryView->getInteger("TotalCycles").getValueOr(-1);
    if (TotalCycles >= 0)
      return TotalCycles;
  } while (false);

  errs() << "Explorative LV: llvm-mca output is malformed\n";
  return UINT_MAX;
}

// Add to the given pass manager everything we need to simulate our backend
// pipeline
// ~~> EmitAssemblyHelper::AddEmitPasses in clang's BackendUtil
static void initCodeGen(legacy::PassManager &PM, TargetMachine &TM,
                        const TargetLibraryInfo &TLI, raw_pwrite_stream &OS,
                        CodeGenFileType CGFT) {
  // Add LibraryInfo.
  PM.add(new TargetLibraryInfoWrapperPass(TLI));

  // Add ObjC ARC final-cleanup optimizations. This is done as part of the
  // "codegen" passes so that it isn't run multiple times when there is
  // inlining happening.
  PM.add(createObjCARCContractPass());

  TM.addPassesToEmitFile(PM, OS, nullptr, CGFT, /* DisableVerify */ false);
}

struct ExplorativeLVPass::OptPipelineContainer {
  PassInstrumentationCallbacks PIC;
  StandardInstrumentations SI;

  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;
  ModulePassManager MPM;

  // Inspired by BackendUtil::RunCustomizationPipeline (clang)
  // TODO: customization of SI using command line flags or clang's CodeGenOpts?
  OptPipelineContainer(TargetMachine *TM, PipelineTuningOptions PTO,
                       Optional<PGOOptions> PGOOpt,
                       const TargetLibraryInfo &TLI)
      : SI(/* DebugLogging */ false) {
    SI.registerCallbacks(PIC, &FAM);

    // Create a new PassBuilder which is the basis of a pipeline
    // TODO: I'm not 100% sure that it's wise to use the original TM here
    PassBuilder PB(TM, PTO, PGOOpt, &PIC);

    // Register the target library analysis directly and give it a customized
    // preset TLI.
    FAM.registerPass([&] { return TargetLibraryAnalysis(TLI); });

    // Register all the basic analyses with the managers.
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

    MPM = PB.buildPerModuleDefaultPipeline(OptimizationLevel::O3);
  }

  void run(Module &M) {
    // This line "fixes" a bug where sometimes, starting the PM run will
    // trigger an invalid invalidation...
    MAM.clear();

    // Run the middle end pipeline
    MPM.run(M, MAM);
  }
};

struct ExplorativeLVPass::NullCGPipelineContainer {
  raw_null_ostream OS;
  legacy::PassManager PM;

  NullCGPipelineContainer(TargetMachine &TM, const TargetLibraryInfo &TLI) {
    initCodeGen(PM, TM, TLI, OS, CGFT_Null);
  }
};

ExplorativeLVPass::~ExplorativeLVPass() {
  delete NullCGPipeline;
  delete OptPipeline;
}

// returns whether an error occured
bool ExplorativeLVPass::processLoop(Function &F, Loop &L,
                                    TargetLibraryInfo &TLI) {
  assert(OptPipeline && "OptPipeline not initialized");

  ExplorativeLVCtx Ctx(F, L, ExplorativeLV == ExplorativeLVMetric::Benchmark);

  unsigned LowestCostVF = 1;
  unsigned LowestCostIF = 1;
  if (ExplorativeLV != ExplorativeLVMetric::Benchmark) {
    unsigned LowestCost = UINT_MAX;
    // Try out which VF works best for this loop
    for (unsigned VF = 1; VF <= 16; VF *= 2) {
      for (unsigned IF = 1; IF <= 8; IF *= 2) {
        Function *LoopFunc = Ctx.buildLoopFunction(VF, IF);
        OptPipeline->run(Ctx.getModule());

        unsigned Result;
        // Expensive, but: If we are using llvm-mca for evaluation, we need
        // a new backend pipeline for each iteration as the print stream is
        // dead after one use
        if (ExplorativeLV == ExplorativeLVMetric::MCA) {
          int ASMFD;
          SmallString<32> ASMPath;
          sys::fs::createTemporaryFile("explorative-lv", "s", ASMFD, ASMPath);
          FileRemover ASMRemover(ASMPath);
          {
            raw_fd_ostream OS(ASMFD, true);
            legacy::PassManager CGPipeline;
            initCodeGen(CGPipeline, *TM, TLI, OS, CGFT_AssemblyFile);
            CGPipeline.run(Ctx.getModule());
          }
          // Perform the cost calculation after the pipeline is deleted, to
          // make sure the assembly printer has finished
          Result = performMCACostCalc(ASMPath, TM->getTargetCPU(),
                                      Ctx.getModule().getTargetTriple());
        } else {
          // Set MCInstCount to 1 to tell MCExplorer to only go for loops,
          // to anything else to have it count all MCInsts in function
          LoopFunc->getContext().setMCInstCount(ExplorePlain ? 0 : 1);
          if (VF == 1) {
            // Tell MachineCodeExplorer that we do not expect to see a
            // vectorized loop
            LoopFunc->getContext().setScalarExploration(true);
          } else {
            LoopFunc->getContext().setScalarExploration(false);
          }
          NullCGPipeline->PM.run(Ctx.getModule());
          Result = LoopFunc->getContext().getMCInstCount();
        }
        Ctx.clearLoopFuncs();

        if (Result == UINT_MAX) {
          LLVM_DEBUG(dbgs() << "Explorative LV: Invalid costs for VF " << VF
                            << ", IF " << IF << "\n");
          continue;
        }

        if (ExploreDivVF)
          Result /= VF;

        LLVM_DEBUG(dbgs() << "Explorative LV: Received result " << Result
                          << " from cost module for VF " << VF << ", IF " << IF
                          << "\n");

        if (Result < LowestCost ||
            (LowestCostVF != 1 && Result == LowestCost)) {
          // If costs are equal to costs retrieved before, always choose
          // the higher VF-IF combination unless best VF is scalar
          LowestCost = Result;
          LowestCostVF = VF;
          LowestCostIF = IF;
        }
      }
    }
  } else {
    // Build a module containing functions for all VF/IF combinations
    for (unsigned VF = 1; VF <= 16; VF *= 2) {
      for (unsigned IF = 1; IF <= 8; IF *= 2)
        Ctx.buildLoopFunction(VF, IF);
    }

    OptPipeline->run(Ctx.getModule());

    // Build the main function -- no need to optimize it
    Ctx.buildMainFunctionBody();

    // Run code generation pipeline
    SmallString<32> ObjectPath;
    int ObjectFD;
    sys::fs::createTemporaryFile("explorative-lv", "o", ObjectFD, ObjectPath);
    FileRemover ObjectRemover(ObjectPath);
    {
      raw_fd_ostream OS(ObjectFD, true);
      legacy::PassManager CGPipeline;
      initCodeGen(CGPipeline, *TM, TLI, OS, CGFT_ObjectFile);
      CGPipeline.run(Ctx.getModule());
    }

    // Create an executable
    auto CCPath = sys::findProgramByName("cc"); // FIXME: do this only once
    if (!CCPath) {
      errs() << "Explorative LV: Could not find cc executable\n";
      return true;
    }
    SmallString<32> ExecPath;
    sys::fs::createTemporaryFile("explorative-lv", "", ExecPath);
    FileRemover ExecRemover(ExecPath);
    if (sys::ExecuteAndWait(CCPath.get(), {"cc", "-o", ExecPath, ObjectPath}) !=
        0) {
      errs() << "Explorative LV: Linking failed\n";
      return true;
    }

    // Do the benchmarking
    SmallString<32> BenchResPath;
    sys::fs::createTemporaryFile("explorative-lv-out", "txt", BenchResPath);
    FileRemover BenchResRemover(BenchResPath);
    if (sys::ExecuteAndWait(
            ExecPath, {"explorative-lv"}, None,
            {StringRef(""), BenchResPath.str(), StringRef("")}) != 0) {
      errs() << "Explorative LV: benchmarking failed\n";
      return true;
    }

    auto BenchResBuf = MemoryBuffer::getFile(BenchResPath);
    if (!BenchResBuf) {
      errs() << "Explorative LV: could not load benchmarking results";
      return true;
    }
    StringRef BenchRes = BenchResBuf.get()->getBuffer();

    uint64_t LowestCost = UINT64_MAX;
    for (unsigned VF = 1; VF <= 16; VF *= 2) {
      for (unsigned IF = 1; IF <= 8; IF *= 2) {
        uint64_t Result;
        if (BenchRes.consumeInteger(10, Result)) {
          errs() << "Explorative LV: could not read integer";
          return true;
        }
        BenchRes = BenchRes.drop_front(); // drop '\n'

        LLVM_DEBUG(dbgs() << "Explorative LV: costs for VF " << VF << ", IF "
                          << IF << ": " << Result << "\n");
        if (Result < LowestCost ||
            (LowestCostVF != 1 && Result == LowestCost)) {
          // If costs are equal to costs retrieved before, always choose
          // the higher VF-IF combination unless best VF is scalar
          LowestCost = Result;
          LowestCostVF = VF;
          LowestCostIF = IF;
        }
      }
    }
  }

  // Force the vectorizer to use the VF and IF that showed to have the lowest
  // cost
  LLVM_DEBUG(
      dbgs() << "Explorative LV: Choosing this VF and IF for vectorization: "
             << LowestCostVF << " and " << LowestCostIF << "\n\n");
  addStringMetadataToLoop(&L, "llvm.loop.vectorize.width", LowestCostVF);
  addStringMetadataToLoop(&L, "llvm.loop.interleave.count", LowestCostIF);

  return false; // no errors
}

// Inspired by LoopVectorize.cpp
static void collectSupportedLoops(Loop &L, LoopInfo *LI,
                                  SmallVectorImpl<Loop *> &V) {
  if (L.isInnermost()) {
    if (L.getBlocks().empty())
      return;

    if (findStringMetadataForLoop(&L, "llvm.loop.vectorize.width")) {
      LLVM_DEBUG(dbgs() << "Explorative LV: Loop already has user annotation, "
                           "skipping exploration\n");
      return;
    }

    LoopBlocksRPO RPOT(&L);
    RPOT.perform(LI);
    if (!containsIrreducibleCFG<const BasicBlock *>(RPOT, *LI)) {
      V.push_back(&L);
      return;
    }
  }
  for (Loop *InnerL : L)
    collectSupportedLoops(*InnerL, LI, V);
}

PreservedAnalyses ExplorativeLVPass::run(Function &F,
                                         FunctionAnalysisManager &AM) {
  auto &AC = AM.getResult<AssumptionAnalysis>(F);
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  auto &LI = AM.getResult<LoopAnalysis>(F);
  auto &SE = AM.getResult<ScalarEvolutionAnalysis>(F);
  auto &TLI = AM.getResult<TargetLibraryAnalysis>(F);

  // ~~> LoopVectorize::runImpl

  // The vectorizer requires loops to be in simplified form.
  // Since simplification may add new inner loops, it has to run before the
  // legality and profitability checks. This means running the loop vectorizer
  // will simplify all loops, regardless of whether anything end up being
  // vectorized.
  for (auto &L : LI)
    simplifyLoop(L, &DT, &LI, &SE, &AC, nullptr, false /* PreserveLCSSA */);

  // Build up a worklist of inner-loops to vectorize.
  SmallVector<Loop *, 8> Worklist;
  for (Loop *L : LI) // outermost loops -> we need this recursive approach
    collectSupportedLoops(*L, &LI, Worklist);
  if (Worklist.empty())
    return PreservedAnalyses::all();

  // Initialize pipeline only if there is something to do.
  if (!OptPipeline)
    OptPipeline = new OptPipelineContainer(TM, PTO, PGOOpt, TLI);
  // If this is a code size evaluation, we can re-use the same backend pipeline.
  if (!NullCGPipeline && ExplorativeLV == ExplorativeLVMetric::InstCount)
    NullCGPipeline = new NullCGPipelineContainer(*TM, TLI);

  do {
    Loop *L = Worklist.pop_back_val();
    if (processLoop(F, *L, TLI)) {
      LLVM_DEBUG(dbgs() << "Processing loop failed, letting the AutoVectorizer "
                           "determine VF and IF\n");
      addStringMetadataToLoop(L, "llvm.loop.vectorize.width", 0);
      addStringMetadataToLoop(L, "llvm.loop.interleave.count", 0);
    }
  } while (!Worklist.empty());

  return PreservedAnalyses::all();
}
