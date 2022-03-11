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
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/LoopIterator.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Value.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Transforms/ObjCARC.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <ctime>

using namespace llvm;
#define DEBUG_TYPE "explorative-lv"

namespace llvm {

cl::opt<ExplorativeLVPass::Metric> ExplorativeLV(
    "explorative-lv", cl::Hidden,
    cl::desc(
        "Which metric to use for machine code evaluation in loop exploration"),
    cl::init(ExplorativeLVPass::Metric::Disable),
    cl::values(clEnumValN(ExplorativeLVPass::Metric::Disable, "disable",
                          "Disable explorative loop vectorization"),
               clEnumValN(ExplorativeLVPass::Metric::InstCount, "inst-count",
                          "Count instructions"),
               clEnumValN(ExplorativeLVPass::Metric::MCA, "mca",
                          "Use llvm-mca"),
               clEnumValN(ExplorativeLVPass::Metric::Benchmark, "benchmark",
                          "Benchmark")));

} // namespace llvm

static cl::opt<unsigned>
    ExploreMaxVF("explore-max-vf", cl::Hidden, cl::init(16),
                 cl::desc("The maximum vectorization factor to use for "
                          "explorative loop vectorization"));
static cl::opt<unsigned>
    ExploreMaxIF("explore-max-if", cl::Hidden, cl::init(8),
                 cl::desc("The maximum interleaving factor to use for "
                          "explorative loop vectorization"));

static cl::opt<uint64_t>
    ExploreBenchmarkTicks("explore-benchmark-ticks", cl::Hidden,
                          cl::init(CLOCKS_PER_SEC / 10),
                          cl::desc("In benchmarking mode: use this number of "
                                   "clock ticks for calibration"));
static cl::opt<unsigned> ExploreBenchmarkWarmup(
    "explore-benchmark-warmup", cl::Hidden, cl::init(16),
    cl::desc("In benchmarking mode: how many times to call each loop function "
             "before benchmarking"));
static cl::opt<unsigned>
    ExploreBenchmarkLoopSize("explore-benchmark-loop-size", cl::Hidden,
                             cl::init(32),
                             cl::desc("In benchmarking mode: size (loop "
                                      "function calls) of the benchmark loop"));

static cl::opt<bool>
    ExplorePlain("explore-plain", cl::Hidden, cl::init(false),
                 cl::desc("If loop exploration is enabled, use "
                          "entire function for cost calculation"));

static cl::opt<bool>
    ExploreDivVF("explore-divide-by-vf", cl::Hidden, cl::init(false),
                 cl::desc("If loop exploration is enabled, aggregate "
                          "the cost results by the VF"));

static cl::opt<bool> ExploreDumpModuleIR(
    "explore-dump-module-ir", cl::Hidden, cl::init(false),
    cl::desc("If true, add a PrintModulePass at the beginning of the "
             "explorative optimization pipeline"));

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

namespace {

class LoopModuleBuilder : public ValueMaterializer {
  const Function &OrigFunc;
  Loop &L;

  Module *M = nullptr;
  ValueToValueMapTy VMap;
  ValueMapper Mapper;
  SmallVector<Value *> Inputs;
  SmallVector<Instruction *> Outputs;
  FunctionType *FuncTy;
  SmallVector<Function *> LoopFuncs;

  bool MakeExecutable;

  bool determineIO();
  void buildFuncTy();

public:
  LoopModuleBuilder(const Function &OrigFunc, Loop &L,
                    bool MakeExecutable = false)
      : OrigFunc(OrigFunc), L(L), Mapper(VMap, RF_None, nullptr, this),
        MakeExecutable(MakeExecutable){};
  virtual ~LoopModuleBuilder() { delete M; }

  Value *materialize(Value *V) override;

  Module *getModule();

  void buildMainFuncBody();

  Function *buildLoopFunc(unsigned VF, unsigned IF);
  unsigned getNumLoopFuncs() { return LoopFuncs.size(); }
  void clearLoopFuncs() {
    for (Function *LoopFunc : LoopFuncs) {
      LoopFunc->eraseFromParent();
    }
    LoopFuncs.clear();
  }
};

bool LoopModuleBuilder::determineIO() {
  // The core idea is to generate a function containing the basic blocks of the
  // loop.  Values used inside the loop and defined outside are parameters to
  // the function.  For each values defined inside the loop and used outside we
  // generate an output parameter to produce an observable side-effect such that
  // the code is not optimized away.

  SmallPtrSet<Value *, 32> ArgSet;

  for (BasicBlock *BB : L.getBlocks()) {
    for (Instruction &I : *BB) {
      if (MakeExecutable && isa<CallBase>(I)) {
        // Dealing with call instructions is hard.  If they have external
        // linkage, the linker would probably fail.  One approach would be to
        // replace them by dummy functions, but for now we refrain from
        // analyzing these.  Inline assembly is also embedded using call
        // instructions, we'd have to make these distinctions here.  However,
        // we allow intrinsics.
        if (!isa<IntrinsicInst>(I))
          return false;

        // TODO: are there intrinsics we should forbid?
      }

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

  return true;
}

Value *LoopModuleBuilder::materialize(Value *V) {
  // Special treatment for mapping global values: we need to copy them from the
  // old module.  This code is inspired by CloneModule, however we use this
  // ValueMaterializer to only clone globals as they are needed.  In case we
  // want to create a signle executable for benchmarking, we can also make sure
  // that there are no references to external symbols.
  const GlobalValue *GVal = dyn_cast<GlobalValue>(V);
  if (!GVal)
    return nullptr;

  // The value should be in the original module
  assert(GVal->getParent() == OrigFunc.getParent() &&
         "Expected GVal to be in original module");

  if (const GlobalVariable *GV = dyn_cast<GlobalVariable>(GVal)) {
    GlobalVariable *NGV = new GlobalVariable(
        *M, GV->getValueType(), GV->isConstant(), GV->getLinkage(), nullptr,
        GV->getName() + ".l", nullptr, GV->getThreadLocalMode(),
        GV->getAddressSpace());

    NGV->copyAttributesFrom(GV);

    SmallVector<std::pair<unsigned, MDNode *>, 1> MDs;
    GV->getAllMetadata(MDs);
    for (auto MD : MDs)
      NGV->addMetadata(MD.first, *Mapper.mapMDNode(*MD.second));

    if (!MakeExecutable)
      return NGV;

    if (!GV->hasInitializer()) {
      NGV->setInitializer(Constant::getNullValue(GV->getValueType()));
      return NGV;
    }

    // Not sure whether the initializer could refer to (the address of) GV ...
    // To be on the safe side, we add the mapping now.
    VMap[GV] = NGV;
    NGV->setInitializer(Mapper.mapConstant(*GV->getInitializer()));

    // Handling COMDATs is probably not needed ...

    return NGV;
  }

  if (const Function *F = dyn_cast<Function>(GVal)) {
    if (!MakeExecutable) {
      Function *NF = Function::Create(cast<FunctionType>(F->getValueType()),
                                      GlobalValue::ExternalLinkage,
                                      F->getName() + ".l", M);
      NF->copyAttributesFrom(F);
      return NF;
    }

    return Constant::getNullValue(F->getType());
  }

  if (const GlobalAlias *GA = dyn_cast<GlobalAlias>(GVal)) {
    // Replace alias by aliasee
    const Constant *Aliasee = GA->getAliasee();
    // FIXME: if the alias somehow pointed to itself, this would probably result
    // in an infinite loop ...
    return Aliasee ? Mapper.mapConstant(*Aliasee) : nullptr;
  }

  // TODO: IFuncs?

  return nullptr;
}

// Builds the loop function type
void LoopModuleBuilder::buildFuncTy() {
  SmallVector<Type *> Params;
  Params.reserve(Inputs.size() + Outputs.size());
  for (Value *V : Inputs)
    Params.push_back(V->getType());
  for (Instruction *I : Outputs)
    Params.push_back(PointerType::getUnqual(I->getType()));

  // Returns void, no varargs
  FuncTy = FunctionType::get(Type::getVoidTy(M->getContext()), Params, false);
}

Module *LoopModuleBuilder::getModule() {
  if (M)
    return M;

  if (!determineIO())
    return nullptr;

  // Create the new module (~~> CloneModule)
  const Module *OrigM = OrigFunc.getParent();
  M = new Module(OrigM->getModuleIdentifier(), OrigM->getContext());
  M->setSourceFileName(OrigM->getSourceFileName());
  M->setDataLayout(OrigM->getDataLayout());
  M->setTargetTriple(OrigM->getTargetTriple());
  if (!MakeExecutable) // Only copy inline ASM if we do not create an executable
    M->setModuleInlineAsm(OrigM->getModuleInlineAsm());
  // Clone named metadata
  for (const NamedMDNode &NMD : OrigM->named_metadata()) {
    NamedMDNode *NewNMD = M->getOrInsertNamedMetadata(NMD.getName());
    for (const MDNode *Operand : NMD.operands())
      NewNMD->addOperand(Mapper.mapMDNode(*Operand));
  }

  buildFuncTy();

  return M;
}

// This method uses a lot of code (and comments) from Utils/CloneFunction.cpp,
// but also from IPO/LoopExtractor.cpp resp. Utils/CodeExtractor.cpp
Function *LoopModuleBuilder::buildLoopFunc(unsigned VF, unsigned IF) {
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

  LLVMContext &C = LoopFunc->getContext();

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

  BasicBlock *BeginBB = BasicBlock::Create(C, "loop_begin", LoopFunc);
  BasicBlock *EndBB = BasicBlock::Create(C, "loop_end", LoopFunc);
  ReturnInst *Ret = ReturnInst::Create(C, EndBB);

  VMap[BeginBB] = BeginBB;
  VMap[EndBB] = EndBB;
  VMap[Ret] = Ret;

  // Copy argument attributes (~~> CloneFunctionInto)
  AttributeList LoopAttrs = LoopFunc->getAttributes();
  // We do not clone function attributes, since the original function might
  // fulfill properties that the new one does not.
  // However, we add a NoInline attribute, otherwise a lot of code might be
  // eliminated in benchmarking mode.
  LoopAttrs = LoopAttrs.addFnAttribute(C, Attribute::NoInline);

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

  LoopFunc->setAttributes(AttributeList::get(
      C, LoopAttrs.getFnAttrs(), LoopAttrs.getRetAttrs(), LoopArgAttrs));

  // From here on we need to deviate from CloneFunctionInto to only clone
  // the loop, not a whole function.

  // We'll need to iterate through the output parameters.  To ease this, create
  // an iterator that we only have to copy.
  Function::arg_iterator OutputArgsBegin = LoopFunc->arg_begin();
  std::advance(OutputArgsBegin, Inputs.size());

  // For performance reasons, we want as many fall-through branches as possible.
  // Hence, we insert the blocks as follows:
  //   1. loop_begin
  //   2. cloned loop blocks
  //   3. store blocks
  //   4. loop_end
  BasicBlock *LoopBBInsertBefore = EndBB;

  // Loop over all of the basic blocks in the loop, cloning them as appropriate.
  for (const BasicBlock *BB : L.getBlocks()) {
    // Create a new basic block
    BasicBlock *NewBB = BasicBlock::Create(C, "", LoopFunc, LoopBBInsertBefore);
    if (BB->hasName())
      NewBB->setName(BB->getName() + ".l");

    // Add basic block mapping.
    VMap[BB] = NewBB;
    // And map block to itself as the remapping step will try to retrieve it.
    VMap[NewBB] = NewBB;

    // Copy instructions
    for (const Instruction &I : *BB) {
      Instruction *NewInst = I.clone();
      if (I.hasName())
        NewInst->setName(I.getName());
      NewBB->getInstList().push_back(NewInst);

      VMap[&I] = NewInst;
      // Need to map each new instruction clone to itself, so VMap doesn't
      // break when later stuff is looking at instructions we generated, not
      // cloned.
      VMap[NewInst] = NewInst;
    }

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

    // For each edge that exits the loop: create store block that stores all
    // live variables in the output parameters.
    if (!L.isLoopExiting(BB))
      continue;

    const Instruction *Term = BB->getTerminator();
    Instruction *NewTerm = NewBB->getTerminator();

    for (unsigned Idx = 0, Num = Term->getNumSuccessors(); Idx < Num; ++Idx) {
      BasicBlock *SuccBB = Term->getSuccessor(Idx);
      if (L.getBlocksSet().contains(SuccBB))
        continue; // Not an exiting edge
      assert(SuccBB->getParent() == &OrigFunc);

      BasicBlock *StoreBB =
          BasicBlock::Create(C, "store_block", LoopFunc, EndBB);
      if (LoopBBInsertBefore == EndBB)
        LoopBBInsertBefore = StoreBB;
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
          StoreInst *Store = new StoreInst(LCSSAPhi, OutputArg, StoreBB);
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

  BasicBlock *NewLoopHeader =
      LoopFunc->getBasicBlockList().getNextNode(*BeginBB);
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
    LoopFunc->addMetadata(MD.first, *Mapper.mapMDNode(*MD.second));
  }

  // Loop over all of the instructions in the new function, fixing up operand
  // references as we go. This uses VMap to do all the hard work.
  for (BasicBlock &BB : *LoopFunc) {
    // Loop over all instructions, fixing each one as we find it...
    for (Instruction &II : BB)
      Mapper.remapInstruction(II);
  }

  LoopFuncs.push_back(LoopFunc);
  return LoopFunc;
}

void LoopModuleBuilder::buildMainFuncBody() {
  LLVMContext &C = M->getContext();

  // Some types and values we will use a few times
  Type *I32Ty = Type::getInt32Ty(C);
  Type *I64Ty = Type::getInt32Ty(C);
  Constant *I320 = ConstantInt::get(I32Ty, 0);
  Constant *I321 = ConstantInt::get(I32Ty, 1);
  Constant *I640 = ConstantInt::get(I64Ty, 0);

  assert(LoopFuncs.size() > 0 && "No loop functions generated yet");
  FunctionType *LoopFuncTy = LoopFuncs[0]->getFunctionType();
  PointerType *LoopFuncPtrTy = LoopFuncs[0]->getType();

  // Declare `clock_t clock(void)`
  static_assert(sizeof(clock_t) == 8, "clock_t is assumed to be 64 bit");
  FunctionType *ClockFuncTy = FunctionType::get(I64Ty, false);
  Function *ClockFunc =
      Function::Create(ClockFuncTy, GlobalValue::ExternalLinkage,
                       OrigFunc.getAddressSpace(), "clock", M);

  // Declare `int printf(char *format, ...)`
  FunctionType *PrintfFuncTy =
      FunctionType::get(I32Ty, {Type::getInt8PtrTy(C)}, true);
  Function *PrintfFunc =
      Function::Create(PrintfFuncTy, GlobalValue::ExternalLinkage,
                       OrigFunc.getAddressSpace(), "printf", M);

  // We build a benchmarking function that takes a loop function and an integer
  // n as argument and the function n * ExploreBenchmarkLoopSize times.  This
  // function itself is called by `int main(void)`.
  // To infer a suitable n, we build a calibration function that counts the
  // repitions needed until the elapsed time exceeds some threshold.
  FunctionType *BenchFuncTy =
      FunctionType::get(Type::getVoidTy(C), {LoopFuncPtrTy, I32Ty}, false);
  Function *BenchFunc =
      Function::Create(BenchFuncTy, GlobalValue::PrivateLinkage, "bench", M);
  FunctionType *CalibFuncTy = FunctionType::get(I32Ty, {LoopFuncPtrTy}, false);
  Function *CalibFunc =
      Function::Create(CalibFuncTy, GlobalValue::PrivateLinkage, "calib", M);

  Argument *FuncArg = BenchFunc->getArg(0);
  FuncArg->setName("func");
  Argument *NArg = BenchFunc->getArg(1);
  NArg->setName("n");
  Argument *CFuncArg = CalibFunc->getArg(0);
  CFuncArg->setName("func");

  BasicBlock *StartBB = BasicBlock::Create(C, "bench_start", BenchFunc);
  BasicBlock *CStartBB = BasicBlock::Create(C, "calib_start", CalibFunc);

  SmallVector<Value *> Args;
  SmallVector<Value *> CArgs;
  /* Create arguments */ {
    assert(Inputs.size() <= LoopFuncTy->getNumParams());

    auto *It = LoopFuncTy->param_begin();
    auto *OutputParamsBegin = It;
    std::advance(OutputParamsBegin, Inputs.size());

    // Create input arguments
    for (; It != OutputParamsBegin; ++It) {
      // FIXME: we should use more meaningful values
      Args.push_back(Constant::getNullValue(*It));
      CArgs.push_back(Constant::getNullValue(*It));
    }

    // Alloc locations for output parameters
    for (auto *End = LoopFuncTy->param_end(); It != End; ++It) {
      Args.push_back(
          new AllocaInst(cast<PointerType>(*It)->getPointerElementType(),
                         BenchFunc->getAddressSpace(), "outarg", StartBB));
      CArgs.push_back(
          new AllocaInst(cast<PointerType>(*It)->getPointerElementType(),
                         CalibFunc->getAddressSpace(), "outarg", CStartBB));
    }
  }

  // Warmup (benchmarking only)
  for (unsigned I = 0; I < ExploreBenchmarkWarmup; ++I)
    CallInst::Create(LoopFuncTy, FuncArg, Args, "", StartBB);

  // Start measuring
  CallInst *TStart = CallInst::Create(ClockFuncTy, ClockFunc, "ts", StartBB);
  CallInst *CTStart = CallInst::Create(ClockFuncTy, ClockFunc, "ts", CStartBB);

  // Benchmarking loop
  BasicBlock *LoopBB = BasicBlock::Create(C, "bench_loop", BenchFunc);
  BranchInst::Create(LoopBB, StartBB);
  PHINode *CountPhi = PHINode::Create(I32Ty, 2, "counter", LoopBB);
  CountPhi->addIncoming(I320, StartBB);

  BasicBlock *CLoopBB = BasicBlock::Create(C, "calib_loop", CalibFunc);
  BranchInst::Create(CLoopBB, CStartBB);
  PHINode *CCountPhi = PHINode::Create(I32Ty, 2, "counter", CLoopBB);
  CCountPhi->addIncoming(I320, CStartBB);

  for (unsigned I = 0; I < ExploreBenchmarkLoopSize; ++I) {
    CallInst::Create(LoopFuncTy, FuncArg, Args, "", LoopBB);
    CallInst::Create(LoopFuncTy, CFuncArg, CArgs, "", CLoopBB);
  }

  // Counters and continue/exit conditions
  ICmpInst *ContLoop = new ICmpInst(*LoopBB, CmpInst::ICMP_ULT, CountPhi, NArg);
  BinaryOperator *CountAdd =
      BinaryOperator::CreateAdd(CountPhi, I321, "counter_next", LoopBB);

  CallInst *CTNow = CallInst::Create(ClockFuncTy, ClockFunc, "tn", CLoopBB);
  BinaryOperator *CTDiff =
      BinaryOperator::CreateSub(CTNow, CTStart, "dt", CLoopBB);
  ICmpInst *CContLoop =
      new ICmpInst(*CLoopBB, CmpInst::ICMP_ULT, CTDiff,
                   ConstantInt::get(I64Ty, ExploreBenchmarkTicks));
  BinaryOperator *CCountAdd =
      BinaryOperator::CreateAdd(CCountPhi, I321, "counter_next", CLoopBB);

  // Branch at the end of the loop
  BasicBlock *EvalBB = BasicBlock::Create(C, "bench_eval", BenchFunc);
  BranchInst::Create(LoopBB, EvalBB, ContLoop, LoopBB);
  CountPhi->addIncoming(CountAdd, LoopBB);

  BasicBlock *CRetBB = BasicBlock::Create(C, "calib_ret", CalibFunc);
  BranchInst::Create(CLoopBB, CRetBB, CContLoop, CLoopBB);
  CCountPhi->addIncoming(CCountAdd, CLoopBB);
  // For calibration, we are already done
  ReturnInst::Create(C, CCountAdd, CRetBB);

  // Stop measuring
  CallInst *TEnd = CallInst::Create(ClockFuncTy, ClockFunc, "te", EvalBB);
  BinaryOperator *TDiff = BinaryOperator::CreateSub(TEnd, TStart, "dt", EvalBB);

  // Create format string
  static_assert(sizeof(unsigned long long) == 8,
                "unsigned long long is expected to be 64 bit");
  Constant *FormatStr = ConstantDataArray::getString(C, "%llu\n\0", false);
  GlobalVariable *FormatStrGV =
      new GlobalVariable(*M, FormatStr->getType(), true,
                         GlobalValue::PrivateLinkage, FormatStr, "fstr");
  Constant *FormatStrPtr = ConstantExpr::getInBoundsGetElementPtr(
      FormatStrGV->getValueType(), FormatStrGV,
      ArrayRef<Constant *>{I640, I640});

  // `printf(fstr, dt); return;`
  CallInst::Create(PrintfFuncTy, PrintfFunc, {FormatStrPtr, TDiff}, "", EvalBB);
  ReturnInst::Create(C, EvalBB);

  // Declare & define `int main(void)`
  Function *MainFunc = Function::Create(
      FunctionType::get(I32Ty, false), GlobalValue::ExternalLinkage, "main", M);

  BasicBlock *MainBB = BasicBlock::Create(C, "main", MainFunc);
  CallInst *NRuns =
      CallInst::Create(CalibFuncTy, CalibFunc, {LoopFuncs[0]}, "n", MainBB);
  for (Function *LoopFunc : LoopFuncs)
    CallInst::Create(BenchFuncTy, BenchFunc, {LoopFunc, NRuns}, "", MainBB);

  ReturnInst::Create(C, I320, MainBB);
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

    if (ExploreDumpModuleIR) {
      MPM.addPass(PrintModulePass(outs()));
      MPM.addPass(PB.buildPerModuleDefaultPipeline(OptimizationLevel::O3));
    } else {
      // This is more efficient as the vector containing the passes can be moved
      // and does not have to be copied.
      MPM = PB.buildPerModuleDefaultPipeline(OptimizationLevel::O3);
    }
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

  LoopModuleBuilder Builder(F, L, ExplorativeLV == Metric::Benchmark);
  Module *M = Builder.getModule();
  if (!M)
    return true;

  unsigned LowestCostVF = 1;
  unsigned LowestCostIF = 1;
  if (ExplorativeLV != Metric::Benchmark) {
    unsigned LowestCost = UINT_MAX;
    // Try out which VF works best for this loop
    for (unsigned VF = 1; VF <= ExploreMaxVF; VF *= 2) {
      for (unsigned IF = 1; IF <= ExploreMaxIF; IF *= 2) {
        Function *LoopFunc = Builder.buildLoopFunc(VF, IF);
        OptPipeline->run(*M);

        unsigned Result;
        // Expensive, but: If we are using llvm-mca for evaluation, we need
        // a new backend pipeline for each iteration as the print stream is
        // dead after one use
        if (ExplorativeLV == Metric::MCA) {
          int ASMFD;
          SmallString<32> ASMPath;
          sys::fs::createTemporaryFile("explorative-lv", "s", ASMFD, ASMPath);
          FileRemover ASMRemover(ASMPath);
          {
            raw_fd_ostream OS(ASMFD, true);
            legacy::PassManager CGPipeline;
            initCodeGen(CGPipeline, *TM, TLI, OS, CGFT_AssemblyFile);
            CGPipeline.run(*M);
          }
          // Perform the cost calculation after the pipeline is deleted, to
          // make sure the assembly printer has finished
          Result = performMCACostCalc(ASMPath, TM->getTargetCPU(),
                                      M->getTargetTriple());
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
          NullCGPipeline->PM.run(*M);
          Result = LoopFunc->getContext().getMCInstCount();
        }
        Builder.clearLoopFuncs();

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
    for (unsigned VF = 1; VF <= ExploreMaxVF; VF *= 2) {
      for (unsigned IF = 1; IF <= ExploreMaxIF; IF *= 2)
        Builder.buildLoopFunc(VF, IF);
    }

    OptPipeline->run(*M);

    // Build the main function -- no need to optimize it
    Builder.buildMainFuncBody();

    // Run code generation pipeline
    SmallString<32> ObjectPath;
    int ObjectFD;
    sys::fs::createTemporaryFile("explorative-lv", "o", ObjectFD, ObjectPath);
    FileRemover ObjectRemover(ObjectPath);
    {
      raw_fd_ostream OS(ObjectFD, true);
      legacy::PassManager CGPipeline;
      // CGPipeline.add(createPrintModulePass(outs()));
      initCodeGen(CGPipeline, *TM, TLI, OS, CGFT_ObjectFile);
      CGPipeline.run(*M);
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
    unsigned TimeoutS =
        1 + ((Builder.getNumLoopFuncs() + 1) * ExploreBenchmarkTicks) /
                CLOCKS_PER_SEC;
    dbgs() << "Explorative LV: Benchmarking for up to " << TimeoutS
           << " s ...\n";
    SmallString<32> BenchResPath;
    sys::fs::createTemporaryFile("explorative-lv-out", "txt", BenchResPath);
    FileRemover BenchResRemover(BenchResPath);
    if (sys::ExecuteAndWait(ExecPath, {"explorative-lv"}, None,
                            {StringRef(""), BenchResPath.str(), StringRef("")},
                            TimeoutS) != 0) {
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
    for (unsigned VF = 1; VF <= ExploreMaxVF; VF *= 2) {
      for (unsigned IF = 1; IF <= ExploreMaxIF; IF *= 2) {
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
             << LowestCostVF << " and " << LowestCostIF << "\n");
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
  if (!NullCGPipeline && ExplorativeLV == Metric::InstCount)
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
