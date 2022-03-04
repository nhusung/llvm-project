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
#include "llvm/Transforms/ObjCARC.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;
#define DEBUG_TYPE "explorative-lv"

namespace llvm {

cl::opt<bool> EnableExplorativeLV(
    "enable-explorative-lv", cl::Hidden, cl::init(false),
    cl::desc("Enable loop vectorization by exploratively determining the VF"));

// Set to true to use llvm-mca for machine code evaluation, to false to use
// the resulting code's size (default: code size)
cl::opt<bool> ExplorativeLVmca(
    "explore-with-mca", cl::Hidden, cl::init(false),
    cl::desc("Use llvm-mca for machine code evaluation in loop exploration"));

} // namespace llvm

static cl::opt<bool>
    ExplorePlain("explore-plain", cl::Hidden, cl::init(false),
                 cl::desc("If loop exploration is enabled, use "
                          "entire function for cost calculation"));

static cl::opt<bool>
    ExploreDivVF("explore-divide-by-vf", cl::Hidden, cl::init(false),
                 cl::desc("If loop exploration is enabled, aggregate "
                          "the cost results by the VF"));

// Function copied and reduced from LoopVectorize.cpp
static void collectSupportedLoops(Loop &L, LoopInfo *LI,
                                  SmallVectorImpl<Loop *> &V) {
  // Collect inner loops and outer loops without irreducible control flow. For
  // now, only collect outer loops that have explicit vectorization hints. If we
  // are stress testing the VPlan H-CFG construction, we collect the outermost
  // loop of every loop nest.
  if (L.isInnermost()) {
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

// This function uses a lot of code (and comments) from Utils/CloneFunction.cpp,
// but also from IPO/LoopExtractor.cpp resp. Utils/CodeExtractor.cpp
static void createNewLoopFunction(Module *M, const Function *OldFunc, Loop *L,
                                  ValueToValueMapTy &VMap,
                                  const StringRef NewFuncName) {
  // The core idea is to generate a function containing the basic blocks of the
  // loop.  Values used inside the loop and defined outside are parameters to
  // the function.  For each values defined inside the loop and used outside we
  // generate an output parameter to produce an observable side-effect such that
  // the code is not optimized away.

  SmallVector<Value *> ArgVMap;
  SmallPtrSet<Value *, 32> ArgSet;
  SmallVector<Instruction *> Outputs;

  for (BasicBlock *BB : L->getBlocks()) {
    for (Instruction &I : *BB) {
      // Used inside loop and defined outside?
      for (Value *Op : I.operand_values()) {
        Instruction *OpInst = dyn_cast<Instruction>(Op);
        if (OpInst) {
          if (L->getBlocksSet().contains(OpInst->getParent()))
            continue;
        } else if (!isa<Argument>(Op) || ArgSet.contains(Op)) {
          continue;
        }
        ArgSet.insert(Op);
        ArgVMap.push_back(Op);
      }

      // Defined inside loop and used outside?
      for (User *U : I.users())
        if (!definedInRegion(L->getBlocksSet(), U)) {
          Outputs.push_back(&I);
          break;
        }
    }
  }

  SmallVector<Type *> ArgTys;
  ArgTys.reserve(ArgVMap.size() + Outputs.size());
  for (Value *V : ArgVMap)
    ArgTys.push_back(V->getType());
  for (Instruction *I : Outputs)
    ArgTys.push_back(PointerType::getUnqual(I->getType()));

  // This function returns void, no varargs
  FunctionType *LoopFuncTy =
      FunctionType::get(Type::getVoidTy(M->getContext()), ArgTys, false);

  Function *LoopFunc =
      Function::Create(LoopFuncTy, GlobalValue::ExternalLinkage,
                       OldFunc->getAddressSpace(), NewFuncName, M);

  /* Insert the generated function arguments into the VMap, add names */ {
    auto *ArgPtr = LoopFunc->arg_begin();
    for (Value *OldV : ArgVMap) {
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

  SmallVector<AttributeSet, 4> LoopArgAttrs(LoopFunc->arg_size());
  AttributeList OldAttrs = OldFunc->getAttributes();
  for (const Argument &OldArg : OldFunc->args()) {
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
  for (const BasicBlock *BB : L->getBlocks()) {

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
      Constant *OldBBAddr = BlockAddress::get(const_cast<Function *>(OldFunc),
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
  std::advance(OutputArgsBegin, ArgVMap.size());
  for (BasicBlock *BB : L->blocks()) {
    if (!L->isLoopExiting(BB))
      continue;

    Instruction *Term = BB->getTerminator();
    BasicBlock *NewBB = dyn_cast<BasicBlock>(VMap[BB]);
    assert(NewBB && "block mapping missing");
    Instruction *NewTerm = NewBB->getTerminator();

    for (unsigned Idx = 0, Num = Term->getNumSuccessors(); Idx < Num; ++Idx) {
      BasicBlock *SuccBB = Term->getSuccessor(Idx);
      if (L->getBlocksSet().contains(SuccBB))
        continue; // Not an exiting edge
      assert(SuccBB->getParent() == OldFunc);

      BasicBlock *StoreBB = BasicBlock::Create(LoopFunc->getContext(),
                                               "store_block", LoopFunc, EndBB);
      VMap[StoreBB] = StoreBB;
      NewTerm->setSuccessor(Idx, StoreBB);

      Function::arg_iterator OutputArg = OutputArgsBegin;
      for (Instruction *I : Outputs) {
        assert(I->getParent()->getParent() == OldFunc);
        for (User *U : I->users()) {
          Instruction *UI = dyn_cast<Instruction>(U);
          // A user might be in the new function as we have not remapped the
          // operands yet.  Just ignore these users.
          if (!UI || UI->getFunction() != OldFunc ||
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
  OldFunc->getAllMetadata(MDs);
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
}

// This function uses code of Utils/CloneModule.cpp
// We do not clone any definitions but declarations only.
static std::unique_ptr<Module>
newMakeLoopOnlyModule(Function *F, Loop *L, const StringRef NewFuncName) {
  ValueToValueMapTy VMap;
  Module *M = F->getParent();

  // First off, we need to create the new module.
  std::unique_ptr<Module> New =
      std::make_unique<Module>(M->getModuleIdentifier(), M->getContext());
  New->setSourceFileName(M->getSourceFileName());
  New->setDataLayout(M->getDataLayout());
  New->setTargetTriple(M->getTargetTriple());
  New->setModuleInlineAsm(M->getModuleInlineAsm());

  // Loop over all of the global variables, making corresponding globals in the
  // new module.  Here we add them to the VMap and to the new Module.  We
  // don't worry about attributes or initializers, they will come later.
  //
  for (const GlobalVariable &I : M->globals()) {
    // TODO: Only copy globals used in loop
    GlobalVariable *GV = new GlobalVariable(
        *New, I.getValueType(), I.isConstant(), I.getLinkage(),
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
  for (const Function &I : M->functions()) {
    Function *NF = Function::Create(cast<FunctionType>(I.getValueType()),
                                    I.getLinkage(), I.getName(), New.get());
    NF->copyAttributesFrom(&I);
    VMap[&I] = NF;
  }

  // Loop over the aliases in the module
  for (const GlobalAlias &I : M->aliases()) {
    // An alias cannot act as an external reference, so we need to create
    // either a function or a global variable depending on the value type.
    // FIXME: Once pointee types are gone we can probably pick one or the
    // other.
    GlobalValue *GV;
    if (I.getValueType()->isFunctionTy())
      GV = Function::Create(cast<FunctionType>(I.getValueType()),
                            GlobalValue::ExternalLinkage, I.getAddressSpace(),
                            I.getName(), New.get());
    else
      GV = new GlobalVariable(*New, I.getValueType(), false,
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
  for (const GlobalVariable &G : M->globals()) {
    GlobalVariable *GV = cast<GlobalVariable>(VMap[&G]);

    SmallVector<std::pair<unsigned, MDNode *>, 1> MDs;
    G.getAllMetadata(MDs);
    for (auto MD : MDs)
      GV->addMetadata(MD.first, *MapMetadata(MD.second, VMap));

    if (G.isDeclaration())
      continue;

    // Skip after setting the correct linkage for an external reference.
    GV->setLinkage(GlobalValue::ExternalLinkage);
  }

  // Here the original clones functions, we do it a little different
  createNewLoopFunction(New.get(), F, L, VMap, NewFuncName);

  // And named metadata....
  for (const NamedMDNode &NMD : M->named_metadata()) {
    NamedMDNode *NewNMD = New->getOrInsertNamedMetadata(NMD.getName());
    for (const MDNode *Operand : NMD.operands())
      NewNMD->addOperand(MapMetadata(Operand, VMap));
  }

  return New;
}

// Executes the command line and returns the total output it sees
// Based on a Stackoverflow solution here:
// https://stackoverflow.com/questions/478898/how-do-i-execute-a-command
// -and-get-the-output-of-the-command-within-c-using-po
static std::string exec(const char *Cmd) {
  std::array<char, 4096> Buffer;
  std::string Res;
  std::unique_ptr<FILE, decltype(&pclose)> Pipe(popen(Cmd, "r"), pclose);
  assert(Pipe &&
         "MachineCodeExplorer cannot open a command line to execute llvm-mca");
  while (fgets(Buffer.data(), Buffer.size(), Pipe.get()) != nullptr) {
    Res += Buffer.data();
  }
  return Res;
}

// The more complex approach, using the runtime estimation mechanisms
// already present in LLVM: llvm-mca
// (To be called from the pass that performs the exploration)
// FIXME: This only works when the compilation is executed from within
// the build folder of llvm-project
static unsigned performMCACostCalc(std::string FileName, std::string TargetCPU,
                                   std::string TargetTriple,
                                   raw_pwrite_stream *OutStream) {
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

  // Build the most specific command line we can generate from
  // the info in TM
  // First, let's wait for a second to make sure printing the file has finished
  std::string CommandMCA = "bin/llvm-mca";
  CommandMCA.append(" --mcpu=" + TargetCPU);
  CommandMCA.append(" --mtriple=" + TargetTriple);
  CommandMCA.append(" --instruction-info=false");
  CommandMCA.append(" --resource-pressure=false ");
  CommandMCA.append(FileName);
  // Immediately reduce the output to the part we're interested in
  CommandMCA.append(" | grep 'Total Cycles' | cut -f2- -d:");

  // Retrieve result
  /*Command.append(" && ");
  Command.append(CommandMCA);
  std::string Output = exec(Command.c_str());*/
  std::string Output = exec(CommandMCA.c_str());

  // Transform result to a number
  unsigned Result = std::stoi(Output);

  // And last, remove the tmp file
  remove(FileName.c_str());

  return Result;
}

// Add to the given pass manager everything we need to simulate our backend
// pipeline
raw_pwrite_stream *ExplorativeLVPass::init(legacy::PassManager &PM, int VecOps,
                                           std::string AssemblyFileName) {
  /********** this is basically BackendUtil::createTLII **********/
  // Add LibraryInfo.
  llvm::Triple TargetTriple(TM->getTargetTriple());
  TargetLibraryInfoImpl *TLII = new TargetLibraryInfoImpl(TargetTriple);
  switch (VecOps) {
  case 1:
    TLII->addVectorizableFunctionsFromVecLib(TargetLibraryInfoImpl::Accelerate);
    break;
  case 2:
    switch (TargetTriple.getArch()) {
    default:
      break;
    case llvm::Triple::x86_64:
      TLII->addVectorizableFunctionsFromVecLib(
          TargetLibraryInfoImpl::LIBMVEC_X86);
      break;
    }
    break;
  case 3:
    TLII->addVectorizableFunctionsFromVecLib(TargetLibraryInfoImpl::MASSV);
    break;
  case 4:
    TLII->addVectorizableFunctionsFromVecLib(TargetLibraryInfoImpl::SVML);
    break;
  default:
    break;
  }
  /**************************************************/
  // Now fill the pipeline.
  std::error_code EC;
  std::unique_ptr<TargetLibraryInfoImpl> TLIIptr(TLII);
  PM.add(new TargetLibraryInfoWrapperPass(*TLIIptr));
  PM.add(createObjCARCContractPass());

  raw_pwrite_stream *MyIntermediateOutput;
  if (AssemblyFileName != "") {
    MyIntermediateOutput = new raw_fd_ostream(AssemblyFileName, EC);
    TM->addPassesToEmitFile(PM, *MyIntermediateOutput, nullptr,
                            CGFT_AssemblyFile, false);
  } else {
    MyIntermediateOutput = new raw_fd_ostream("trash.tmp", EC);
    TM->addPassesToEmitFile(PM, *MyIntermediateOutput, nullptr, CGFT_Null,
                            false);
  }
  return MyIntermediateOutput;
}

PreservedAnalyses ExplorativeLVPass::run(Function &F,
                                         FunctionAnalysisManager &AM) {

  int CodeGenOpts = AM.getCodeGenOpts();
  // Don't do anything if we are not supposed to do anything, or if
  // this is part of a running exploration
  // (which is both true if the CodeGenOpts haven't been set)
  if (CodeGenOpts == -1 || !EnableExplorativeLV)
    return PreservedAnalyses::all();

  auto &AC = AM.getResult<AssumptionAnalysis>(F);
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  auto &LI = AM.getResult<LoopAnalysis>(F);
  auto &SE = AM.getResult<ScalarEvolutionAnalysis>(F);
  Module *Original = F.getParent();

  // Building our own pipeline

  /******* Code from BackendUtil::RunOptimizationPipeline ******/
  PassInstrumentationCallbacks PIC;
  StandardInstrumentations SI(DebugLogging);
  SI.registerCallbacks(PIC);

  // Create a new PassBuilder which is the basis of a pipeline
  // TODO: I'm not 100% sure that it's wise to use the original TM here
  PassBuilder NewPB(TM, PTO, PGOOpt, &PIC);
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;
  // Register the AA manager first so that our version is the one used.
  FAM.registerPass([&] { return NewPB.buildDefaultAAPipeline(); });

  // Register the target library analysis directly and give it a customized
  // preset TLI.
  Triple TargetTriple(Original->getTargetTriple());
  std::unique_ptr<TargetLibraryInfoImpl> TLII(
      new TargetLibraryInfoImpl(TargetTriple));
  FAM.registerPass([&] { return TargetLibraryAnalysis(*TLII); });

  // Register all the basic analyses with the managers.
  NewPB.registerModuleAnalyses(MAM);
  NewPB.registerCGSCCAnalyses(CGAM);
  NewPB.registerFunctionAnalyses(FAM);
  NewPB.registerLoopAnalyses(LAM);
  NewPB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  ModulePassManager ExplorativeMPM =
      NewPB.buildPerModuleDefaultPipeline(OptimizationLevel::O3);

  /**************************************************/

  legacy::PassManager NewCodeGenPasses;
  // If this is a code size evaluation, we can re-use the same backend
  // pipeline for each iteration.
  if (!ExplorativeLVmca)
    init(NewCodeGenPasses, CodeGenOpts, "");

  /********* Code from LoopVectorize::runImpl *******/

  // The vectorizer requires loops to be in simplified form.
  // Since simplification may add new inner loops, it has to run before the
  // legality and profitability checks. This means running the loop vectorizer
  // will simplify all loops, regardless of whether anything end up being
  // vectorized.
  for (auto &L : LI)
    simplifyLoop(L, &DT, &LI, &SE, &AC, nullptr, false /* PreserveLCSSA */);

  // Build up a worklist of inner-loops to vectorize.
  SmallVector<Loop *, 8> Worklist;

  for (Loop *L : LI)
    collectSupportedLoops(*L, &LI, Worklist);

  /**************************************************/

  while (!Worklist.empty()) {
    Loop *L = Worklist.pop_back_val();
    if (findStringMetadataForLoop(L, "llvm.loop.vectorize.width")) {
      LLVM_DEBUG(dbgs() << "Explorative LV: Loop already has user annotation, "
                        << "skipping exploration\n\n");
      continue;
    }

    if (L->getBlocksSet().empty())
      continue;

    unsigned LowestCost = UINT_MAX;
    unsigned LowestCostVF = 1;
    unsigned LowestCostIF = 1;
    // Try out which VF works best for this loop
    for (unsigned VF = 1; VF <= 16; VF *= 2) {
      for (unsigned IF = 1; IF <= 8; IF *= 2) {

        // Place the loop as a function in its own private module
        const std::string NewFuncName = F.getName().str() + ".vf" +
                                        std::to_string(VF) + ".if" +
                                        std::to_string(IF);
        // Force the loop to the given VF and IF (will be cloned together with
        // loop)
        addStringMetadataToLoop(L, "llvm.loop.vectorize.width", VF);
        addStringMetadataToLoop(L, "llvm.loop.interleave.count", IF);
        std::unique_ptr<Module> LoopModule =
            newMakeLoopOnlyModule(&F, L, NewFuncName);

        // This line "fixes" a bug where sometimes, starting the PM run will
        // trigger an invalid invalidation...
        MAM.clear();

        // Run the middle end pipeline
        ExplorativeMPM.run(*LoopModule.get(), MAM);

        unsigned Result;
        // Expensive, but: If we are using llvm-mca for evaluation, we need
        // a new backend pipeline for each iteration as the print stream is
        // dead after one use
        if (ExplorativeLVmca) {
          std::string AssemblyFileName = "ExploredAssembly" +
                                         std::to_string(VF) + "_" +
                                         std::to_string(IF) + ".tmp";
          raw_pwrite_stream *OutStream;
          {
            legacy::PassManager FreshPipeline;
            OutStream = init(FreshPipeline, CodeGenOpts, AssemblyFileName);
            FreshPipeline.run(*LoopModule.get());
            OutStream->flush();
          }
          // Perform the cost calculation after the pipeline is deleted, to
          // make sure the assembly printer has finished
          Result =
              performMCACostCalc(AssemblyFileName, TM->getTargetCPU().str(),
                                 TargetTriple.str(), OutStream);

        } else {
          Function *LoopFunc = LoopModule->getFunction(NewFuncName);
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
          NewCodeGenPasses.run(*LoopModule.get());
          Result = LoopModule->getFunction(NewFuncName)
                       ->getContext()
                       .getMCInstCount();
        }

        if (Result == UINT_MAX) {
          LLVM_DEBUG(dbgs() << "\nExplorative LV: Invalid costs for VF " << VF
                            << ", IF " << IF << "\n\n");
          continue;
        }

        if (ExploreDivVF)
          Result /= VF;

        LLVM_DEBUG(dbgs() << "\nExplorative LV: Received result " << Result
                          << " from cost module for VF " << VF << ", IF " << IF
                          << "\n\n");

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
    // Force the vectorizer to use the VF and IF that showed to have the lowest
    // cost
    LLVM_DEBUG(
        dbgs() << "Explorative LV: Choosing this VF and IF for vectorization: "
               << LowestCostVF << " and " << LowestCostIF << "\n\n");
    addStringMetadataToLoop(L, "llvm.loop.vectorize.width", LowestCostVF);
    addStringMetadataToLoop(L, "llvm.loop.interleave.count", LowestCostIF);
  }

  return PreservedAnalyses::all();
}
