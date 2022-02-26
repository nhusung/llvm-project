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
#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/LoopIterator.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/Passes/PassBuilder.h"
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
    if (Blocks.count(I->getParent()))
      return true;
  return false;
}

// This function uses a lot of code (and comments) from Utils/CloneFunction.cpp,
// but also from IPO/LoopExtractor.cpp resp. Utils/CodeExtractor.cpp
static void createNewLoopFunction(std::unique_ptr<Module> *M, Function *OldFunc,
                                  Loop *L, ValueToValueMapTy &VMap,
                                  std::string NewFuncName) {

  // Collect the variables from outside the loop that are being used
  // Collect the arguments we'll have to add, and what maps to them
  SmallVector<Value *> ArgumentVMap;
  std::map<std::string, int> Arguments;
  // We require a type vector to build the function, may as well collect them
  // here
  SmallVector<Type *> ArgumentTypes;
  // FIXME: Pushing the old func args too, even though an Argument is not a
  // Value
  for (Argument &A : OldFunc->args()) {
    ArgumentVMap.push_back(&A);
    ArgumentTypes.push_back(A.getType());
    Arguments[A.getName().str()] = ArgumentTypes.size() - 1;
  }
  SmallVector<Instruction *> Outputs;
  std::set<unsigned int> MetaNodes;

  for (BasicBlock *BB : L->getBlocks()) {
    for (Instruction &I : *BB) {
      unsigned int End = I.getNumOperands();

      CallInst *CI = dyn_cast<CallInst>(&I);
      if (CI && CI->isTailCall())
        // If this is a tail call, the last operand should be its "name",
        // so ignore it
        End--;
      for (unsigned int i = 0; i < End; i++) {
        Value *Op = I.getOperand(i);
        if (Op->getType()->isMetadataTy())
          // Should be handled below
          continue;

        if (isa<BasicBlock>(Op))
          // Ignore blocks, we already dealt with them
          continue;

        std::string Name = Op->getName().str();
        if (Arguments.find(Name) != Arguments.end() || dyn_cast<BasicBlock>(Op))
          continue;
        if (Instruction *OpInst = dyn_cast<Instruction>(Op)) {
          if (!L->getBlocksSet().contains(OpInst->getParent())) {
            ArgumentVMap.push_back(Op);
            ArgumentTypes.push_back(OpInst->getType());
            Arguments[Name] = ArgumentTypes.size() - 1;
          }
        } else if (!dyn_cast<Constant>(Op) || dyn_cast<GlobalVariable>(Op)) {
          // Things that are global, or function arguments used by this loop are
          // not caught by the above instruction thing, but must still be fixed
          ArgumentVMap.push_back(Op);
          ArgumentTypes.push_back(Op->getType());
          Arguments[Name] = ArgumentTypes.size() - 1;
        }
      }
      // Check whether the instruction is being used outside the loop; if
      // yes, we'll add it to the arguments so LLVM believes us that
      // this code shouldn't be optimized away
      for (User *U : I.users())
        if (!definedInRegion(L->getBlocksSet(), U)) {
          Outputs.push_back(&I);
          break;
        }
    }
  }

  // This function returns void, outputs via stores in function arguments.
  Type *RetTy = Type::getVoidTy(OldFunc->getContext());

  for (Value *V : Outputs)
    ArgumentTypes.push_back(PointerType::getUnqual(V->getType()));

  Function *LoopFunction = Function::Create(
      FunctionType::get(RetTy, ArgumentTypes, false), OldFunc->getLinkage(),
      OldFunc->getAddressSpace(), NewFuncName, M->get());

  // Introduce empty label at the beginning of the function, and at the
  // end of the function, so we can refer to "before" and "after"
  IRBuilder<> Builder(LoopFunction->getContext());
  BasicBlock *Begin = BasicBlock::Create(LoopFunction->getContext(),
                                         "loop_begin", LoopFunction);
  Builder.SetInsertPoint(Begin);
  VMap[Begin] = Begin;

  // Map arguments of old function to arguments of new function
  for (unsigned int i = 0; i < OldFunc->arg_size(); i++) {
    VMap[OldFunc->getArg(i)] = LoopFunction->getArg(i);
  }
  // Insert the generated function arguments into the VMap
  for (unsigned int i = OldFunc->arg_size(); i < ArgumentVMap.size(); i++) {
    Value *V = ArgumentVMap[i];
    VMap[V] = LoopFunction->getArg(Arguments[V->getName().str()]);
  }

  AttributeList NewAttrs = LoopFunction->getAttributes();
  LoopFunction->setAttributes(NewAttrs);

  SmallVector<AttributeSet, 4> NewArgAttrs(LoopFunction->arg_size());
  AttributeList OldAttrs = OldFunc->getAttributes();

  // Clone any argument attributes that are present in the VMap.
  for (const Argument &OldArg : OldFunc->args()) {
    if (!VMap[&OldArg])
      continue;
    if (Argument *NewArg = dyn_cast<Argument>(VMap[&OldArg])) {
      NewArgAttrs[NewArg->getArgNo()] =
          OldAttrs.getParamAttrs(OldArg.getArgNo());
    }
  }

  LoopFunction->setAttributes(
      AttributeList::get(LoopFunction->getContext(),
                         NewAttrs.getFnAttrs(),  // OldAttrs.getFnAttributes(),
                         NewAttrs.getRetAttrs(), // OldAttrs.getRetAttributes(),
                         NewArgAttrs));

  // When we remap instructions within the same module, we want to avoid
  // duplicating inlined DISubprograms, so record all subprograms we find as we
  // duplicate instructions and then freeze them in the MD map. We also record
  // information about dbg.value and dbg.declare to avoid duplicating the
  // types.
  DebugInfoFinder DIFinder;

  DISubprogram *SP = OldFunc->getSubprogram();
  if (SP) {
    // Add mappings for some DebugInfo nodes that we don't want duplicated
    // even if they're distinct.
    auto &MD = VMap.MD();
    MD[SP->getUnit()].reset(SP->getUnit());
    MD[SP->getType()].reset(SP->getType());
    MD[SP->getFile()].reset(SP->getFile());
    // If we're not cloning into the same module, no need to clone the
    // subprogram
    MD[SP].reset(SP);
  }

  // From here on we need to deviate from CloneFunctionInto to only clone
  // the loop, not a whole function.

  // Loop over all of the basic blocks in the loop, cloning them as
  // appropriate.
  bool first = true;
  for (const BasicBlock *BB : L->getBlocks()) {

    // Create a new basic block and copy instructions into it!
    BasicBlock *CBB =
        CloneBasicBlock(BB, VMap, "_suffix", LoopFunction, nullptr, &DIFinder);

    if (first) {
      Builder.SetInsertPoint(Begin);
      BranchInst *BI = Builder.CreateBr(CBB);
      VMap[BI] = BI;
      first = false;
    }

    // Add basic block mapping.
    VMap[BB] = CBB;
    // And map block to itself because the remapping step
    // will try to retrieve it
    VMap[CBB] = CBB;

    // Need to map each new instruction clone to itself, so VMap doesn't
    // break when later stuff is looking at instructions we generated, not
    // cloned
    for (Instruction &I : *CBB)
      VMap[&I] = &I;

    // It is only legal to clone a function if a block address within that
    // function is never referenced outside of the function.  Given that, we
    // want to map block addresses from the old function to block addresses in
    // the clone. (This is different from the generic ValueMapper
    // implementation, which generates an invalid blockaddress when
    // cloning a function.)
    if (BB->hasAddressTaken()) {
      Constant *OldBBAddr = BlockAddress::get(const_cast<Function *>(OldFunc),
                                              const_cast<BasicBlock *>(BB));
      VMap[OldBBAddr] = BlockAddress::get(LoopFunction, CBB);
    }

    // Change any returns to return void
    if (ReturnInst *RI = dyn_cast<ReturnInst>(CBB->getTerminator())) {
      ReturnInst::Create(LoopFunction->getContext(), nullptr, CBB);
      RI->removeFromParent();
    }
  }

  for (DISubprogram *ISP : DIFinder.subprograms())
    if (ISP != SP)
      VMap.MD()[ISP].reset(ISP);

  for (DICompileUnit *CU : DIFinder.compile_units())
    VMap.MD()[CU].reset(CU);

  for (DIType *Type : DIFinder.types())
    VMap.MD()[Type].reset(Type);

  BasicBlock *End = BasicBlock::Create(LoopFunction->getContext(), "loop_end",
                                       LoopFunction, nullptr);
  VMap[End] = End;

  // Following LoopExtractor's example, use the arguments to store values
  // that are used outside of the loop
  Function::arg_iterator LoopFuncArgs = LoopFunction->arg_begin();
  std::advance(LoopFuncArgs, OldFunc->arg_size());
  unsigned int NumInputs = ArgumentVMap.size();
  SmallVector<BasicBlock *> ExitingBlocks;
  L->getExitingBlocks(ExitingBlocks);
  for (unsigned int j = 0;; j++) {
    if (j == LoopFunction->arg_size())
      break;
    if (j < NumInputs) {
    } else {
      // First make sure VMap knows this func arg
      Value *FuncArg = LoopFunction->getArg(j);
      VMap[FuncArg] = FuncArg;
      Instruction *I = Outputs[j - NumInputs];
      assert(I && "If this is no instruction we did something wrong");

      // Check the exiting blocks to find one that is reachable from the
      // instruction, such that we can build an artificial successor that
      // can do the store for us (preserving LCSSA form using a phi and
      // preserving reductions because it's outside the loop)
      // We do the analysis in the original loop, then use VMap to get
      // the instructions and blocks we'll want to change
      for (BasicBlock *BB : ExitingBlocks) {
        if (isPotentiallyReachable(BB->getTerminator(), I)) {
          BasicBlock *TargetBlock = dyn_cast<BasicBlock>(VMap[BB]);
          BranchInst *TargetBranch =
              dyn_cast<BranchInst>(TargetBlock->getTerminator());
          if (!TargetBranch)
            // It could e.g. be a return, let's not mess with that
            continue;
          // Find the (one) successor that leaves the loop, that we have
          // to change to our new block
          bool Done = false;
          for (unsigned int i = 0; i < TargetBranch->getNumSuccessors(); i++) {
            // All blocks inside the loop we cloned, and set a VMap entry
            // for them, so if we find one that is not in VMap it's not in the
            // loop
            if (!VMap[TargetBranch->getSuccessor(i)]) {
              Instruction *TargetInstruction = dyn_cast<Instruction>(VMap[I]);
              BasicBlock *PhiBlock = BasicBlock::Create(
                  LoopFunction->getContext(), "store_block", LoopFunction, End);
              VMap[PhiBlock] = PhiBlock;

              TargetBranch->setSuccessor(i, PhiBlock);
              PHINode *LcssaPhi = PHINode::Create(TargetInstruction->getType(),
                                                  1, "lcssa_phi", PhiBlock);
              LcssaPhi->addIncoming(TargetInstruction, TargetBlock);
              Instruction *NewStore =
                  new StoreInst(LcssaPhi, FuncArg, PhiBlock);
              BranchInst *B = BranchInst::Create(End, PhiBlock);
              // Add all the new instructions to VMap
              VMap[LcssaPhi] = LcssaPhi;
              VMap[NewStore] = NewStore;
              VMap[B] = B;

              Done = true;
              break;
            }
          }
          if (Done)
            break;
        }
      }
    }
    ++LoopFuncArgs;
  }
  Builder.SetInsertPoint(End);
  Builder.CreateRetVoid();

  const auto RemapFlag = RF_None;
  // Duplicate the metadata that is attached to the cloned function.
  // Subprograms/CUs/types that were already mapped to themselves won't be
  // duplicated.
  SmallVector<std::pair<unsigned, MDNode *>, 1> MDs;
  OldFunc->getAllMetadata(MDs);
  for (auto MD : MDs) {
    LoopFunction->addMetadata(MD.first,
                              *MapMetadata(MD.second, VMap, RemapFlag));
  }

  // Loop over all of the instructions in the new function, fixing up operand
  // references as we go. This uses VMap to do all the hard work.
  for (BasicBlock &BB : *LoopFunction)
    // Loop over all instructions, fixing each one as we find it...
    for (Instruction &II : BB) {

      // Special treatment for instructions that might refer to blocks
      // we don't have in our VMap
      // This is a simplification, but Phis need to be reachable from the
      // incoming block, so take Begin, the others can simply default to End
      PHINode *P = dyn_cast<PHINode>(&II);
      if (P)
        for (unsigned int i = 0; i < P->getNumIncomingValues(); i++) {
          BasicBlock *SomeBlock = P->getIncomingBlock(i);
          if (!VMap[SomeBlock])
            P->setIncomingBlock(i, Begin);
        }
      BranchInst *B = dyn_cast<BranchInst>(&II);
      if (B)
        for (unsigned int i = 0; i < B->getNumSuccessors(); i++) {
          BasicBlock *SomeBlock = B->getSuccessor(i);
          if (!VMap[SomeBlock])
            B->setSuccessor(i, End);
        }
      SwitchInst *S = dyn_cast<SwitchInst>(&II);
      if (S) {
        BasicBlock *SomeBlock = S->getDefaultDest();
        if (!VMap[SomeBlock])
          S->setDefaultDest(End);
        for (unsigned int i = 0; i < S->getNumSuccessors(); i++) {
          BasicBlock *SomeBlock = S->getSuccessor(i);
          if (!VMap[SomeBlock])
            S->setSuccessor(i, End);
        }
      }
      RemapInstruction(&II, VMap, RemapFlag);
    }
}

// This function uses code of Utils/CloneModule.cpp
std::unique_ptr<Module> NewMakeLoopOnlyModule(Function *F, Loop *L,
                                              std::string NewFuncName) {

  ValueToValueMapTy VMap;
  Module *M = F->getParent();
  // FIXME: Not entirely sure whether this should be true or false.
  //        If false is correct, that would save us some lines of code.
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
  for (Module::const_global_iterator I = M->global_begin(), E = M->global_end();
       I != E; ++I) { // TODO: Only copy globals used in loop
    GlobalVariable *GV = new GlobalVariable(
        *New, I->getValueType(), I->isConstant(), I->getLinkage(),
        (Constant *)nullptr, I->getName(), (GlobalVariable *)nullptr,
        I->getThreadLocalMode(), I->getType()->getAddressSpace());
    GV->copyAttributesFrom(&*I);
    VMap[&*I] = GV;
  }

  // Loop over the functions in the module, making external functions as before
  for (const Function &I : M->functions()) {
    Function *NF = Function::Create(cast<FunctionType>(I.getValueType()),
                                    I.getLinkage(), I.getName(), New.get());
    NF->copyAttributesFrom(&I);
    VMap[&I] = NF;
  }

  // Loop over the aliases in the module
  for (Module::const_alias_iterator I = M->alias_begin(), E = M->alias_end();
       I != E; ++I) {
    // An alias cannot act as an external reference, so we need to create
    // either a function or a global variable depending on the value type.
    // FIXME: Once pointee types are gone we can probably pick one or the
    // other.
    GlobalValue *GV;
    if (I->getValueType()->isFunctionTy())
      GV = Function::Create(cast<FunctionType>(I->getValueType()),
                            GlobalValue::ExternalLinkage, I->getAddressSpace(),
                            I->getName(), New.get());
    else
      GV = new GlobalVariable(*New, I->getValueType(), false,
                              GlobalValue::ExternalLinkage, nullptr,
                              I->getName(), nullptr, I->getThreadLocalMode(),
                              I->getType()->getAddressSpace());
    VMap[&*I] = GV;
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
  createNewLoopFunction(&New, F, L, VMap, NewFuncName);

  // And named metadata....
  for (Module::const_named_metadata_iterator I = M->named_metadata_begin(),
                                             E = M->named_metadata_end();
       I != E; ++I) {
    const NamedMDNode &NMD = *I;
    NamedMDNode *NewNMD = New->getOrInsertNamedMetadata(NMD.getName());
    for (unsigned i = 0, e = NMD.getNumOperands(); i != e; ++i)
      NewNMD->addOperand(MapMetadata(NMD.getOperand(i), VMap));
  }

  return New;
}

// Executes the command line and returns the total output it sees
// Based on a Stackoverflow solution here:
// https://stackoverflow.com/questions/478898/how-do-i-execute-a-command
// -and-get-the-output-of-the-command-within-c-using-po
static std::string exec(const char *Cmd) {
  std::array<char, 128> Buffer;
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
static int performMCACostCalc(std::string FileName, std::string TargetCPU,
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
  int Result = std::atoi(Output.c_str());

  // And last, remove the tmp file
  std::string Remove = "rm " + FileName;
  exec(Remove.c_str());

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

    int LowestCost = -1;
    unsigned LowestCostVF = 1;
    unsigned LowestCostIF = 1;
    // Try out which VF works best for this loop
    for (unsigned VF = 1; VF <= 16; VF *= 2) {
      for (unsigned IF = 1; IF <= 8; IF *= 2) {

        // Place the loop as a function in its own private module
        std::string newFuncName = F.getName().str() + "_vf" +
                                  std::to_string(VF) + "-if" +
                                  std::to_string(IF);
        // Force the loop to the given VF and IF (will be cloned together with
        // loop)
        addStringMetadataToLoop(L, "llvm.loop.vectorize.width", VF);
        addStringMetadataToLoop(L, "llvm.loop.interleave.count", IF);
        std::unique_ptr<Module> LoopModule =
            NewMakeLoopOnlyModule(&F, L, newFuncName);

        // This line "fixes" a bug where sometimes, starting the PM run will
        // trigger an invalid invalidation...
        MAM.clear();

        // Run the middle end pipeline
        ExplorativeMPM.run(*LoopModule.get(), MAM);

        int Result;
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
          Function *LoopFunc = LoopModule->getFunction(newFuncName);
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
          Result = LoopModule->getFunction(newFuncName)
                       ->getContext()
                       .getMCInstCount();
        }

        if (ExploreDivVF)
          Result /= VF;

        LLVM_DEBUG(dbgs() << "\nExplorative LV: Received result " << Result
                          << " from cost module for VF " << VF << ", IF " << IF
                          << "\n\n");
        if (Result == -1)
          continue;

        if (LowestCost == -1 || Result < LowestCost ||
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
