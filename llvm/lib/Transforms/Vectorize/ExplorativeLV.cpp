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
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/ExplorativeLV.h"
#include "llvm/Analysis/LoopIterator.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Value.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
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

cl::opt<ExplorativeLVPass::Metric> XLVMetric(
    "xlv-metric", cl::Hidden,
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

// Used by MachineCodeExplorer
cl::opt<bool> XLVPlain("xlv-plain", cl::Hidden, cl::init(false),
                       cl::desc("If loop exploration is enabled, use "
                                "entire function for cost calculation"));

} // namespace llvm

static cl::opt<unsigned>
    XLVMaxVF("xlv-max-vf", cl::Hidden, cl::init(16),
             cl::desc("The maximum vectorization factor to use for "
                      "explorative loop vectorization"));
static cl::opt<unsigned>
    XLVMaxIF("xlv-max-if", cl::Hidden, cl::init(8),
             cl::desc("The maximum interleaving factor to use for "
                      "explorative loop vectorization"));

static cl::opt<unsigned> XLVDesiredTripCount(
    "xlv-desired-tripcount", cl::Hidden, cl::init(1024),
    cl::desc("In benchmarking mode: try to run the loop with this trip count"));
static cl::opt<uint64_t>
    XLVBenchmarkTicks("xlv-benchmark-ticks", cl::Hidden,
                      cl::init(CLOCKS_PER_SEC / 10),
                      cl::desc("In benchmarking mode: use this number of "
                               "clock ticks for calibration"));
static cl::opt<unsigned> XLVBenchmarkWarmup(
    "xlv-benchmark-warmup", cl::Hidden, cl::init(16),
    cl::desc("In benchmarking mode: how many times to call each loop function "
             "before benchmarking"));
static cl::opt<unsigned>
    XLVBenchmarkLoopSize("xlv-benchmark-loop-size", cl::Hidden, cl::init(32),
                         cl::desc("In benchmarking mode: size (loop "
                                  "function calls) of the benchmark loop"));

static cl::opt<bool>
    XLVDivVF("xlv-divide-by-vf", cl::Hidden, cl::init(false),
             cl::desc("If loop exploration is enabled, aggregate "
                      "the cost results by the VF"));

static cl::opt<bool>
    XLVDumpFuncIR("xlv-dump-func-ir", cl::Hidden, cl::init(false),
                  cl::desc("If true, print the loop function used for "
                           "explorative vectorization to stderr"));
static cl::opt<bool> XLVDumpModuleIR(
    "xlv-dump-module-ir", cl::Hidden, cl::init(false),
    cl::desc("If true, add a PrintModulePass at the beginning of the "
             "explorative codegen pipeline, printing to stdout"));

class ExplorativeLVPass::InputBuilder {
  using ArrayID = unsigned;

  class MemAccessRange {
    friend class InputBuilder;

    int64_t Start;
    int64_t End;

    Type *ElementType;

    // We'll use global variables to allocate memory
    GlobalVariable *GV = nullptr;

  public:
    MemAccessRange(int64_t Start, int64_t End, Type *ElementType)
        : Start(Start), End(End), ElementType(ElementType) {}
  };

  DenseMap<Value *, ConstantInt *> IntInputs;
  DenseMap<Value *, std::pair<ArrayID, int64_t>> MemAccesses;
  SmallVector<MemAccessRange> MemRanges;

public:
  ArrayID addMemAccess(Value *Val, int64_t Start, int64_t End);
  void addMemAccessForArray(Value *Val, ArrayID Array, int64_t Offset);

  APInt tryAddIntInput(Value *Val, APInt &&Desired);

  void build(Module &M, SmallVectorImpl<Value *> &Args);

  unsigned NumInputArgs = 0;
};
using InputBuilder = ExplorativeLVPass::InputBuilder;

InputBuilder::ArrayID InputBuilder::addMemAccess(Value *Val, int64_t Start,
                                                 int64_t End) {
  assert(Start <= End && "Start <= End required");

  ArrayID RangeID = MemRanges.size();
  auto Res = MemAccesses.try_emplace(Val, RangeID, 0);
  if (Res.second) { // Did emplace
    // Sanitize the range.  Even if the memory access is in [Start, End], we
    // also want that the pointer itself is in [Start, End].
    if (Start > 0)
      Start = 0;
    if (End < 0)
      End = 0;

    MemRanges.emplace_back(Start, End,
                           Val->getType()->getNonOpaquePointerElementType());
    return RangeID;
  }

  RangeID = Res.first->second.first;
  MemAccessRange &Range = MemRanges[RangeID];
  if (Start < Range.Start)
    Range.Start = Start;
  if (End > Range.End)
    Range.End = End;
  return RangeID;
}

void InputBuilder::addMemAccessForArray(Value *Val, ArrayID Array,
                                        int64_t Offset) {
  MemAccessRange &Range = MemRanges[Array];
  assert(Val->getType()->getNonOpaquePointerElementType() ==
             Range.ElementType &&
         "Pointee types must agree");
  if (Offset < Range.Start) {
    Range.Start = Offset;
  } else if (Offset > Range.End) {
    Range.End = Offset;
  }
  MemAccesses.try_emplace(Val, Array, Offset);
}

APInt InputBuilder::tryAddIntInput(Value *Val, APInt &&Desired) {
  // Use a dummy pointer first
  auto Res = IntInputs.try_emplace(Val, nullptr);

  if (!Res.second) // Didn't emplace
    return Res.first->second->getValue();

  // Now, we need to construct the value
  Res.first->second = ConstantInt::get(Val->getType()->getContext(), Desired);
  return Desired;
}

void InputBuilder::build(Module &M, SmallVectorImpl<Value *> &Args) {
  assert(NumInputArgs == Args.size() && "NumInputArgs was not properly set");

  for (auto &Input : IntInputs) {
    Value *Val = Input.first;
    if (const Argument *Arg = dyn_cast<Argument>(Val)) {
      assert(Arg->getArgNo() < NumInputArgs &&
             "Inferring value for output argument?");
      Args[Arg->getArgNo()] = Input.second;
      continue;
    }

    GlobalVariable *GV = cast<GlobalVariable>(Val);
    if (GV->hasInitializer()) {
      // TODO: Should we preserve already present initializers?
      LLVM_DEBUG(dbgs() << "XLV: overwriting initializer "
                        << *GV->getInitializer() << "\n");
    }
    GV->setInitializer(Input.second);
  }

  DataLayout DL(&M);
  for (MemAccessRange &Range : MemRanges) {
    uint64_t Size =
        (Range.End - Range.Start) / DL.getTypeAllocSize(Range.ElementType) + 1;
    ArrayType *Ty = ArrayType::get(Range.ElementType, Size);
    Range.GV = new GlobalVariable(M, Ty, false, GlobalValue::PrivateLinkage,
                                  ConstantAggregateZero::get(Ty));
  }

  LLVMContext &C = M.getContext();
  Type *I64Ty = Type::getInt64Ty(C);
  Constant *I640 = ConstantInt::get(I64Ty, 0);
  for (auto &Access : MemAccesses) {
    Value *Val = Access.first;
    ArrayID RangeID = Access.second.first;
    MemAccessRange &Range = MemRanges[RangeID];
    GlobalVariable *GV = Range.GV;
    assert(GV);
    uint64_t Offset = (Access.second.second - Range.Start) /
                      DL.getTypeAllocSize(Range.ElementType);

    Constant *GEP = ConstantExpr::getInBoundsGetElementPtr(
        GV->getValueType(), GV,
        ArrayRef<Constant *>{I640, ConstantInt::get(I64Ty, Offset)});

    if (const Argument *Arg = dyn_cast<Argument>(Val)) {
      assert(Arg->getArgNo() < Args.size() &&
             "Inferring value for output argument?");
      Args[Arg->getArgNo()] = GEP;
      continue;
    }

    LLVM_DEBUG(
        dbgs() << "XLV: ignoring memory access to global (not via argument) - "
                  "is loop invariant code motion enabled?\n");
    // TODO: should we handle this case or should we move message?
  }
}

namespace {

class LoopModuleBuilder : public ValueMaterializer {
  const Function &OrigFunc;
  Loop &L;
  ScalarEvolution &SE;

  Module *M = nullptr;
  ValueToValueMapTy VMap;
  ValueMapper Mapper;
  SmallVector<const Value *> Inputs;
  SmallVector<const Instruction *> Outputs;
  FunctionType *FuncTy;
  SmallVector<Function *> LoopFuncs;

  InputBuilder InferredInputs;

  bool MakeExecutable;

  bool determineIO();
  void buildFuncTy();

public:
  LoopModuleBuilder(const Function &OrigFunc, Loop &L, ScalarEvolution &SE,
                    bool MakeExecutable = false)
      : OrigFunc(OrigFunc), L(L), SE(SE), Mapper(VMap, RF_None, nullptr, this),
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

  InputBuilder &getInferredInputsRef() {
    assert(M && "No loop func has been built yet");
    return InferredInputs;
  }
};

bool LoopModuleBuilder::determineIO() {
  // The core idea is to generate a function containing the basic blocks of the
  // loop.  Values used inside the loop and defined outside are parameters to
  // the function.  For each values defined inside the loop and used outside we
  // generate an output parameter to produce an observable side-effect such that
  // the code is not optimized away.

  SmallPtrSet<const Value *, 32> ArgSet;

  for (const BasicBlock *BB : L.getBlocks()) {
    for (const Instruction &I : *BB) {
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
      for (const Value *Op : I.operand_values()) {
        const Instruction *OpInst = dyn_cast<Instruction>(Op);
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
      for (const User *U : I.users()) {
        const Instruction *UI = dyn_cast<Instruction>(U);
        if (!UI)
          continue; // ignore users that aren't instructions
        if (!L.getBlocksSet().contains(UI->getParent())) {
          Outputs.push_back(&I);
          break;
        }
      }
    }
  }

  InferredInputs.NumInputArgs = Inputs.size();

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

    if (!F->isIntrinsic())
      // We only call intrinsics.
      return Constant::getNullValue(F->getType());

    Function *NF = Function::Create(cast<FunctionType>(F->getValueType()),
                                    F->getLinkage(), F->getName(), M);
    NF->copyAttributesFrom(F);
    return NF;
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
  for (const Value *V : Inputs)
    Params.push_back(V->getType());
  for (const Instruction *I : Outputs)
    Params.push_back(PointerType::getUnqual(I->getType()));

  // Returns void, no varargs
  FuncTy = FunctionType::get(Type::getVoidTy(M->getContext()), Params, false);
}

Module *LoopModuleBuilder::getModule() {
  if (M)
    return M;

  if (MakeExecutable &&
      isa<SCEVCouldNotCompute>(SE.getSymbolicMaxBackedgeTakenCount(&L))) {
    LLVM_DEBUG(dbgs() << "XLV: cannot infer trip count\n");
    return nullptr;
  }

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
    for (const Value *OldV : Inputs) {
      VMap[OldV] = ArgPtr;
      VMap[ArgPtr] = ArgPtr;
      if (OldV->hasName())
        ArgPtr->setName(OldV->getName() + ".i");
      ++ArgPtr;
    }
    for (const Instruction *I : Outputs) {
      VMap[ArgPtr] = ArgPtr;
      if (I->hasName())
        ArgPtr->setName(I->getName() + ".o");
      ++ArgPtr;
    }
  }

  BasicBlock *PreHdrBB = BasicBlock::Create(C, "loop.preheader", LoopFunc);
  VMap[PreHdrBB] = PreHdrBB;

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
  BasicBlock *FirstStoreBB = nullptr;

  // Loop over all of the basic blocks in the loop, cloning them as appropriate.
  for (const BasicBlock *BB : L.getBlocks()) {
    // Create a new basic block
    BasicBlock *NewBB = BasicBlock::Create(C, "", LoopFunc, FirstStoreBB);
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
      const Constant *OldBBAddr = BlockAddress::get(
          const_cast<Function *>(&OrigFunc), const_cast<BasicBlock *>(BB));
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

      BasicBlock *StoreBB = BasicBlock::Create(C, "store_block", LoopFunc);
      if (!FirstStoreBB)
        FirstStoreBB = StoreBB;
      VMap[StoreBB] = StoreBB;
      NewTerm->setSuccessor(Idx, StoreBB);

      Function::arg_iterator OutputArg = OutputArgsBegin;
      for (const Instruction *I : Outputs) {
        assert(I->getParent()->getParent() == &OrigFunc);
        for (const User *U : I->users()) {
          const Instruction *UI = dyn_cast<Instruction>(U);
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

      ReturnInst *Ret = ReturnInst::Create(C, nullptr, StoreBB);
      VMap[Ret] = Ret;
    }
  }

  BasicBlock *NewLoopHeader =
      LoopFunc->getBasicBlockList().getNextNode(*PreHdrBB);
  BranchInst *Branch = BranchInst::Create(NewLoopHeader, PreHdrBB);
  VMap[Branch] = Branch;

  // The first basic block might have had multiple predecessors that were not
  // part of the loop.  In our new setup, they are all replaced by the
  // preheader.  Update the PHINodes accordingly.  Since we have a (natural)
  // loop, no other basic block than the loop header has predecedecessors
  // outside the loop.
  for (PHINode &P : NewLoopHeader->phis()) {
    for (BasicBlock **It = P.block_begin(), **End = P.block_end(); It != End;
         ++It) {
      if (VMap.count(*It) == 0)
        *It = PreHdrBB;
    }
  }

  // Clone the MDMap such that such that we can reset it to the state where it
  // just contained module-level metadata.  This is needed because there might
  // be 'distinct' nodes that must not be shared by multiple loop functions.
  auto MDModuleMap = VMap.MD();

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

  // Restore MDMap
  VMap.MD().swap(MDModuleMap);

  if (VF == 1 && IF == 1 && XLVDumpFuncIR)
    dbgs() << *LoopFunc << "\n";

  LoopFuncs.push_back(LoopFunc);
  return LoopFunc;
}

void LoopModuleBuilder::buildMainFuncBody() {
  LLVMContext &C = M->getContext();

  // Some types and values we will use a few times
  Type *I32Ty = Type::getInt32Ty(C);
  Type *I64Ty = Type::getInt64Ty(C);
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

    // Create input arguments
    Args.assign(Inputs.size(), nullptr);
    InferredInputs.build(*M, Args);
    for (Value *&Arg : Args) {
      if (!Arg)
        Arg = Constant::getNullValue(*It);
      LLVM_DEBUG(dbgs() << "XLV: arg " << *Arg << "\n");
      ++It;
    }
    CArgs = Args;

    // Alloc locations for output parameters
    for (auto *End = LoopFuncTy->param_end(); It != End; ++It) {
      Type *ElementType = (*It)->getNonOpaquePointerElementType();
      Args.push_back(new AllocaInst(ElementType, BenchFunc->getAddressSpace(),
                                    "outarg", StartBB));
      CArgs.push_back(new AllocaInst(ElementType, CalibFunc->getAddressSpace(),
                                     "outarg", CStartBB));
    }
  }

  // Warmup
  for (unsigned I = 0; I < XLVBenchmarkWarmup; ++I) {
    CallInst::Create(LoopFuncTy, FuncArg, Args, "", StartBB);
    CallInst::Create(LoopFuncTy, CFuncArg, CArgs, "", CStartBB);
  }

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

  for (unsigned I = 0; I < XLVBenchmarkLoopSize; ++I) {
    CallInst::Create(LoopFuncTy, FuncArg, Args, "", LoopBB);
    CallInst::Create(LoopFuncTy, CFuncArg, CArgs, "", CLoopBB);
  }

  // Counters and exit conditions
  ICmpInst *ContLoop = new ICmpInst(*LoopBB, CmpInst::ICMP_EQ, CountPhi, NArg);
  BinaryOperator *CountAdd =
      BinaryOperator::CreateAdd(CountPhi, I321, "counter_next", LoopBB);

  CallInst *CTNow = CallInst::Create(ClockFuncTy, ClockFunc, "tn", CLoopBB);
  BinaryOperator *CTDiff =
      BinaryOperator::CreateSub(CTNow, CTStart, "dt", CLoopBB);
  ICmpInst *CContLoop =
      new ICmpInst(*CLoopBB, CmpInst::ICMP_ULT, CTDiff,
                   ConstantInt::get(I64Ty, XLVBenchmarkTicks));
  BinaryOperator *CCountAdd =
      BinaryOperator::CreateAdd(CCountPhi, I321, "counter_next", CLoopBB);

  // Branch at the end of the loop
  BasicBlock *EvalBB = BasicBlock::Create(C, "bench_eval", BenchFunc);
  BranchInst::Create(EvalBB, LoopBB, ContLoop, LoopBB);
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
static uint64_t performMCACostCalc(StringRef FileName, StringRef TargetCPU,
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
    errs() << "XLV: could not find llvm-mca executable\n";
    return ExplorativeLVPass::InvalidCosts;
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
    errs() << "XLV: executing llvm-mca failed";
    return ExplorativeLVPass::InvalidCosts;
  }

  // Read JSON
  auto MCAOutBuf = MemoryBuffer::getFile(MCAOutPath);
  if (!MCAOutBuf) {
    errs() << "XLV: could not read llvm-mca output\n";
    return ExplorativeLVPass::InvalidCosts;
  }
  StringRef MCAOut = MCAOutBuf.get()->getBuffer();
  auto MCAJSON = json::parse(MCAOut);
  if (!MCAJSON) {
    errs() << "XLV: could not parse llvm-mca JSON\n";
    return ExplorativeLVPass::InvalidCosts;
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

  errs() << "XLV: llvm-mca output is malformed\n";
  return ExplorativeLVPass::InvalidCosts;
}

// Add to the given pass manager everything we need to simulate our backend
// pipeline
// ~~> EmitAssemblyHelper::AddEmitPasses in clang's BackendUtil
static void initCodeGen(legacy::PassManager &PM, TargetMachine &TM,
                        ExplorativeLVPass *XLV, const TargetLibraryInfo &TLI,
                        raw_pwrite_stream &OS, CodeGenFileType CGFT) {
  if (XLVDumpModuleIR)
    PM.add(createPrintModulePass(outs()));

  PM.add(new ExplorativeLVHelper(XLV));

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
                       Optional<PGOOptions> PGOOpt, ExplorativeLVPass *XLV,
                       const TargetLibraryInfo &TLI)
      : SI(/* DebugLogging */ false) {
    SI.registerCallbacks(PIC, &FAM);

    // Create a new PassBuilder which is the basis of a pipeline
    // TODO: I'm not 100% sure that it's wise to use the original TM here
    PassBuilder PB(TM, PTO, PGOOpt, &PIC);

    // Register the ExplorativeLVAnalysis with the current ExplorativeLVPass.
    // The optimization pipeline we create will contain a new ExplorativeLVPass
    // and it should behave different compared to the current ExplorativeLVPass
    // (i. e. do not clone the loops again, pass inferred arguments back in
    // benchmarking mode).
    FAM.registerPass([&] { return ExplorativeLVAnalysis(XLV); });

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

  NullCGPipelineContainer(TargetMachine &TM, ExplorativeLVPass *XLV,
                          const TargetLibraryInfo &TLI) {
    initCodeGen(PM, TM, XLV, TLI, OS, CGFT_Null);
  }
};

ExplorativeLVPass::~ExplorativeLVPass() {
  delete NullCGPipeline;
  delete OptPipeline;
}

static void updateBestVFIF(unsigned &BestVF, unsigned &BestIF,
                           uint64_t &MinCosts, unsigned VF, unsigned IF,
                           uint64_t Costs) {
  if (Costs == ExplorativeLVPass::InvalidCosts) {
    LLVM_DEBUG(dbgs() << "XLV: invalid costs for VF " << VF << ", IF " << IF
                      << "\n");
    return;
  }

  if (XLVDivVF)
    Costs /= VF;

  LLVM_DEBUG(dbgs() << "XLV: costs for VF " << VF << ", IF " << IF << ": "
                    << Costs << "\n");

  if (Costs < MinCosts) {
    MinCosts = Costs;
    BestVF = VF;
    BestIF = IF;
  } else if (Costs == MinCosts && (VF == 1 || VF >= BestVF) && IF >= BestIF) {
    // If costs are equal to costs retrieved before, always choose the
    // higher VF-IF combination unless best VF is scalar.
    BestVF = VF;
    BestIF = IF;
  }
}

// returns whether an error occured
bool ExplorativeLVPass::processLoop(Function &F, Loop &L, ScalarEvolution &SE,
                                    TargetLibraryInfo &TLI) {
  assert(OptPipeline && "OptPipeline not initialized");

  LoopModuleBuilder Builder(F, L, SE, XLVMetric == Metric::Benchmark);
  Module *M = Builder.getModule();
  if (!M)
    return true;

  unsigned BestVF = 1;
  unsigned BestIF = 1;
  uint64_t MinCosts = InvalidCosts;

  if (XLVMetric == Metric::InstCount) {
    DenseMap<Function *, EvaluationInfo> ResultMap;
    EvaluationInfoMap = &ResultMap;

    // Build a module containing functions for all VF/IF combinations
    for (unsigned VF = 1; VF <= XLVMaxVF; VF *= 2) {
      for (unsigned IF = 1; IF <= XLVMaxIF; IF *= 2) {
        Function *Func = Builder.buildLoopFunc(VF, IF);
        ResultMap[Func] = {VF, IF, InvalidCosts};
      }
    }

    // Run pipelines
    OptPipeline->run(*M);
    NullCGPipeline->PM.run(*M);

    for (auto &KV : ResultMap) {
      EvaluationInfo &Info = KV.second;
      updateBestVFIF(BestVF, BestIF, MinCosts, Info.VF, Info.IF, Info.Costs);
    }

    EvaluationInfoMap = nullptr;
  } else if (XLVMetric == Metric::MCA) {
    // Try out which VF works best for this loop
    for (unsigned VF = 1; VF <= XLVMaxVF; VF *= 2) {
      for (unsigned IF = 1; IF <= XLVMaxIF; IF *= 2) {
        Builder.buildLoopFunc(VF, IF);
        OptPipeline->run(*M);

        // Expensive, but: If we are using llvm-mca for evaluation, we need
        // a new backend pipeline for each iteration as the print stream is
        // dead after one use
        int ASMFD;
        SmallString<32> ASMPath;
        sys::fs::createTemporaryFile("explorative-lv", "s", ASMFD, ASMPath);
        FileRemover ASMRemover(ASMPath);
        {
          raw_fd_ostream OS(ASMFD, true);
          legacy::PassManager CGPipeline;
          initCodeGen(CGPipeline, *TM, this, TLI, OS, CGFT_AssemblyFile);
          CGPipeline.run(*M);
        }
        // Perform the cost calculation after the pipeline is deleted, to make
        // sure the assembly printer has finished
        uint64_t Costs = performMCACostCalc(ASMPath, TM->getTargetCPU(),
                                            M->getTargetTriple());
        updateBestVFIF(BestVF, BestIF, MinCosts, VF, IF, Costs);

        Builder.clearLoopFuncs();
      }
    }
  } else {
    assert(XLVMetric == Metric::Benchmark);

    // Build a module containing functions for all VF/IF combinations
    for (unsigned VF = 1; VF <= XLVMaxVF; VF *= 2) {
      for (unsigned IF = 1; IF <= XLVMaxIF; IF *= 2)
        Builder.buildLoopFunc(VF, IF);
    }

    // Get a reference to the inferred inputs vector.  The ExplorativeLVPass in
    // the optimization pipeline will (try to) infer the arguments.
    InferredInputs = &Builder.getInferredInputsRef();

    OptPipeline->run(*M);

    // The arguments should be inferred exactly once.  To achieve this,
    // InferredInputs is cleared.
    assert(!InferredInputs && "Arguments have not been inferred");

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
      initCodeGen(CGPipeline, *TM, this, TLI, OS, CGFT_ObjectFile);
      CGPipeline.run(*M);
    }

    // Create an executable
    auto CCPath = sys::findProgramByName("cc"); // FIXME: do this only once
    if (!CCPath) {
      errs() << "XLV: could not find cc executable\n";
      return true;
    }
    SmallString<32> ExecPath;
    sys::fs::createTemporaryFile("explorative-lv", "", ExecPath);
    FileRemover ExecRemover(ExecPath);
    if (sys::ExecuteAndWait(CCPath.get(), {"cc", "-o", ExecPath, ObjectPath}) !=
        0) {
      errs() << "XLV: linking failed\n";
      return true;
    }

    // Do the benchmarking
    unsigned TimeoutS =
        1 + 2 * ((Builder.getNumLoopFuncs() + 1) * XLVBenchmarkTicks) /
                CLOCKS_PER_SEC; // ceil(2 * (calib_secs + n * bench_secs))
    LLVM_DEBUG(dbgs() << "XLV: benchmarking for up to " << TimeoutS
                      << " s ...\n");
    SmallString<32> BenchResPath;
    sys::fs::createTemporaryFile("explorative-lv-out", "txt", BenchResPath);
    FileRemover BenchResRemover(BenchResPath);
    int RetCode = sys::ExecuteAndWait(
        ExecPath, {"explorative-lv"}, None,
        {StringRef(""), BenchResPath.str(), StringRef("")}, TimeoutS);
    if (RetCode == -2) {
      errs()
          << "XLV: benchmarking failed/timed out.  You can find the binary at "
          << ExecPath << "\n";
      ExecRemover.releaseFile();
      return true;
    }
    if (RetCode == -1) {
      errs() << "XLV: failed to execute " << ExecPath << "\n";
      ExecRemover.releaseFile();
      return true;
    }

    auto BenchResBuf = MemoryBuffer::getFile(BenchResPath);
    if (!BenchResBuf) {
      errs() << "XLV: could not load benchmarking results";
      return true;
    }
    StringRef BenchRes = BenchResBuf.get()->getBuffer();

    for (unsigned VF = 1; VF <= XLVMaxVF; VF *= 2) {
      for (unsigned IF = 1; IF <= XLVMaxIF; IF *= 2) {
        uint64_t Costs;
        if (BenchRes.consumeInteger(10, Costs)) {
          errs() << "XLV: could not read integer from benchmarking results";
          return true;
        }
        BenchRes = BenchRes.drop_front(); // drop '\n'

        updateBestVFIF(BestVF, BestIF, MinCosts, VF, IF, Costs);
      }
    }
  }

  // Force the vectorizer to use the VF and IF that showed to have the lowest
  // cost
  LLVM_DEBUG(dbgs() << "XLV: choosing this VF and IF for vectorization: "
                    << BestVF << " and " << BestIF << "\n");
  addStringMetadataToLoop(&L, "llvm.loop.vectorize.width", BestVF);
  addStringMetadataToLoop(&L, "llvm.loop.interleave.count", BestIF);

  return false; // no errors
}

// Inspired by LoopVectorize.cpp
static void collectSupportedLoops(Loop &L, LoopInfo *LI,
                                  SmallVectorImpl<Loop *> &V) {
  if (L.isInnermost()) {
    if (L.getBlocks().empty())
      return;

    if (findStringMetadataForLoop(&L, "llvm.loop.vectorize.width")) {
      LLVM_DEBUG(dbgs() << "XLV: loop already has user annotation, skipping "
                           "exploration\n");
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

namespace {

/// This visitor is slightly specialized compared to SCEVVisitor.  It always
/// computes an Optional<APInt> and already handles SCEVMinMaxExprs as well as
/// SCEVSequentialMinMaxExprs.
template <typename SC, typename... ArgVals> class SCEVCompVisitor {
  /// Fold an n-ary expression with the accumulator initialized to the
  /// "visited" first operand.  The folding function takes a reference to the
  /// accumulator as the first argument and the "new" value as second argument.
  template <typename FuncT>
  Optional<APInt> foldNAry(const SCEVNAryExpr *Expr, FuncT Func,
                           ArgVals... Args) {
    const auto *It = Expr->op_begin(), *End = Expr->op_end();
    assert(It != End && "Expected at least one operand");
    Optional<APInt> Res = visit(*It, Args...);
    if (!Res)
      return None;
    for (++It; It != End; ++It) {
      Optional<APInt> Tmp = visit(*It, Args...);
      if (!Tmp)
        return None;
      Func(*Res, std::move(*Tmp));
    }
    return Res;
  }

public:
  Optional<APInt> visit(const SCEV *S, ArgVals... Args) {
    switch (S->getSCEVType()) {
    case scConstant:
      return ((SC *)this)->visitConstant((const SCEVConstant *)S, Args...);
    case scPtrToInt:
      return ((SC *)this)
          ->visitPtrToIntExpr((const SCEVPtrToIntExpr *)S, Args...);
    case scTruncate:
      return ((SC *)this)
          ->visitTruncateExpr((const SCEVTruncateExpr *)S, Args...);
    case scZeroExtend:
      return ((SC *)this)
          ->visitZeroExtendExpr((const SCEVZeroExtendExpr *)S, Args...);
    case scSignExtend:
      return ((SC *)this)
          ->visitSignExtendExpr((const SCEVSignExtendExpr *)S, Args...);
    case scAddExpr:
      return ((SC *)this)->visitAddExpr((const SCEVAddExpr *)S, Args...);
    case scMulExpr:
      return ((SC *)this)->visitMulExpr((const SCEVMulExpr *)S, Args...);
    case scUDivExpr:
      return ((SC *)this)->visitUDivExpr((const SCEVUDivExpr *)S, Args...);
    case scAddRecExpr:
      return ((SC *)this)->visitAddRecExpr((const SCEVAddRecExpr *)S, Args...);
    case scSMaxExpr:
      return foldNAry((const SCEVNAryExpr *)S,
                      [](APInt &Acc, APInt &&Sub) {
                        if (Sub.sgt(Acc))
                          Acc = std::move(Sub);
                      },
                      Args...);
    case scUMaxExpr:
      return foldNAry((const SCEVNAryExpr *)S,
                      [](APInt &Acc, APInt &&Sub) {
                        if (Sub.sgt(Acc))
                          Acc = std::move(Sub);
                      },
                      Args...);
    case scSMinExpr:
      return foldNAry((const SCEVNAryExpr *)S,
                      [](APInt &Acc, APInt &&Sub) {
                        if (Sub.sgt(Acc))
                          Acc = std::move(Sub);
                      },
                      Args...);
    case scUMinExpr:
    case scSequentialUMinExpr:
      return foldNAry((const SCEVNAryExpr *)S,
                      [](APInt &Acc, APInt &&Sub) {
                        if (Sub.sgt(Acc))
                          Acc = std::move(Sub);
                      },
                      Args...);
    case scUnknown:
      return ((SC *)this)->visitUnknown((const SCEVUnknown *)S, Args...);
    case scCouldNotCompute:
      return None;
    }
    llvm_unreachable("Unknown SCEV kind!");
  }
};

class SCEVBackedgeTakenAnalyzer
    : private SCEVCompVisitor<SCEVBackedgeTakenAnalyzer, APInt> {
  friend class SCEVCompVisitor<SCEVBackedgeTakenAnalyzer, APInt>;

  ScalarEvolution &SE;
  InputBuilder &Inferred;

  /// Start pointer value
  ///
  /// Currently, we only permit a single combination of a start and an end
  /// pointer to determine the trip count.
  Value *InferredPtrStart = nullptr;

  /// End pointer value
  Value *InferredPtrEnd = nullptr;

  /// Difference between Start and End pointer (if present)
  uint64_t InferredPtrDiff;

  SCEVBackedgeTakenAnalyzer(ScalarEvolution &SE, InputBuilder &InferredInputs)
      : SE(SE), Inferred(InferredInputs) {}

  Optional<APInt> visitConstant(const SCEVConstant *Constant, APInt) {
    return Constant->getAPInt();
  }

  Optional<APInt> visitPtrToIntExpr(const SCEVPtrToIntExpr *Expr,
                                    APInt Desired) {
    const SCEV *Sub = Expr->getOperand();
    unsigned Bits = SE.getTypeSizeInBits(Expr->getType());
    unsigned SubBits = SE.getTypeSizeInBits(Sub->getType());
    return visit(Sub, Desired.zextOrTrunc(SubBits)).map([Bits](APInt &&Res) {
      return Res.zextOrTrunc(Bits);
    });
  }
  Optional<APInt> visitTruncateExpr(const SCEVTruncateExpr *Expr,
                                    APInt Desired) {
    const SCEV *Sub = Expr->getOperand();
    unsigned Bits = SE.getTypeSizeInBits(Expr->getType());
    unsigned SubBits = SE.getTypeSizeInBits(Sub->getType());
    return visit(Sub, Desired.zext(SubBits)).map([Bits](APInt &&Res) {
      return Res.trunc(Bits);
    });
  }
  Optional<APInt> visitZeroExtendExpr(const SCEVZeroExtendExpr *Expr,
                                      APInt Desired) {
    const SCEV *Sub = Expr->getOperand();
    unsigned Bits = SE.getTypeSizeInBits(Expr->getType());
    unsigned SubBits = SE.getTypeSizeInBits(Sub->getType());
    return visit(Sub, Desired.trunc(SubBits)).map([Bits](APInt &&Res) {
      return Res.zext(Bits);
    });
  }
  Optional<APInt> visitSignExtendExpr(const SCEVSignExtendExpr *Expr,
                                      APInt Desired) {
    const SCEV *Sub = Expr->getOperand();
    unsigned Bits = SE.getTypeSizeInBits(Expr->getType());
    unsigned SubBits = SE.getTypeSizeInBits(Sub->getType());
    return visit(Sub, Desired.trunc(SubBits)).map([Bits](APInt &&Res) {
      return Res.sext(Bits);
    });
  }

  Optional<APInt> visitAddExpr(const SCEVAddExpr *Expr, APInt Desired) {
    APInt Res(SE.getTypeSizeInBits(Expr->getType()), 0);

    // Handle constant operands first
    for (const SCEV *Sub : Expr->operands()) {
      if (const SCEVConstant *Const = dyn_cast<SCEVConstant>(Sub))
        Res += Const->getAPInt();
    }
    Desired -= Res;

    for (const SCEV *Sub : Expr->operands()) {
      if (isa<SCEVConstant>(Sub))
        continue;

      // We hope that there is just one unknown in a "positive" position.
      // Otherwise we end up with more iterations than desired, but this should
      // not be too bad either.
      Optional<APInt> SubRes = visit(Sub, Desired);
      if (!SubRes)
        return None;
      Res += *SubRes;
    }

    return Res;
  }

  /// desired = c * x <=> x = desired /u c
  /// desired = -c * x '<=>' x = 0
  Optional<APInt> visitMulExpr(const SCEVMulExpr *Expr, APInt Desired) {
    uint64_t Bits = SE.getTypeSizeInBits(Expr->getType());
    APInt Res(Bits, 1);

    // Handle constant operands first
    for (const SCEV *Sub : Expr->operands()) {
      if (const SCEVConstant *Const = dyn_cast<SCEVConstant>(Sub))
        Res *= Const->getAPInt();
    }

    // If this is a "negative" position the desired value is 0.
    Desired = Res.isNegative() ? APInt(Bits, 0) : Desired.udiv(Res);
    for (const SCEV *Sub : Expr->operands()) {
      if (isa<SCEVConstant>(Sub))
        continue;

      // We hope that there is just one unknown in a "positive" position.
      // Otherwise we end up with more iterations than desired, but this should
      // not be too bad either.
      Optional<APInt> SubRes = visit(Sub, Desired);
      if (!SubRes)
        return None;
      Res *= *SubRes;
    }

    return Res;
  }

  /// desired = dividend /u divisor <=> dividend = desired * divisor
  Optional<APInt> visitUDivExpr(const SCEVUDivExpr *Expr, APInt Desired) {
    APInt MyDesired = std::move(Desired);
    Desired = APInt(SE.getTypeSizeInBits(Expr->getType()), 1);
    Optional<APInt> DivisorRes = visit(Expr->getRHS(), Desired);
    if (!DivisorRes)
      return None;

    Desired = MyDesired * *DivisorRes;
    Optional<APInt> DividendRes = visit(Expr->getLHS(), Desired);
    if (!DividendRes)
      return None;

    Desired = std::move(MyDesired);
    return DividendRes->udiv(*DivisorRes);
  }

  Optional<APInt> visitAddRecExpr(const SCEVAddRecExpr *, APInt) {
    // We should not encounter a SCEVAddRecExpr as the extracted loop is the
    // outermost loop
    return None;
  }

  Optional<APInt> visitUnknown(const SCEVUnknown *Unknown, APInt Desired) {
    Value *Val = Unknown->getValue();
    Type *Ty = Val->getType();

    if (Ty->isIntegerTy())
      return Inferred.tryAddIntInput(Val, std::move(Desired));

    assert(Ty->isPointerTy() &&
           "SCEV analysis deals with integer and pointer types only?");

    if (Desired.isZero()) {
      if (!InferredPtrStart) {
        InferredPtrStart = Val;
        return Desired;
      }
      if (InferredPtrStart == Val)
        return Desired;
    } else {
      if (!InferredPtrEnd) {
        InferredPtrEnd = Val;
        InferredPtrDiff = Desired.getZExtValue();
        return Desired;
      }
      if (InferredPtrEnd == Val)
        return Desired;
    }

    // There is already another start or end pointer set.  We cannot deal with
    // this.
    return None;
  }

public:
  static Optional<APInt> analyze(ScalarEvolution &SE, const SCEV *Expr,
                                 InputBuilder &Inferred,
                                 unsigned DesiredBackedgeTaken) {
    if (isa<SCEVCouldNotCompute>(Expr))
      return None; // else Expr->getType() will fail

    SCEVBackedgeTakenAnalyzer Analyzer(SE, Inferred);
    Optional<APInt> Res =
        Analyzer.visit(Expr, APInt(SE.getTypeSizeInBits(Expr->getType()),
                                   DesiredBackedgeTaken));
    if (Analyzer.InferredPtrStart && Analyzer.InferredPtrEnd) {
      // One less than InferredPtrDiff would be enough, since the range is
      // inclusive.  However, we'd also need to change addMemAccessForArray such
      // that the element is not allocated there ...
      auto Array = Inferred.addMemAccess(Analyzer.InferredPtrStart, 0,
                                         Analyzer.InferredPtrDiff);
      Inferred.addMemAccessForArray(Analyzer.InferredPtrEnd, Array,
                                    Analyzer.InferredPtrDiff);
    }
    return Res;
  }
};

class SCEVMemAccessAnalyzer : private SCEVCompVisitor<SCEVMemAccessAnalyzer> {
  friend class SCEVCompVisitor<SCEVMemAccessAnalyzer>;

  ScalarEvolution &SE;
  InputBuilder &Inferred;
  const SCEV *ItStart;
  const SCEV *ItEnd;

  Value *BasePointer = nullptr;

  Optional<APInt> visitConstant(const SCEVConstant *Constant) {
    return Constant->getAPInt();
  }

  Optional<APInt> visitPtrToIntExpr(const SCEVPtrToIntExpr *Expr) {
    const SCEV *Sub = Expr->getOperand();
    unsigned Bits = SE.getTypeSizeInBits(Expr->getType());
    return visit(Sub).map(
        [Bits](APInt &&Res) { return Res.zextOrTrunc(Bits); });
  }
  Optional<APInt> visitTruncateExpr(const SCEVTruncateExpr *Expr) {
    const SCEV *Sub = Expr->getOperand();
    unsigned Bits = SE.getTypeSizeInBits(Expr->getType());
    return visit(Sub).map([Bits](APInt &&Res) { return Res.trunc(Bits); });
  }
  Optional<APInt> visitZeroExtendExpr(const SCEVZeroExtendExpr *Expr) {
    const SCEV *Sub = Expr->getOperand();
    unsigned Bits = SE.getTypeSizeInBits(Expr->getType());
    return visit(Sub).map([Bits](APInt &&Res) { return Res.zext(Bits); });
  }
  Optional<APInt> visitSignExtendExpr(const SCEVSignExtendExpr *Expr) {
    const SCEV *Sub = Expr->getOperand();
    unsigned Bits = SE.getTypeSizeInBits(Expr->getType());
    return visit(Sub).map([Bits](APInt &&Res) { return Res.sext(Bits); });
  }

  Optional<APInt> visitAddExpr(const SCEVAddExpr *Expr) {
    APInt Res(SE.getTypeSizeInBits(Expr->getType()), 0);
    for (const SCEV *Sub : Expr->operands()) {
      Optional<APInt> SubRes = visit(Sub);
      if (!SubRes)
        return None;
      Res += *SubRes;
    }
    return Res;
  }
  Optional<APInt> visitMulExpr(const SCEVMulExpr *Expr) {
    APInt Res(SE.getTypeSizeInBits(Expr->getType()), 1);
    for (const SCEV *Sub : Expr->operands()) {
      Optional<APInt> SubRes = visit(Sub);
      if (!SubRes)
        return None;
      Res *= *SubRes;
    }
    return Res;
  }
  Optional<APInt> visitUDivExpr(const SCEVUDivExpr *Expr) {
    Optional<APInt> DivisorRes = visit(Expr->getRHS());
    if (!DivisorRes)
      return None;
    return visit(Expr->getLHS()).map([&DivisorRes](APInt &&DividendRes) {
      return DividendRes.udiv(*DivisorRes);
    });
  }

  Optional<APInt> visitAddRecExpr(const SCEVAddRecExpr *Expr) {
    // TODO: fix this
    LLVM_DEBUG(dbgs() << "XLV: found an inner SCEVAddRecExpr when analyzing "
                         "memory accesses\n");
    return None;
  }

  Optional<APInt> visitUnknown(const SCEVUnknown *Unknown) {
    Value *Val = Unknown->getValue();
    if (!isa<Argument>(Val) && !isa<GlobalVariable>(Val))
      return None;

    Type *Ty = Val->getType();

    if (IntegerType *IntTy = dyn_cast<IntegerType>(Ty))
      return Inferred.tryAddIntInput(Val, APInt(IntTy->getBitWidth(), 1));

    assert(Ty->isPointerTy() &&
           "SCEV analysis deals with integer and pointer types only?");

    if (BasePointer != Val) {
      if (BasePointer) {
        LLVM_DEBUG(dbgs() << "XLV: found two base pointers when analyzing "
                             "memory accesses\n");
        return None;
      }

      BasePointer = Val;
    }

    // We calculate accesses relative to the pointer
    return APInt(SE.getTypeSizeInBits(Ty), 0);
  }

  Optional<APInt> visitCouldNotCompute(const SCEVCouldNotCompute *) {
    return None;
  }

public:
  SCEVMemAccessAnalyzer(ScalarEvolution &SE, InputBuilder &InferredInputs,
                        const SCEV *ItStart, const SCEV *ItEnd)
      : SE(SE), Inferred(InferredInputs), ItStart(ItStart), ItEnd(ItEnd) {}

  void analyze(const SCEV *Expr) {
    BasePointer = nullptr;
    if (const SCEVAddRecExpr *AddRec = dyn_cast<SCEVAddRecExpr>(Expr)) {
      if (AddRec->isAffine()) {
        Optional<APInt> StartRes =
            visit(AddRec->evaluateAtIteration(ItStart, SE));
        if (!StartRes)
          return;
        Optional<APInt> EndRes = visit(AddRec->evaluateAtIteration(ItEnd, SE));
        if (!EndRes)
          return;
        if (BasePointer) {
          int64_t Start = StartRes->getSExtValue();
          int64_t End = EndRes->getSExtValue();
          if (Start > End)
            std::swap(Start, End);
          Inferred.addMemAccess(BasePointer, Start, End);
        }
      } else {
        // TODO: handle this
        dbgs() << "Got non-affine memory access: " << *Expr << "\n";
      }
    } else {
      Optional<APInt> Res = visit(Expr);
      if (!Res)
        return;
      int64_t Loc = Res->getSExtValue();
      if (BasePointer)
        Inferred.addMemAccess(BasePointer, Loc, Loc);
    }
  }
};

} // end anonymous namespace

static void inferInputs(const Function &F, Loop &L, ScalarEvolution &SE,
                        ExplorativeLVPass::InputBuilder &InferredInputs) {
  // Obtain/choose a trip count
  const SCEV *MaxBackedgeTakenCount = SE.getSymbolicMaxBackedgeTakenCount(&L);
  LLVM_DEBUG(dbgs() << "XLV: backedge taken count is " << *MaxBackedgeTakenCount
                    << "\n");
  Optional<APInt> BTRes = SCEVBackedgeTakenAnalyzer::analyze(
      SE, MaxBackedgeTakenCount, InferredInputs, XLVDesiredTripCount - 1);
  if (!BTRes) {
    LLVM_DEBUG(dbgs() << "XLV: could not obtain/choose a trip count\n");
    return;
  }
  const SCEV *ItStart = SE.getConstant(APInt(BTRes->getBitWidth(), 0));
  const SCEV *ItEnd = SE.getConstant(*BTRes);
  SCEVMemAccessAnalyzer MAAnalyzer(SE, InferredInputs, ItStart, ItEnd);

  // Look at all load an store instructions and determine memory to allocate
  for (const BasicBlock &BB : F) {
    for (const Instruction &I : BB) {
      const Value *Loc;
      if (const LoadInst *LI = dyn_cast<LoadInst>(&I)) {
        Loc = LI->getPointerOperand();
      } else if (const StoreInst *SI = dyn_cast<StoreInst>(&I)) {
        Loc = SI->getPointerOperand();

        // ignore output params
        if (const Argument *Arg = dyn_cast<Argument>(Loc)) {
          if (Arg->getArgNo() >= InferredInputs.NumInputArgs)
            continue;
        }
      } else {
        continue;
      }

      const SCEV *LocSCEV = SE.getSCEV(const_cast<Value *>(Loc));
      LLVM_DEBUG(dbgs() << "XLV: memory access at " << *LocSCEV << "\n");
      MAAnalyzer.analyze(LocSCEV);
    }
  }
}

PreservedAnalyses ExplorativeLVPass::run(Function &F,
                                         FunctionAnalysisManager &AM) {
  ExplorativeLVPass *MainPass = AM.getResult<ExplorativeLVAnalysis>(F).Pass;
  auto &LI = AM.getResult<LoopAnalysis>(F);
  auto &SE = AM.getResult<ScalarEvolutionAnalysis>(F);

  if (MainPass) { // "child" pass
    if (MainPass->InferredInputs) {
      auto LoopIt = LI.begin();
      if (LoopIt + 1 != LI.end()) {
        LLVM_DEBUG(dbgs() << "XLV: expected exactly one loop in loop function "
                          << F.getName() << "\n");
        return PreservedAnalyses::all();
      }
      inferInputs(F, **LoopIt, SE, *MainPass->InferredInputs);

      // The inputs should be inferred once only and not for every VF-IF
      // combination.
      MainPass->InferredInputs = nullptr;
    }
    return PreservedAnalyses::all();
  }

  // Further analyses needed to run simplifyLoop()
  auto &AC = AM.getResult<AssumptionAnalysis>(F);
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  // Needed to properly "clone" the pipelines
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
    OptPipeline = new OptPipelineContainer(TM, PTO, PGOOpt, this, TLI);
  // If this is a code size evaluation, we can re-use the same backend pipeline.
  if (!NullCGPipeline && XLVMetric == Metric::InstCount)
    NullCGPipeline = new NullCGPipelineContainer(*TM, this, TLI);

  unsigned LoopNo = 0;
  do {
    Loop *L = Worklist.pop_back_val();
    LLVM_DEBUG(dbgs() << "XLV: --- processing loop " << ++LoopNo << " in "
                      << F.getName() << " ---\n");
    if (processLoop(F, *L, SE, TLI)) {
      LLVM_DEBUG(dbgs() << "XLV: processing loop failed, letting the "
                           "AutoVectorizer determine VF and IF\n");
      addStringMetadataToLoop(L, "llvm.loop.vectorize.width", 0);
      addStringMetadataToLoop(L, "llvm.loop.interleave.count", 0);
    }
  } while (!Worklist.empty());

  return PreservedAnalyses::all();
}
