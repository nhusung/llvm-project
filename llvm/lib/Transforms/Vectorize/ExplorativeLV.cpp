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
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopIterator.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/Argument.h"
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
#include "llvm/IR/Metadata.h"
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
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/ObjCARC.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include <algorithm>
#include <ctime>
#include <random>
#include <system_error>

using namespace llvm;
#define DEBUG_TYPE "explorative-lv"

namespace llvm {

// Defined in TargetPassConfig.cpp
extern cl::opt<ExplorativeLVPass::Metric> XLVMetric;

} // namespace llvm

namespace {
enum class XLVClockid {
  MONOTONIC = CLOCK_MONOTONIC,
  PROCESS_CPUTIME_ID = CLOCK_PROCESS_CPUTIME_ID
};
} // end anonymous namespace

static cl::opt<unsigned>
    XLVMaxVF("xlv-max-vf", cl::Hidden, cl::init(16),
             cl::desc("The maximum vectorization factor to use for "
                      "explorative loop vectorization"));
static cl::opt<unsigned>
    XLVMaxIF("xlv-max-if", cl::Hidden, cl::init(8),
             cl::desc("The maximum interleaving factor to use for "
                      "explorative loop vectorization"));
static cl::opt<unsigned>
    XLVMaxUF("xlv-max-unroll", cl::Hidden, cl::init(8),
             cl::desc("The maximum unroll factor to use for explorative loop "
                      "vectorization, 0 to determine it automatically"));
static cl::opt<unsigned> XLVMaxFullUnroll(
    "xlv-max-full-unroll", cl::Hidden, cl::init(0),
    cl::desc("If the loop's trip count is known and below this threshold, "
             "explore the fully unrolled loop as well"));

static cl::opt<unsigned> XLVDesiredTripCount(
    "xlv-desired-tripcount", cl::Hidden, cl::init(1024),
    cl::desc("In benchmarking mode: try to run the loop with this trip count"));
static cl::opt<uint64_t> XLVBenchmarkUSecs(
    "xlv-benchmark-usecs", cl::Hidden, cl::init(100000),
    cl::desc("In benchmarking mode: microseconds for calibration run"));
static cl::opt<XLVClockid> XLVBenchmarkClockid(
    "xlv-benchmark-clock", cl::Hidden,
    cl::desc("In benchmarking mode: clock to use, see clock_gettime(3)"),
    cl::init(XLVClockid::PROCESS_CPUTIME_ID),
    cl::values(clEnumValN(XLVClockid::MONOTONIC, "monotonic",
                          "CLOCK_MONOTONIC"),
               clEnumValN(XLVClockid::PROCESS_CPUTIME_ID, "process",
                          "CLOCK_PROCESS_CPUTIME_ID")));
static cl::opt<unsigned> XLVBenchmarkWarmup(
    "xlv-benchmark-warmup", cl::Hidden, cl::init(16),
    cl::desc("In benchmarking mode: how many times to call each loop function "
             "before benchmarking"));
static cl::opt<unsigned>
    XLVBenchmarkLoopSize("xlv-benchmark-loop-size", cl::Hidden, cl::init(32),
                         cl::desc("In benchmarking mode: size (loop "
                                  "function calls) of the benchmark loop"));
static cl::opt<bool> XLVBenchmarkRandOrder(
    "xlv-benchmark-rand-order", cl::Hidden, cl::init(false),
    cl::desc("In benchmarking mode: randomize the loop order"));
static cl::opt<bool>
    XLVBenchmarkKeepExec("xlv-benchmark-keep-exec", cl::Hidden, cl::init(false),
                         cl::desc("In benchmarking mode: keep the executable"));

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
static cl::opt<std::string>
    XLVCSVOut("xlv-csv-out", cl::Hidden, cl::init(""),
              cl::desc("Print explorative loop vectorization results as CSV to "
                       "the given file"));

class ExplorativeLVPass::InputBuilder {
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

  SmallVector<ConstantInt *> IntInputs;
  SmallVector<Optional<std::pair<unsigned, int64_t>>> MemAccesses;
  SmallVector<MemAccessRange> MemRanges;

public:
  void addMemAccess(const Argument *Arg, int64_t Loc);
  void addMemAccessWithEndptr(const Argument *ArgStart, const Argument *ArgEnd,
                              uint64_t Diff);

  APInt tryAddIntInput(const Argument *Arg, APInt &&Desired) {
    ConstantInt *&Input = IntInputs[Arg->getArgNo()];
    if (Input)
      return Input->getValue();

    Input = ConstantInt::get(Arg->getType()->getContext(), Desired);
    return Desired;
  }

  void build(Module &M, ArrayRef<Type *> Params,
             SmallVectorImpl<Value *> &Args);

  void setNumInputArgs(unsigned Num) {
    assert(IntInputs.size() == 0 && MemAccesses.size() == 0 &&
           "Must set NumInputArgs once only");
    IntInputs.assign(Num, nullptr);
    MemAccesses.assign(Num, None);
  }

  bool isInputArg(const Argument *Arg) {
    return Arg->getArgNo() < IntInputs.size();
  }
};
using InputBuilder = ExplorativeLVPass::InputBuilder;

void InputBuilder::addMemAccess(const Argument *Arg, int64_t Loc) {
  auto &MemAccess = MemAccesses[Arg->getArgNo()];
  if (MemAccess) { // Already present, just extend the range (if necessary)
    MemAccessRange &Range = MemRanges[MemAccess->first];
    if (Loc < Range.Start) {
      Range.Start = Loc;
    } else if (Loc > Range.End) {
      Range.End = Loc;
    }
  } else { // New access
    MemAccess.emplace(MemRanges.size(), /* offset */ 0);

    // Sanitize the range.  Even if the memory access is in [Start, End], we
    // also want the pointer itself to be in [Start, End].
    MemRanges.emplace_back(Loc > 0 ? 0 : Loc, Loc < 0 ? 0 : Loc,
                           Arg->getType()->getNonOpaquePointerElementType());
  }
}

void InputBuilder::addMemAccessWithEndptr(const Argument *ArgStart,
                                          const Argument *ArgEnd,
                                          uint64_t Diff) {
  assert(ArgStart->getType() == ArgEnd->getType() && "types must agree");
  auto &MemAccessStart = MemAccesses[ArgStart->getArgNo()];
  auto &MemAccessEnd = MemAccesses[ArgEnd->getArgNo()];
  assert(!MemAccessStart && !MemAccessEnd &&
         "currently we only support adding new mem accesses with end pointer");

  unsigned RangeID = MemRanges.size();
  MemAccessStart.emplace(RangeID, /* offset */ 0);
  MemAccessEnd.emplace(RangeID, Diff);

  MemRanges.emplace_back(0, Diff - 1,
                         ArgStart->getType()->getNonOpaquePointerElementType());
}

void InputBuilder::build(Module &M, ArrayRef<Type *> Params,
                         SmallVectorImpl<Value *> &Args) {
  for (MemAccessRange &Range : MemRanges) {
    ArrayType *Ty =
        ArrayType::get(Range.ElementType, Range.End + 1 - Range.Start);
    Range.GV = new GlobalVariable(M, Ty, false, GlobalValue::PrivateLinkage,
                                  ConstantAggregateZero::get(Ty));
  }

  Type *I64Ty = Type::getInt64Ty(M.getContext());
  Constant *I640 = ConstantInt::get(I64Ty, 0);

  for (unsigned I = 0, E = IntInputs.size(); I < E; ++I) {
    Value *ArgVal;
    if (ConstantInt *IntInput = IntInputs[I]) { // Inferred integers
      ArgVal = IntInput;
    } else if (MemAccesses[I]) { // Inferred pointers (now to global variables)
      auto &Access = MemAccesses[I].getValue();
      unsigned RangeID = Access.first;
      MemAccessRange &Range = MemRanges[RangeID];
      GlobalVariable *GV = Range.GV;
      uint64_t Offset = Access.second - Range.Start;

      ArgVal = ConstantExpr::getInBoundsGetElementPtr(
          GV->getValueType(), GV,
          ArrayRef<Constant *>{I640, ConstantInt::get(I64Ty, Offset)});
    } else { // Fall back to null values
      ArgVal = Constant::getNullValue(Params[I]);
    }

    LLVM_DEBUG(dbgs() << "XLV: arg " << *ArgVal << '\n');
    Args.push_back(ArgVal);
  }
}

namespace {

class LoopModuleBuilder : public ValueMaterializer {
  const Function &OrigFunc;
  const Loop &L;
  ScalarEvolution &SE;

  Module *M = nullptr;
  ValueToValueMapTy VMap;
  ValueMapper Mapper;
  SmallVector<const Value *> Inputs;
  SmallVector<const Instruction *> Outputs;
  FunctionType *FuncTy;
  Function *RefLoopFunc = nullptr;
  DenseMap<Function *, ExplorativeLVPass::EvaluationInfo> LoopFuncs;

  InputBuilder InferredInputs;
  unsigned FullUnroll = 0;

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

  Function *buildLoopFunc(unsigned VF, unsigned IF, unsigned UF);
  unsigned getNumLoopFuncs() { return LoopFuncs.size(); }
  DenseMap<Function *, ExplorativeLVPass::EvaluationInfo> *getLoopFuncs() {
    return &LoopFuncs;
  }
  void clearLoopFuncs() {
    for (auto &KV : LoopFuncs)
      KV.first->eraseFromParent();
    LoopFuncs.clear();
  }

  InputBuilder &getInferredInputsRef() {
    assert(M && "No loop func has been built yet");
    return InferredInputs;
  }
  unsigned getFullUnrollCount() { return FullUnroll; }
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
        if (!isa<IntrinsicInst>(I)) {
          LLVM_DEBUG(dbgs()
                     << "XLV: loop contains a non-intrinsic call instruction: "
                     << I << '\n');
          return false;
        }

        // TODO: are there intrinsics we should forbid?
      }

      // Used inside loop and defined outside?
      for (const Value *Op : I.operand_values()) {
        const Instruction *OpInst = dyn_cast<Instruction>(Op);
        if (OpInst) {
          if (L.getBlocksSet().contains(OpInst->getParent()))
            continue;
        } else if (!isa<Argument>(Op)) {
          continue;
        }
        if (ArgSet.contains(Op))
          continue;
        ArgSet.insert(Op);
        Inputs.push_back(Op);
      }

      // Defined inside loop and used outside?
      if (!ArgSet.contains(&I)) {
        for (const User *U : I.users()) {
          const Instruction *UI = dyn_cast<Instruction>(U);
          if (!UI)
            continue; // ignore users that aren't instructions
          if (!L.getBlocksSet().contains(UI->getParent())) {
            ArgSet.insert(&I);
            Outputs.push_back(&I);
            break;
          }
        }
      }
    }
  }

  InferredInputs.setNumInputArgs(Inputs.size());

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

  if (MakeExecutable) {
    const SCEV *BackedgeTaken = SE.getSymbolicMaxBackedgeTakenCount(&L);
    if (isa<SCEVCouldNotCompute>(BackedgeTaken)) {
      LLVM_DEBUG(dbgs() << "XLV: cannot infer trip count\n");
      return nullptr;
    }
    if (const SCEVConstant *Const = dyn_cast<SCEVConstant>(BackedgeTaken)) {
      unsigned ConstBT = Const->getAPInt().getLimitedValue(UINT_MAX - 1) + 1;
      if (ConstBT > XLVMaxUF && ConstBT <= XLVMaxFullUnroll)
        FullUnroll = ConstBT;
    }
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
Function *LoopModuleBuilder::buildLoopFunc(unsigned VF, unsigned IF,
                                           unsigned UF) {
  Function *LoopFunc = Function::Create(
      FuncTy, GlobalValue::ExternalLinkage, OrigFunc.getAddressSpace(),
      OrigFunc.getName() + "." + Twine(VF) + "." + Twine(IF) + "." + Twine(UF),
      M);

  LLVMContext &C = LoopFunc->getContext();

  /* Insert the generated function arguments into the VMap, add names */ {
    auto *ArgPtr = LoopFunc->arg_begin();
    for (const Value *OrigV : Inputs) {
      VMap[OrigV] = ArgPtr;
      VMap[ArgPtr] = ArgPtr;
      ArgPtr->setName(OrigV->hasName() ? OrigV->getName() + ".i" : "i");
      ++ArgPtr;
    }
    for (const Instruction *I : Outputs) {
      VMap[ArgPtr] = ArgPtr;
      ArgPtr->setName(I->hasName() ? I->getName() + ".o" : "o");
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
    if (const Argument *NewArg = dyn_cast<Argument>(VMap[&OldArg])) {
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
    for (Instruction &I : BB)
      Mapper.remapInstruction(I);
  }

  // Restore MDMap
  VMap.MD().swap(MDModuleMap);

  LoopFuncs.try_emplace(LoopFunc, VF, IF, UF);
  if (VF == 1 && IF == 1 && UF <= 1) {
    RefLoopFunc = LoopFunc;

    if (XLVDumpFuncIR)
      dbgs() << *LoopFunc << '\n';
  }

  // Note: We did not set the desired VF/IF/UF here.  This will happen in the
  // "child" pass (ExplorativeLVPass::run()) since otherwise, the unrolling
  // might be performed before loop vectorization.

  return LoopFunc;
}

void LoopModuleBuilder::buildMainFuncBody() {
  LLVMContext &C = M->getContext();

  // Some types and values we will use a few times
  Type *IntTy = Type::getIntNTy(C, sizeof(int) * 8);
  Type *I32Ty = Type::getInt32Ty(C);
  Type *I64Ty = Type::getInt64Ty(C);
  Type *ClockidTy = Type::getIntNTy(C, sizeof(clockid_t) * 8);
  static_assert(sizeof(time_t) <= 8, "time_t larger than expected");
  Type *TimeTy = Type::getIntNTy(C, sizeof(time_t) * 8);
  static_assert(sizeof(long) <= 8, "long larger than expected");
  Type *LongTy = Type::getIntNTy(C, sizeof(long) * 8);
  Type *TimespecTy = StructType::create({TimeTy, LongTy}, "struct.timespec");
  Type *TimespecPtrTy = PointerType::get(TimespecTy, 0);
  Constant *Int0 = ConstantInt::get(IntTy, 0);
  Constant *Int1 = ConstantInt::get(IntTy, 1);
  Constant *TimespecSecIndex = ConstantInt::get(I32Ty, 0);
  Constant *TimespecNSecIndex = ConstantInt::get(I32Ty, 1);
  Constant *I640 = ConstantInt::get(I64Ty, 0);
  Constant *Nanos = ConstantInt::get(I64Ty, 1000000000);
  Constant *Clockid =
      ConstantInt::get(ClockidTy, (uint64_t)XLVBenchmarkClockid.getValue());

  assert(RefLoopFunc && "Must generate a function with VF 1, IF 1 and UF 0/1");
  FunctionType *LoopFuncTy = RefLoopFunc->getFunctionType();
  PointerType *LoopFuncPtrTy = RefLoopFunc->getType();

  // Declare `int clock_gettime(clockid_t clock_id, struct timespec *res)`
  FunctionType *ClockGettimeFuncTy =
      FunctionType::get(IntTy, {ClockidTy, TimespecPtrTy}, false);
  Function *ClockGettimeFunc = Function::Create(
      ClockGettimeFuncTy, GlobalValue::ExternalLinkage, 0, "clock_gettime", M);

  // Declare `int printf(char *format, ...)`
  FunctionType *PrintfFuncTy =
      FunctionType::get(IntTy, {Type::getInt8PtrTy(C)}, true);
  Function *PrintfFunc =
      Function::Create(PrintfFuncTy, GlobalValue::ExternalLinkage,
                       OrigFunc.getAddressSpace(), "printf", M);

  // We build a benchmarking function that takes a loop function and an integer
  // n as argument and calls the loop function n * ExploreBenchmarkLoopSize
  // times.  The function furthermore takes the VF, IF and UF, which are just
  // printed along with the time.  The benchmarking function itself is called by
  // `int main(void)`.
  FunctionType *BenchFuncTy = FunctionType::get(
      Type::getVoidTy(C), {LoopFuncPtrTy, IntTy, IntTy, IntTy, IntTy}, false);
  Function *BenchFunc =
      Function::Create(BenchFuncTy, GlobalValue::PrivateLinkage, "bench", M);
  // To infer a suitable n, we build a calibration function that counts the
  // repitions needed until the elapsed time exceeds some threshold.
  FunctionType *CalibFuncTy = FunctionType::get(IntTy, {LoopFuncPtrTy}, false);
  Function *CalibFunc =
      Function::Create(CalibFuncTy, GlobalValue::PrivateLinkage, "calib", M);

  Argument *FuncArg = BenchFunc->getArg(0);
  FuncArg->setName("func");
  Argument *NArg = BenchFunc->getArg(1);
  NArg->setName("n");
  Argument *VFArg = BenchFunc->getArg(2);
  VFArg->setName("vf");
  Argument *IFArg = BenchFunc->getArg(3);
  IFArg->setName("if");
  Argument *UFArg = BenchFunc->getArg(4);
  UFArg->setName("uf");
  Argument *CFuncArg = CalibFunc->getArg(0);
  CFuncArg->setName("func");

  BasicBlock *StartBB = BasicBlock::Create(C, "bench_start", BenchFunc);
  BasicBlock *CStartBB = BasicBlock::Create(C, "calib_start", CalibFunc);

  SmallVector<Value *> Args;
  SmallVector<Value *> CArgs;
  /* Create arguments */ {
    unsigned NumParams = LoopFuncTy->getNumParams();
    Args.reserve(NumParams);
    CArgs.reserve(NumParams);

    InferredInputs.build(*M, LoopFuncTy->params(), Args);
    CArgs = Args;

    // Allocate locations for output parameters
    assert(Inputs.size() <= LoopFuncTy->getNumParams());
    auto *It = LoopFuncTy->param_begin() + Inputs.size();
    for (auto *End = LoopFuncTy->param_end(); It != End; ++It) {
      Type *ElementType = (*It)->getNonOpaquePointerElementType();
      Args.push_back(new AllocaInst(ElementType, BenchFunc->getAddressSpace(),
                                    "outarg", StartBB));
      CArgs.push_back(new AllocaInst(ElementType, CalibFunc->getAddressSpace(),
                                     "outarg", CStartBB));
    }
  }

  // Allocate timespec structs
  AllocaInst *TStartVar = new AllocaInst(TimespecTy, 0, "ts", StartBB);
  AllocaInst *TEndVar = new AllocaInst(TimespecTy, 0, "te", StartBB);
  AllocaInst *CTStartVar = new AllocaInst(TimespecTy, 0, "ts", CStartBB);
  AllocaInst *CTEndVar = new AllocaInst(TimespecTy, 0, "te", CStartBB);

  // Warmup
  for (unsigned I = 0; I < XLVBenchmarkWarmup; ++I) {
    CallInst::Create(LoopFuncTy, FuncArg, Args, "", StartBB);
    CallInst::Create(LoopFuncTy, CFuncArg, CArgs, "", CStartBB);
  }

  // Start measuring
  CallInst::Create(ClockGettimeFuncTy, ClockGettimeFunc, {Clockid, TStartVar},
                   "", StartBB);
  CallInst::Create(ClockGettimeFuncTy, ClockGettimeFunc, {Clockid, CTStartVar},
                   "", CStartBB);

  auto TimeDiff = [=](AllocaInst *StartVar, AllocaInst *EndVar,
                      BasicBlock *BB) {
    CallInst::Create(ClockGettimeFuncTy, ClockGettimeFunc, {Clockid, EndVar},
                     "", BB);

    GetElementPtrInst *Ptr = GetElementPtrInst::CreateInBounds(
        TimespecTy, EndVar, {I640, TimespecSecIndex}, "te.sec_ptr", BB);
    Value *EndSec = new LoadInst(TimeTy, Ptr, "te.sec", BB);
    Ptr = GetElementPtrInst::CreateInBounds(
        TimespecTy, StartVar, {I640, TimespecSecIndex}, "ts.sec_ptr", BB);
    Value *StartSec = new LoadInst(TimeTy, Ptr, "ts.sec", BB);
    Value *DiffSec = BinaryOperator::CreateSub(EndSec, StartSec, "td.sec", BB);
    if (sizeof(time_t) < 8)
      DiffSec = new ZExtInst(DiffSec, I64Ty, "td.sec64", BB);
    Value *DiffNSec = BinaryOperator::CreateMul(DiffSec, Nanos, "", BB);

    Ptr = GetElementPtrInst::CreateInBounds(
        TimespecTy, EndVar, {I640, TimespecNSecIndex}, "te.nsec_ptr", BB);
    Value *EndNSec = new LoadInst(LongTy, Ptr, "te.nsec", BB);
    if (sizeof(long) < 8)
      EndNSec = new ZExtInst(EndNSec, I64Ty, "te.nsec64", BB);
    DiffNSec = BinaryOperator::CreateAdd(DiffNSec, EndNSec, "", BB);

    Ptr = GetElementPtrInst::CreateInBounds(
        TimespecTy, StartVar, {I640, TimespecNSecIndex}, "ts.nsec_ptr", BB);
    Value *StartNSec = new LoadInst(LongTy, Ptr, "ts.nsec", BB);
    if (sizeof(long) < 8)
      StartNSec = new ZExtInst(EndNSec, I64Ty, "te.nsec64", BB);
    return BinaryOperator::CreateSub(DiffNSec, StartNSec, "td", BB);
  };

  // Benchmarking loop
  BasicBlock *LoopBB = BasicBlock::Create(C, "bench_loop", BenchFunc);
  BranchInst::Create(LoopBB, StartBB);
  PHINode *CountPhi = PHINode::Create(IntTy, 2, "counter", LoopBB);
  CountPhi->addIncoming(Int1, StartBB);

  BasicBlock *CLoopBB = BasicBlock::Create(C, "calib_loop", CalibFunc);
  BranchInst::Create(CLoopBB, CStartBB);
  PHINode *CCountPhi = PHINode::Create(IntTy, 2, "counter", CLoopBB);
  CCountPhi->addIncoming(Int1, CStartBB);

  for (unsigned I = 0; I < XLVBenchmarkLoopSize; ++I) {
    CallInst::Create(LoopFuncTy, FuncArg, Args, "", LoopBB);
    CallInst::Create(LoopFuncTy, CFuncArg, CArgs, "", CLoopBB);
  }

  // Counters and exit conditions
  ICmpInst *ContLoop = new ICmpInst(*LoopBB, CmpInst::ICMP_EQ, CountPhi, NArg);
  BinaryOperator *CountAdd =
      BinaryOperator::CreateAdd(CountPhi, Int1, "counter_next", LoopBB);

  Value *CTDiff = TimeDiff(CTStartVar, CTEndVar, CLoopBB);
  ICmpInst *CContLoop =
      new ICmpInst(*CLoopBB, CmpInst::ICMP_ULT, CTDiff,
                   ConstantInt::get(I64Ty, 1000 * XLVBenchmarkUSecs));
  BinaryOperator *CCountAdd =
      BinaryOperator::CreateAdd(CCountPhi, Int1, "counter_next", CLoopBB);

  // Branch at the end of the loop
  BasicBlock *EvalBB = BasicBlock::Create(C, "bench_eval", BenchFunc);
  BranchInst::Create(EvalBB, LoopBB, ContLoop, LoopBB);
  CountPhi->addIncoming(CountAdd, LoopBB);

  BasicBlock *CEvalBB = BasicBlock::Create(C, "calib_ret", CalibFunc);
  BranchInst::Create(CLoopBB, CEvalBB, CContLoop, CLoopBB);
  CCountPhi->addIncoming(CCountAdd, CLoopBB);

  // Stop measuring
  Value *TDiff = TimeDiff(TStartVar, TEndVar, EvalBB);

  // Create format strings
  Constant *FormatStr = ConstantDataArray::getString(C, "%u %u %u %llu\n");
  Constant *CFormatStr = ConstantDataArray::getString(C, "%llu %u\n");
  GlobalVariable *FormatStrGV =
      new GlobalVariable(*M, FormatStr->getType(), true,
                         GlobalValue::PrivateLinkage, FormatStr, "fstr");
  GlobalVariable *CFormatStrGV =
      new GlobalVariable(*M, CFormatStr->getType(), true,
                         GlobalValue::PrivateLinkage, CFormatStr, "cfstr");
  Constant *FormatStrPtr = ConstantExpr::getInBoundsGetElementPtr(
      FormatStrGV->getValueType(), FormatStrGV,
      ArrayRef<Constant *>{I640, I640});
  Constant *CFormatStrPtr = ConstantExpr::getInBoundsGetElementPtr(
      CFormatStrGV->getValueType(), CFormatStrGV,
      ArrayRef<Constant *>{I640, I640});

  // `printf(fstr, vf, if, uf, dt); return;` / `printf(fstr, dt, n); return;`
  CallInst::Create(PrintfFuncTy, PrintfFunc,
                   {FormatStrPtr, VFArg, IFArg, UFArg, TDiff}, "", EvalBB);
  ReturnInst::Create(C, EvalBB);

  CallInst::Create(PrintfFuncTy, PrintfFunc, {CFormatStrPtr, CTDiff, CCountPhi},
                   "", CEvalBB);
  ReturnInst::Create(C, CCountPhi, CEvalBB);

  // Declare & define `int main(void)`
  Function *MainFunc = Function::Create(
      FunctionType::get(IntTy, false), GlobalValue::ExternalLinkage, "main", M);

  BasicBlock *MainBB = BasicBlock::Create(C, "main", MainFunc);
  CallInst *NRuns =
      CallInst::Create(CalibFuncTy, CalibFunc, {RefLoopFunc}, "n", MainBB);
  for (auto &KV : LoopFuncs) {
    Constant *VF = ConstantInt::get(IntTy, KV.second.VF);
    Constant *IF = ConstantInt::get(IntTy, KV.second.IF);
    Constant *UF = ConstantInt::get(IntTy, KV.second.UF);
    CallInst::Create(BenchFuncTy, BenchFunc, {KV.first, NRuns, VF, IF, UF}, "",
                     MainBB);
  }

  ReturnInst::Create(C, Int0, MainBB);
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

struct ExplorativeLVPass::CSVOutputContainer {
  std::error_code EC;
  raw_fd_ostream OS;
  std::time_t Timestamp = std::time(nullptr);
  CSVOutputContainer() : OS(XLVCSVOut, EC, sys::fs::OpenFlags::OF_Append) {}
};

ExplorativeLVPass::~ExplorativeLVPass() {
  delete NullCGPipeline;
  delete OptPipeline;
  delete CSVOutput;
}

// returns whether an error occured
bool ExplorativeLVPass::processLoop(Function &F, Loop &L, unsigned LoopNo,
                                    ScalarEvolution &SE,
                                    TargetLibraryInfo &TLI) {
  assert(OptPipeline && "OptPipeline not initialized");

  LoopModuleBuilder Builder(F, L, SE, XLVMetric == Metric::Benchmark);
  Module *M = Builder.getModule();
  if (!M)
    return true;

  EvaluationInfoMap = Builder.getLoopFuncs();

  unsigned BestVF = 1;
  unsigned BestIF = 1;
  unsigned BestUF = 0;
  uint64_t MinCosts = InvalidCosts;

  auto UpdateBest = [this, Func = F.getName(), LoopNo, &BestVF, &BestIF,
                     &BestUF, &MinCosts](unsigned VF, unsigned IF, unsigned UF,
                                         uint64_t Costs,
                                         uint64_t Reference = InvalidCosts) {
    if (CSVOutput) {
      // We use the following columns:
      // - timestamp (secs since unix epoch)
      // - metric ("inst-count"/"mca"/"benchmark")
      // - function name
      // - loop number
      // - VF
      // - IF
      // - UF
      // - costs (empty if invalid)
      // - reference (number of runs if in benchmarking mode, else empty)
      CSVOutput->OS << CSVOutput->Timestamp << ',';
      switch (XLVMetric) {
      case Metric::Disable:
        llvm_unreachable("XLVMetric is Disable in evaluation");
      case Metric::InstCount:
        CSVOutput->OS << "inst-count,";
        break;
      case Metric::MCA:
        CSVOutput->OS << "mca,";
        break;
      case Metric::Benchmark:
        CSVOutput->OS << "benchmark,";
        break;
      }
      CSVOutput->OS << Func << ',' << LoopNo << ',' << VF << ',' << IF << ','
                    << UF << ',';
      if (Costs != InvalidCosts)
        CSVOutput->OS << Costs;
      CSVOutput->OS << ',';
      if (Reference != InvalidCosts)
        CSVOutput->OS << Reference;
      CSVOutput->OS << '\n';
    }

    if (Costs == InvalidCosts) {
      LLVM_DEBUG(dbgs() << "XLV: costs for VF " << VF << ", IF " << IF
                        << ", UF " << UF << ": (invalid)\n");
      return;
    }

    // FIXME: We should remove this switch.  For the InstCount and MCA metric,
    // we need to take into account how much work is done by the given
    // instructions.  This is probably more complex than just dividing by the
    // VF.  We should also multiply the costs first to avoid loss of precision.
    // In benchmarking mode this switch does not make sense at all.
    if (XLVDivVF)
      Costs /= VF;

    LLVM_DEBUG(dbgs() << "XLV: costs for VF " << VF << ", IF " << IF << ", UF "
                      << UF << ": " << Costs << '\n');

    if (Costs < MinCosts) {
      MinCosts = Costs;
      BestVF = VF;
      BestIF = IF;
      BestUF = UF;
    } else if (Costs == MinCosts && (VF == 1 || VF >= BestVF) && IF >= BestIF &&
               UF <= BestUF) {
      // If costs are equal to costs retrieved before, always choose the
      // higher VF-IF combination unless best VF is scalar.  We always choose
      // the lower UF for the sake of smaller code size.
      BestVF = VF;
      BestIF = IF;
      BestUF = UF;
    }
  };

  auto ForeachFactor = [FullUnroll = Builder.getFullUnrollCount()](auto Body) {
    for (unsigned VF = 1; VF <= XLVMaxVF; VF *= 2) {
      for (unsigned IF = 1; IF <= XLVMaxIF; IF *= 2) {
        if (XLVMaxUF == 0) {
          Body(VF, IF, 0);
          continue;
        }
        for (unsigned UF = 1; UF <= XLVMaxUF; UF *= 2)
          Body(VF, IF, UF);
        if (FullUnroll)
          Body(VF, IF, FullUnroll);
      }
    }
  };

  if (XLVMetric == Metric::InstCount) {
    // Build a module containing functions for all VF/IF combinations.  We only
    // consider UF 1, everything else is not interesting when counting
    // instructions.
    for (unsigned VF = 1; VF <= XLVMaxVF; VF *= 2) {
      for (unsigned IF = 1; IF <= XLVMaxIF; IF *= 2)
        Builder.buildLoopFunc(VF, IF, 1);
    }

    // Run pipelines
    OptPipeline->run(*M);
    NullCGPipeline->PM.run(*M);

    for (auto &KV : *EvaluationInfoMap) {
      EvaluationInfo &Info = KV.second;
      UpdateBest(Info.VF, Info.IF, Info.UF, Info.Costs);
    }
  } else if (XLVMetric == Metric::MCA) {
    ForeachFactor([this, &Builder, M, TLI,
                   &UpdateBest](unsigned VF, unsigned IF, unsigned UF) {
      Builder.buildLoopFunc(VF, IF, UF);
      OptPipeline->run(*M);

      // Expensive, but: If we are using llvm-mca for evaluation, we need a new
      // backend pipeline for each iteration as the print stream is dead after
      // one use
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
      uint64_t Costs =
          performMCACostCalc(ASMPath, TM->getTargetCPU(), M->getTargetTriple());
      UpdateBest(VF, IF, UF, Costs);

      Builder.clearLoopFuncs();
    });
  } else {
    assert(XLVMetric == Metric::Benchmark);

    /* Build a module containing functions for all VF/IF/UF combinations */ {
      SmallVector<std::tuple<unsigned, unsigned, unsigned>> Factors;
      ForeachFactor([&Factors](unsigned VF, unsigned IF, unsigned UF) {
        Factors.emplace_back(VF, IF, UF);
      });
      if (XLVBenchmarkRandOrder) {
        std::random_device RD;
        std::default_random_engine RNG(RD());
        std::shuffle(Factors.begin(), Factors.end(), RNG);
      }
      for (auto &Fs : Factors)
        Builder.buildLoopFunc(std::get<0>(Fs), std::get<1>(Fs),
                              std::get<2>(Fs));
    }

    // Get a reference to the InputBuilder.  The ExplorativeLVPass in the
    // optimization pipeline will (try to) infer the arguments.
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
      errs() << "XLV: linking failed.  You can find the object file at "
             << ObjectPath << '\n';
      ObjectRemover.releaseFile();
      return true;
    }
    if (XLVBenchmarkKeepExec) {
      dbgs() << "XLV: keeping the benchmarking executable at " << ExecPath
             << '\n';
      ExecRemover.releaseFile();
    }

    // Do the benchmarking
    // Timout: ceil(2 * (calib_secs + n * bench_secs)))
    unsigned FuncCount = Builder.getNumLoopFuncs();
    unsigned TimeoutS = 1 + 2 * (FuncCount + 1) * XLVBenchmarkUSecs / 1000000;
    LLVM_DEBUG(dbgs() << "XLV: benchmarking for up to " << TimeoutS
                      << " s ...\n");
    SmallString<32> BenchResPath;
    sys::fs::createTemporaryFile("explorative-lv-out", "txt", BenchResPath);
    FileRemover BenchResRemover(BenchResPath);
    int RetCode = sys::ExecuteAndWait(
        ExecPath, {"explorative-lv"}, None,
        {StringRef(""), BenchResPath.str(), StringRef("")}, TimeoutS);
    if (RetCode == -2) {
      errs() << "XLV: benchmarking failed/timed out\n";
      return true;
    }
    if (RetCode == -1) {
      errs() << "XLV: failed to execute " << ExecPath << '\n';
      ExecRemover.releaseFile();
      return true;
    }

    auto BenchResBuf = MemoryBuffer::getFile(BenchResPath);
    if (!BenchResBuf) {
      errs() << "XLV: could not load benchmarking results\n";
      return true;
    }
    StringRef BenchRes = BenchResBuf.get()->getBuffer();

    // Read calibration info
    uint64_t CalibTime;
    unsigned NRuns;
    if (BenchRes.consumeInteger(10, CalibTime) ||
        !BenchRes.consume_front(" ") || BenchRes.consumeInteger(10, NRuns)) {
      errs() << "XLV: malformed benchmarking results\n";
      return true;
    }
    LLVM_DEBUG(dbgs() << "XLV: calibration run took " << CalibTime
                      << " ns, executed each loop function "
                      << XLVBenchmarkLoopSize * NRuns << " times\n");

    for (unsigned I = 0; I < FuncCount; ++I) {
      BenchRes = BenchRes.drop_front(); // drop '\n'

      unsigned VF, IF, UF;
      uint64_t Costs;
      if (BenchRes.consumeInteger(10, VF) || !BenchRes.consume_front(" ") ||
          BenchRes.consumeInteger(10, IF) || !BenchRes.consume_front(" ") ||
          BenchRes.consumeInteger(10, UF) || !BenchRes.consume_front(" ") ||
          BenchRes.consumeInteger(10, Costs)) {
        errs() << "XLV: malformed benchmarking results\n";
        return true;
      }

      UpdateBest(VF, IF, UF, Costs, NRuns);
    }
  }

  // Force the vectorizer to use the VF and IF that showed to have the lowest
  // cost
  LLVM_DEBUG(dbgs() << "XLV: choosing VF " << BestVF << ", IF " << BestIF
                    << ", UF " << BestUF << '\n');
  addStringMetadataToLoop(&L, "llvm.loop.vectorize.width", BestVF);
  addStringMetadataToLoop(&L, "llvm.loop.interleave.count", BestIF);
  if (BestUF)
    addStringMetadataToLoop(&L, "llvm.loop.unroll.count", BestUF);

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
  /// pointer to determine the trip count.  Note that we require arguments here.
  /// Global variables should not occur, because then the trip count should be
  /// constant.
  const Argument *InferredPtrStart = nullptr;

  /// End pointer value
  const Argument *InferredPtrEnd = nullptr;

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
    LLVM_DEBUG(dbgs() << "XLV: found SCEVAddRecExpr when analyzing backedge "
                         "taken count\n");
    return None;
  }

  Optional<APInt> visitUnknown(const SCEVUnknown *Unknown, APInt Desired) {
    const Argument *Arg = dyn_cast<Argument>(Unknown->getValue());
    if (!Arg) {
      // This case should not occur.  Global variables should not occur, because
      // then the trip count should be constant.  And in our loop function, no
      // other values are live outside the loop.
      LLVM_DEBUG(dbgs() << "XLV: found non-argument unknown " << Unknown
                        << " when analyzing backedge taken count\n");
      return None;
    }
    Type *Ty = Arg->getType();

    if (Ty->isIntegerTy())
      return Inferred.tryAddIntInput(Arg, std::move(Desired));

    assert(Ty->isPointerTy() &&
           "SCEV analysis deals with integer and pointer types only?");

    if (Desired.isZero()) {
      if (!InferredPtrStart) {
        InferredPtrStart = Arg;
        return Desired;
      }
      if (InferredPtrStart == Arg)
        return Desired;
    } else {
      if (!InferredPtrEnd) {
        InferredPtrEnd = Arg;
        InferredPtrDiff = Desired.getZExtValue();
        return Desired;
      }
      if (InferredPtrEnd == Arg)
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
      unsigned BytesPerElement = SE.getDataLayout().getTypeAllocSize(
          Analyzer.InferredPtrStart->getType()
              ->getNonOpaquePointerElementType());
      Inferred.addMemAccessWithEndptr(
          Analyzer.InferredPtrStart, Analyzer.InferredPtrEnd,
          Analyzer.InferredPtrDiff / BytesPerElement);
    }
    return Res;
  }
};

class SCEVMemAccessAnalyzer : private SCEVCompVisitor<SCEVMemAccessAnalyzer> {
  friend class SCEVCompVisitor<SCEVMemAccessAnalyzer>;

  ScalarEvolution &SE;
  InputBuilder &Inferred;
  const SCEVConstant *ItStart;
  const SCEVConstant *ItCurrent;
  const SCEVConstant *ItEnd;

  enum { NoAddRec, MonotoneAddRec, ArbitraryAddRec } AddRecKind;

  const Argument *BasePointer = nullptr;

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
    if (ItCurrent == ItStart && AddRecKind < ArbitraryAddRec) {
      AddRecKind = MonotoneAddRec;
      if (!Expr->isAffine()) {
        const auto *It = Expr->op_begin() + 1;
        const auto *End = Expr->op_end();
        for (; It != End; ++It) {
          const SCEVConstant *Const = dyn_cast<SCEVConstant>(*It);
          if (!Const || Const->getAPInt().slt(0)) {
            AddRecKind = ArbitraryAddRec;
            break;
          }
        }
      }
    }
    return visit(Expr->evaluateAtIteration(ItCurrent, SE));
  }

  Optional<APInt> visitUnknown(const SCEVUnknown *Unknown) {
    const Argument *Arg = dyn_cast<Argument>(Unknown->getValue());
    if (!Arg)
      return None;

    Type *Ty = Arg->getType();

    if (const IntegerType *IntTy = dyn_cast<IntegerType>(Ty))
      return Inferred.tryAddIntInput(Arg, APInt(IntTy->getBitWidth(), 1));

    assert(Ty->isPointerTy() &&
           "SCEV analysis deals with integer and pointer types only?");

    if (BasePointer != Arg) {
      if (BasePointer) {
        LLVM_DEBUG(dbgs() << "XLV: found two base pointers when analyzing "
                             "memory accesses\n");
        return None;
      }

      BasePointer = Arg;
    }

    // We calculate accesses relative to the pointer
    return APInt(SE.getTypeSizeInBits(Ty), 0);
  }

  Optional<APInt> visitCouldNotCompute(const SCEVCouldNotCompute *) {
    return None;
  }

public:
  SCEVMemAccessAnalyzer(ScalarEvolution &SE, InputBuilder &InferredInputs,
                        const SCEVConstant *ItStart, const SCEVConstant *ItEnd)
      : SE(SE), Inferred(InferredInputs), ItStart(ItStart), ItEnd(ItEnd) {}

  void analyze(const SCEV *Expr) {
    ItCurrent = ItStart;
    AddRecKind = NoAddRec;
    BasePointer = nullptr;

    Optional<APInt> Res = visit(Expr);
    if (!Res)
      return;
    if (!BasePointer)
      return;

    unsigned BytesPerElement = SE.getDataLayout().getTypeAllocSize(
        BasePointer->getType()->getNonOpaquePointerElementType());
    Inferred.addMemAccess(BasePointer, Res->getSExtValue() / BytesPerElement);

    if (AddRecKind == NoAddRec)
      return;

    if (AddRecKind == MonotoneAddRec) {
      ItCurrent = ItEnd;
      Inferred.addMemAccess(BasePointer,
                            visit(Expr)->getSExtValue() / BytesPerElement);
      return;
    }

    assert(AddRecKind == ArbitraryAddRec);
    // In this case, it does not necessarily suffice to evaluate the SCEV for
    // the first and the last iteration.  However, we can evaluate it for every
    // iteration.
    while (ItCurrent != ItEnd) {
      ItCurrent = cast<SCEVConstant>(SE.getConstant(ItCurrent->getAPInt() + 1));
      Inferred.addMemAccess(BasePointer,
                            visit(Expr)->getSExtValue() / BytesPerElement);
    }
  }
};

} // end anonymous namespace

static void inferInputs(const Function &F, Loop &L, ScalarEvolution &SE,
                        ExplorativeLVPass::InputBuilder &InferredInputs) {
  // Obtain/choose a trip count
  const SCEV *MaxBackedgeTakenCount = SE.getSymbolicMaxBackedgeTakenCount(&L);
  LLVM_DEBUG(dbgs() << "XLV: backedge taken count is " << *MaxBackedgeTakenCount
                    << '\n');
  Optional<APInt> BTRes = SCEVBackedgeTakenAnalyzer::analyze(
      SE, MaxBackedgeTakenCount, InferredInputs, XLVDesiredTripCount - 1);
  if (!BTRes) {
    LLVM_DEBUG(dbgs() << "XLV: could not obtain/choose a trip count\n");
    return;
  }
  const SCEVConstant *ItStart =
      cast<SCEVConstant>(SE.getConstant(APInt(BTRes->getBitWidth(), 0)));
  const SCEVConstant *ItEnd = cast<SCEVConstant>(SE.getConstant(*BTRes));
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
          if (!InferredInputs.isInputArg(Arg))
            continue;
        }
      } else {
        continue;
      }

      const SCEV *LocSCEV = SE.getSCEV(const_cast<Value *>(Loc));
      LLVM_DEBUG(dbgs() << "XLV: memory access at " << *LocSCEV << '\n');
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
    assert(MainPass->EvaluationInfoMap);
    EvaluationInfo *Factors = MainPass->getEvalutaionInfo(F);

    // Find loop
    auto LoopIt = LI.begin();
    if (LoopIt + 1 != LI.end()) {
      LLVM_DEBUG(dbgs() << "XLV: expected exactly one loop in loop function "
                        << F.getName() << '\n');
      return PreservedAnalyses::all();
    }
    Loop *L = *LoopIt;

    // Add VF/IF/UF
    addStringMetadataToLoop(L, "llvm.loop.vectorize.width", Factors->VF);
    addStringMetadataToLoop(L, "llvm.loop.interleave.count", Factors->IF);
    if (Factors->UF)
      addStringMetadataToLoop(L, "llvm.loop.unroll.count", Factors->UF);

    // Infer inputs (if desired, i. e. MainPass->InferredInputs is present)
    if (MainPass->InferredInputs) {
      inferInputs(F, *L, SE, *MainPass->InferredInputs);

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

  if (!CSVOutput && !XLVCSVOut.empty()) {
    CSVOutput = new CSVOutputContainer();
    if (CSVOutput->EC) {
      errs() << "XLV: could not open " << XLVCSVOut << ": "
             << CSVOutput->EC.message() << '\n';
      delete CSVOutput;
      CSVOutput = nullptr;
    }
  }

  unsigned LoopNo = 0;
  do {
    Loop *L = Worklist.pop_back_val();
    LLVM_DEBUG(dbgs() << "XLV: --- processing loop " << LoopNo << " in "
                      << F.getName() << " ---\n");
    if (processLoop(F, *L, LoopNo, SE, TLI)) {
      LLVM_DEBUG(dbgs() << "XLV: processing loop failed, letting the "
                           "AutoVectorizer determine VF and IF\n");
    }
    ++LoopNo;
  } while (!Worklist.empty());

  EvaluationInfoMap = nullptr; // clean up

  return PreservedAnalyses::all();
}
