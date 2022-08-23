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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/ExplorativeLV.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
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
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Value.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
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
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <random>
#include <string>
#include <system_error>
#include <utility>

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
static cl::opt<unsigned>
    XLVMaxCodeSize("xlv-max-code-size", cl::Hidden, cl::init(0),
                   cl::desc("Limit the loop's code size (useful if an inner "
                            "loop was fully unrolled)"));

static cl::opt<bool> XLVAllocStatic(
    "xlv-alloc-static", cl::Hidden, cl::init(false),
    cl::desc("In benchmarking mode: statically allocate memory"));
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
    "xlv-benchmark-warmup", cl::Hidden, cl::init(2),
    cl::desc("In benchmarking mode: how many times to call each loop function "
             "before benchmarking"));
static cl::opt<unsigned>
    XLVBenchmarkLoopSize("xlv-benchmark-loop-size", cl::Hidden, cl::init(1),
                         cl::desc("In benchmarking mode: size (loop "
                                  "function calls) of the benchmark loop"));
static cl::opt<std::string> XLVBenchmarkPinCpu(
    "xlv-benchmark-pin-cpu", cl::Hidden, cl::init(""),
    cl::desc("In benchmarking mode: pin the program to a specific core (using "
             "taskset, only works on Linux)"));
static cl::opt<bool> XLVBenchmarkLikwid(
    "xlv-benchmark-likwid", cl::Hidden, cl::init(false),
    cl::desc("In benchmarking mode: generate markers for likwid-perfctr.  Only "
             "useful together with --xlv-artifacts-dir"));

static cl::opt<bool>
    XLVRandomizeOrder("xlv-randomize-order", cl::Hidden, cl::init(false),
                      cl::desc("Randomize the order of the loop functions"));

static cl::opt<std::string>
    XLVOnlyFunc("xlv-only-func", cl::Hidden, cl::init(""),
                cl::desc("Only consider the given function (after inlining)"));
static cl::opt<std::string>
    XLVOnlyLoop("xlv-only-loop", cl::Hidden, cl::init(""),
                cl::desc("Only consider the given loop (name)"));
static cl::opt<std::string> XLVForceFactors(
    "xlv-force-factors", cl::Hidden, cl::init(""),
    cl::desc("Enforce VF/IF/UF for the given functions and loop names.  "
             "Example: 'main,for.body.9.i,4,1,8;func,for.body.42,2,2,5'"));

static cl::opt<std::string>
    XLVArtifactsDir("xlv-artifacts-dir", cl::Hidden, cl::init(""),
                    cl::desc("Collect build artifacts in this directory"));
static cl::opt<bool>
    XLVDumpFuncIR("xlv-dump-func-ir", cl::Hidden, cl::init(false),
                  cl::desc("If true, print the loop function used for "
                           "explorative vectorization to stderr"));
static cl::opt<std::string>
    XLVCSVOut("xlv-csv-out", cl::Hidden, cl::init(""),
              cl::desc("Print explorative loop vectorization results as CSV to "
                       "the given file"));

namespace {

struct Artifact {
  SmallString<32> Path;

  ~Artifact() {
    if (XLVArtifactsDir.empty())
      sys::fs::remove(Path);
  }

  std::error_code create(const Twine &Prefix, StringRef Suffix, int &ResultFD,
                         sys::fs::OpenFlags Flags = sys::fs::OF_None) {
    if (XLVArtifactsDir.empty())
      return sys::fs::createTemporaryFile(Prefix, Suffix, ResultFD, Path,
                                          Flags);
    const char *Middle = Suffix.empty() ? "" : ".";
    (XLVArtifactsDir + sys::path::get_separator() + Prefix + Middle + Suffix)
        .toStringRef(Path);
    return sys::fs::openFile(Path, ResultFD, sys::fs::CD_CreateAlways,
                             sys::fs::FA_Read | sys::fs::FA_Write, Flags);
  }

  std::error_code create(const Twine &Prefix, StringRef Suffix,
                         sys::fs::OpenFlags Flags = sys::fs::OF_None) {
    int FD;
    auto EC = create(Prefix, Suffix, FD, Flags);
    if (EC)
      return EC;
    // FD is only needed to avoid race conditions or to ensure that the file is
    // truncated, respectively. Close it right away.
    sys::fs::file_t F = sys::fs::convertFDToNativeFile(FD);
    sys::fs::closeFile(F);
    return EC;
  }
};

} // end anonymous namespace

class ExplorativeLVPass::InputBuilder {
  class MemAccessRange {
    friend class InputBuilder;

    int64_t Start;
    int64_t End;

    Type *ElementType;

    // We'll use global variables to allocate memory if --xlv-alloc-static
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

  /// Adds a memory access with a known end pointer.
  ///
  /// \param ArgStart  the argument corresponding to the start pointer
  /// \param ArgEnd    the argument corresponding to the end pointer
  /// \param Diff      the difference between start and end pointer (elements,
  ///                  not bytes)
  void addMemAccessWithEndptr(const Argument *ArgStart, const Argument *ArgEnd,
                              uint64_t Diff);

  APInt tryAddIntInput(const Argument *Arg, APInt &&Desired) {
    ConstantInt *&Input = IntInputs[Arg->getArgNo()];
    if (Input)
      return Input->getValue();

    Input = ConstantInt::get(Arg->getType()->getContext(), Desired);
    return Desired;
  }

  void build(Module &M, ArrayRef<Type *> Params, SmallVectorImpl<Value *> &Args,
             BasicBlock *BB);

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
                         SmallVectorImpl<Value *> &Args, BasicBlock *BB) {
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
      uint64_t Offset = Access.second - Range.Start;
      uint64_t NumElements = Range.End + 1 - Range.Start;
      if (XLVAllocStatic) {
        ArrayType *Ty = ArrayType::get(Range.ElementType, NumElements);
        if (!Range.GV) {
          Range.GV =
              new GlobalVariable(M, Ty, false, GlobalValue::PrivateLinkage,
                                 ConstantAggregateZero::get(Ty));
        }

        GlobalVariable *GV = Range.GV;
        ArgVal = ConstantExpr::getInBoundsGetElementPtr(
            GV->getValueType(), GV,
            ArrayRef<Constant *>{I640, ConstantInt::get(I64Ty, Offset)});
      } else { // Dynamic allocation
        LLVMContext &C = M.getContext();
        Type *SizeTy = Type::getIntNTy(C, sizeof(size_t) * 8);
        FunctionType *CallocFuncTy =
            FunctionType::get(Type::getInt8PtrTy(C), {SizeTy, SizeTy}, false);
        Function *CallocFunc = M.getFunction("calloc");
        if (!CallocFunc) {
          CallocFunc = Function::Create(
              CallocFuncTy, GlobalValue::ExternalLinkage, 0, "calloc", &M);
        }
        Value *Mem = CallInst::Create(
            CallocFuncTy, CallocFunc,
            {ConstantInt::get(SizeTy, NumElements),
             ConstantInt::get(SizeTy, M.getDataLayout().getTypeAllocSize(
                                          Range.ElementType))},
            "arg" + Twine(I) + ".raw", BB);
        ArgVal = new BitCastInst(Mem, PointerType::get(Range.ElementType, 0),
                                 "arg" + Twine(I), BB);
        if (Offset != 0) {
          ArgVal = GetElementPtrInst::CreateInBounds(
              Range.ElementType, ArgVal, {ConstantInt::get(I64Ty, Offset)},
              "arg" + Twine(I) + ".offset", BB);
        }
        // Note that we do not generate a `free()`, we just leak the memory.
      }
    } else { // Fall back to null values
      ArgVal = Constant::getNullValue(Params[I]);
    }

    Args.push_back(ArgVal);
  }
}

namespace {

class LoopModuleBuilder : public ValueMaterializer {
  const Function &OrigFunc;
  const Loop &L;
  const unsigned LoopNo;
  ScalarEvolution &SE;

  Module *M = nullptr;
  ValueToValueMapTy VMap;
  ValueMapper Mapper;
  SmallVector<const Value *> Inputs;
  SmallVector<const Instruction *> Outputs;
  FunctionType *FuncTy;
  Function *RefLoopFunc = nullptr;

  InputBuilder InferredInputs;
  unsigned FullUnroll = 0;
  unsigned LoopSize = 0;

  bool MakeExecutable;

  bool determineIO();
  void buildFuncTy();

public:
  LoopModuleBuilder(const Function &OrigFunc, Loop &L, unsigned LoopNo,
                    ScalarEvolution &SE, bool MakeExecutable = false)
      : OrigFunc(OrigFunc), L(L), LoopNo(LoopNo), SE(SE),
        Mapper(VMap, RF_None, nullptr, this), MakeExecutable(MakeExecutable){};
  virtual ~LoopModuleBuilder() { delete M; }

  Value *materialize(Value *V) override;

  Module *getModule();

  void buildMainFuncBody(ArrayRef<ExplorativeLVPass::LoopFuncInfo> LoopFuncs);

  Function *buildLoopFunc(unsigned VF, unsigned IF, unsigned UF);
  void filterOutUnvectorized(bool AddMCAAnnotations);

  InputBuilder &getInferredInputsRef() {
    assert(M && "No loop func has been built yet");
    return InferredInputs;
  }
  unsigned getFullUnrollCount() { return FullUnroll; }
  unsigned getLoopSize() { return LoopSize; }
};

bool LoopModuleBuilder::determineIO() {
  // The core idea is to generate a function containing the basic blocks of the
  // loop.  Values used inside the loop and defined outside are parameters to
  // the function.  For each values defined inside the loop and used outside we
  // generate an output parameter to produce an observable side-effect such that
  // the code is not optimized away.

  SmallPtrSet<const Value *, 32> ArgSet;

  for (const BasicBlock *BB : L.getBlocks()) {
    LoopSize += BB->size();

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
        *M, GV->getValueType(), GV->isConstant(),
        MakeExecutable ? GV->getLinkage() : GlobalValue::ExternalLinkage,
        nullptr, GV->getName() + ".l", nullptr, GV->getThreadLocalMode(),
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
      Function *NF = Function::Create(
          cast<FunctionType>(F->getValueType()), GlobalValue::ExternalLinkage,
          F->getAddressSpace(), F->getName() + ".l", M);
      NF->copyAttributesFrom(F);
      return NF;
    }

    if (!F->isIntrinsic())
      // We only call intrinsics.
      return Constant::getNullValue(F->getType());

    Function *NF =
        Function::Create(cast<FunctionType>(F->getValueType()), F->getLinkage(),
                         F->getAddressSpace(), F->getName(), M);
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
      LLVM_DEBUG(
          dbgs() << "XLV: cannot infer trip count in original function\n");
      return nullptr;
    }
    if (const SCEVConstant *Const = dyn_cast<SCEVConstant>(BackedgeTaken))
      FullUnroll = Const->getAPInt().getLimitedValue(UINT_MAX - 1) + 1;
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
      OrigFunc.getName() + "." + Twine(LoopNo) + "." + Twine(VF) + "." +
          Twine(IF) + "." + Twine(UF),
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

void LoopModuleBuilder::buildMainFuncBody(
    ArrayRef<ExplorativeLVPass::LoopFuncInfo> LoopFuncs) {
  LLVMContext &C = M->getContext();

  // Some types and values we will use a few times
  Type *IntTy = Type::getIntNTy(C, sizeof(int) * 8);
  Type *I32Ty = Type::getInt32Ty(C);
  Type *I64Ty = Type::getInt64Ty(C);
  Type *ClockidTy = Type::getIntNTy(C, sizeof(clockid_t) * 8);
  static_assert(sizeof(timespec::tv_sec) <= 8,
                "timespec::tv_sec larger than expected");
  Type *TimespecSecTy = Type::getIntNTy(C, sizeof(timespec::tv_sec) * 8);
  static_assert(sizeof(timespec::tv_nsec) <= 8,
                "timespec::tv_nsec larger than expected");
  Type *TimespecNSecTy = Type::getIntNTy(C, sizeof(timespec::tv_nsec) * 8);
  Type *TimespecTy =
      StructType::create({TimespecSecTy, TimespecNSecTy}, "struct.timespec");
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
  // times.  The function furthermore takes and ID as well as the VF, IF and UF,
  // which are just printed along with the time.  The benchmarking function
  // itself is called by `int main(void)`.
  FunctionType *BenchFuncTy = FunctionType::get(
      Type::getVoidTy(C), {LoopFuncPtrTy, IntTy, IntTy, IntTy, IntTy, IntTy},
      false);
  Function *BenchFunc =
      Function::Create(BenchFuncTy, GlobalValue::ExternalLinkage, "bench", M);
  // To infer a suitable n, we build a calibration function that counts the
  // repitions needed until the elapsed time exceeds some threshold.
  FunctionType *CalibFuncTy = FunctionType::get(IntTy, {LoopFuncPtrTy}, false);
  Function *CalibFunc =
      Function::Create(CalibFuncTy, GlobalValue::ExternalLinkage, "calib", M);

  auto BuildBenchCalib = [=](bool IsBench) {
    LLVMContext &C = M->getContext();

    Function *Func;
    Argument *NArg, *IDArg, *VFArg, *IFArg, *UFArg;
    if (IsBench) {
      Func = BenchFunc;
      NArg = Func->getArg(1);
      NArg->setName("n");
      IDArg = Func->getArg(2);
      IDArg->setName("id");
      VFArg = Func->getArg(3);
      VFArg->setName("vf");
      IFArg = Func->getArg(4);
      IFArg->setName("if");
      UFArg = Func->getArg(5);
      UFArg->setName("uf");
    } else {
      Func = CalibFunc;
    }

    Argument *FuncArg = Func->getArg(0);
    FuncArg->setName("func");

    BasicBlock *StartBB = BasicBlock::Create(C, "start", Func);
    SmallVector<Value *> Args;
    /* Create arguments */ {
      unsigned NumParams = LoopFuncTy->getNumParams();
      Args.reserve(NumParams);

      InferredInputs.build(*M, LoopFuncTy->params(), Args, StartBB);
      if (IsBench) {
        for (Value *ArgVal : Args)
          LLVM_DEBUG(dbgs() << "XLV: arg " << *ArgVal << '\n');
      }

      // Allocate locations for output parameters
      assert(Inputs.size() <= LoopFuncTy->getNumParams());
      auto *It = LoopFuncTy->param_begin() + Inputs.size();
      for (auto *End = LoopFuncTy->param_end(); It != End; ++It) {
        Type *ElementType = (*It)->getNonOpaquePointerElementType();
        Args.push_back(new AllocaInst(ElementType, Func->getAddressSpace(),
                                      "outarg", StartBB));
      }
    }

    // Allocate timespec structs
    AllocaInst *TStartVar = new AllocaInst(TimespecTy, 0, "ts", StartBB);
    AllocaInst *TEndVar = new AllocaInst(TimespecTy, 0, "te", StartBB);

    // Call int likwid_markerRegisterRegion(const char *regionTag)
    Constant *LRegionTagPtr;
    FunctionType *LMarkerFuncTy;
    if (IsBench && XLVBenchmarkLikwid) {
      // Create the regionTag first:
      // snprintf(regionTag, 64, "%u.%u.%u", VF, IF, UF);
      const uint64_t NumElements = 64;
      ArrayType *LRegionTagTy = ArrayType::get(Type::getInt8Ty(C), NumElements);
      GlobalVariable *LRegionTagGV = new GlobalVariable(
          *M, LRegionTagTy, false, GlobalValue::PrivateLinkage,
          Constant::getNullValue(LRegionTagTy), "regionTag");
      LRegionTagPtr = ConstantExpr::getInBoundsGetElementPtr(
          LRegionTagGV->getValueType(), LRegionTagGV,
          ArrayRef<Constant *>{I640, I640});

      Constant *FormatStr = ConstantDataArray::getString(C, "%u.%u.%u");
      GlobalVariable *FormatStrGV = new GlobalVariable(
          *M, FormatStr->getType(), true, GlobalValue::PrivateLinkage,
          FormatStr, "regionTag_fstr");
      Constant *FormatStrPtr = ConstantExpr::getInBoundsGetElementPtr(
          FormatStrGV->getValueType(), FormatStrGV,
          ArrayRef<Constant *>{I640, I640});

      Type *SizeTy = Type::getIntNTy(C, sizeof(size_t) * 8);
      FunctionType *SnprintfTy = FunctionType::get(
          IntTy, {LRegionTagPtr->getType(), SizeTy, FormatStrPtr->getType()},
          true);
      CallInst::Create(SnprintfTy,
                       Function::Create(SnprintfTy,
                                        GlobalValue::ExternalLinkage,
                                        "snprintf", M),
                       {LRegionTagPtr, ConstantInt::get(SizeTy, NumElements),
                        FormatStrPtr, VFArg, IFArg, UFArg},
                       "", StartBB);

      // Register the region
      LMarkerFuncTy = FunctionType::get(IntTy, {Type::getInt8PtrTy(C)}, false);
      CallInst::Create(LMarkerFuncTy,
                       Function::Create(LMarkerFuncTy,
                                        GlobalValue::ExternalLinkage,
                                        "likwid_markerRegisterRegion", M),
                       {LRegionTagPtr}, "", StartBB);
    }

    // Warmup
    for (unsigned I = 0; I < XLVBenchmarkWarmup; ++I) {
      CallInst::Create(LoopFuncTy, FuncArg, Args, "", StartBB);
    }

    // Start measuring
    CallInst::Create(ClockGettimeFuncTy, ClockGettimeFunc, {Clockid, TStartVar},
                     "", StartBB);
    if (IsBench && XLVBenchmarkLikwid) {
      CallInst::Create(LMarkerFuncTy,
                       Function::Create(LMarkerFuncTy,
                                        GlobalValue::ExternalLinkage,
                                        "likwid_markerStartRegion", M),
                       {LRegionTagPtr}, "", StartBB);
    }

    auto TimeDiff = [=](BasicBlock *BB) {
      CallInst::Create(ClockGettimeFuncTy, ClockGettimeFunc, {Clockid, TEndVar},
                       "", BB);

      GetElementPtrInst *Ptr = GetElementPtrInst::CreateInBounds(
          TimespecTy, TEndVar, {I640, TimespecSecIndex}, "te.sec_ptr", BB);
      Value *EndSec = new LoadInst(TimespecSecTy, Ptr, "te.sec", BB);
      Ptr = GetElementPtrInst::CreateInBounds(
          TimespecTy, TStartVar, {I640, TimespecSecIndex}, "ts.sec_ptr", BB);
      Value *StartSec = new LoadInst(TimespecSecTy, Ptr, "ts.sec", BB);
      Value *DiffSec =
          BinaryOperator::CreateSub(EndSec, StartSec, "td.sec", BB);
      if (sizeof(time_t) < 8)
        DiffSec = new ZExtInst(DiffSec, I64Ty, "td.sec64", BB);
      Value *DiffNSec = BinaryOperator::CreateMul(DiffSec, Nanos, "", BB);

      Ptr = GetElementPtrInst::CreateInBounds(
          TimespecTy, TEndVar, {I640, TimespecNSecIndex}, "te.nsec_ptr", BB);
      Value *EndNSec = new LoadInst(TimespecNSecTy, Ptr, "te.nsec", BB);
      if (sizeof(long) < 8)
        EndNSec = new ZExtInst(EndNSec, I64Ty, "te.nsec64", BB);
      DiffNSec = BinaryOperator::CreateAdd(DiffNSec, EndNSec, "", BB);

      Ptr = GetElementPtrInst::CreateInBounds(
          TimespecTy, TStartVar, {I640, TimespecNSecIndex}, "ts.nsec_ptr", BB);
      Value *StartNSec = new LoadInst(TimespecNSecTy, Ptr, "ts.nsec", BB);
      if (sizeof(long) < 8)
        StartNSec = new ZExtInst(EndNSec, I64Ty, "te.nsec64", BB);
      return BinaryOperator::CreateSub(DiffNSec, StartNSec, "td", BB);
    };

    // Benchmarking loop
    BasicBlock *LoopBB = BasicBlock::Create(C, "loop", Func);
    BranchInst::Create(LoopBB, StartBB);
    PHINode *CountPhi = PHINode::Create(IntTy, 2, "counter", LoopBB);
    CountPhi->addIncoming(Int1, StartBB);

    for (unsigned I = 0; I < XLVBenchmarkLoopSize; ++I) {
      CallInst::Create(LoopFuncTy, FuncArg, Args, "", LoopBB);
    }

    // Counters and exit conditions
    ICmpInst *ContLoop;
    Value *TDiff;
    if (IsBench) {
      ContLoop = new ICmpInst(*LoopBB, CmpInst::ICMP_EQ, CountPhi, NArg);
    } else {
      TDiff = TimeDiff(LoopBB);
      ContLoop =
          new ICmpInst(*LoopBB, CmpInst::ICMP_UGT, TDiff,
                       ConstantInt::get(I64Ty, 1000 * XLVBenchmarkUSecs));
    }
    BinaryOperator *CountAdd =
        BinaryOperator::CreateAdd(CountPhi, Int1, "counter_next", LoopBB);

    // Branch at the end of the loop
    BasicBlock *EvalBB = BasicBlock::Create(C, "eval", Func);
    BranchInst::Create(EvalBB, LoopBB, ContLoop, LoopBB);
    CountPhi->addIncoming(CountAdd, LoopBB);

    // Stop measuring
    if (IsBench) {
      if (XLVBenchmarkLikwid) {
        CallInst::Create(LMarkerFuncTy,
                         Function::Create(LMarkerFuncTy,
                                          GlobalValue::ExternalLinkage,
                                          "likwid_markerStopRegion", M),
                         {LRegionTagPtr}, "", EvalBB);
      }
      TDiff = TimeDiff(EvalBB);

      // `printf(fstr, id, dt, vf, if, uf); return;`
      Constant *FormatStr =
          ConstantDataArray::getString(C, "%u %llu %u.%u.%u\n");
      GlobalVariable *FormatStrGV =
          new GlobalVariable(*M, FormatStr->getType(), true,
                             GlobalValue::PrivateLinkage, FormatStr, "fstr");
      Constant *FormatStrPtr = ConstantExpr::getInBoundsGetElementPtr(
          FormatStrGV->getValueType(), FormatStrGV,
          ArrayRef<Constant *>{I640, I640});

      CallInst::Create(PrintfFuncTy, PrintfFunc,
                       {FormatStrPtr, IDArg, TDiff, VFArg, IFArg, UFArg}, "",
                       EvalBB);
      ReturnInst::Create(C, EvalBB);
    } else {
      // `printf(fstr, dt, n); return n;`
      Constant *FormatStr = ConstantDataArray::getString(C, "%llu %u\n");
      GlobalVariable *FormatStrGV =
          new GlobalVariable(*M, FormatStr->getType(), true,
                             GlobalValue::PrivateLinkage, FormatStr, "cfstr");
      Constant *FormatStrPtr = ConstantExpr::getInBoundsGetElementPtr(
          FormatStrGV->getValueType(), FormatStrGV,
          ArrayRef<Constant *>{I640, I640});

      CallInst::Create(PrintfFuncTy, PrintfFunc,
                       {FormatStrPtr, TDiff, CountPhi}, "", EvalBB);
      ReturnInst::Create(C, CountPhi, EvalBB);
    }
  };
  BuildBenchCalib(false);
  BuildBenchCalib(true);

  // Declare & define `int main(void)`
  Function *MainFunc =
      Function::Create(FunctionType::get(IntTy, false),
                       GlobalValue::ExternalLinkage, 0, "main", M);

  BasicBlock *MainBB = BasicBlock::Create(C, "main", MainFunc);

  FunctionType *VoidFuncTy;
  if (XLVBenchmarkLikwid) {
    VoidFuncTy = FunctionType::get(Type::getVoidTy(C), false);
    CallInst::Create(VoidFuncTy,
                     Function::Create(VoidFuncTy, GlobalValue::ExternalLinkage,
                                      0, "likwid_markerInit", M),
                     "", MainBB);
    CallInst::Create(VoidFuncTy,
                     Function::Create(VoidFuncTy, GlobalValue::ExternalLinkage,
                                      0, "likwid_markerThreadInit", M),
                     "", MainBB);
  }

  CallInst *NRuns =
      CallInst::Create(CalibFuncTy, CalibFunc, {RefLoopFunc}, "n", MainBB);
  unsigned LoopFuncID = 0;
  for (const ExplorativeLVPass::LoopFuncInfo &Info : LoopFuncs) {
    if (Info.Func) {
      Constant *ID = ConstantInt::get(IntTy, LoopFuncID);
      Constant *VF = ConstantInt::get(IntTy, Info.VF);
      Constant *IF = ConstantInt::get(IntTy, Info.IF);
      Constant *UF = ConstantInt::get(IntTy, Info.UF);
      CallInst::Create(BenchFuncTy, BenchFunc,
                       {Info.Func, NRuns, ID, VF, IF, UF}, "", MainBB);
    }
    ++LoopFuncID;
  }

  if (XLVBenchmarkLikwid) {
    CallInst::Create(VoidFuncTy,
                     Function::Create(VoidFuncTy, GlobalValue::ExternalLinkage,
                                      0, "likwid_markerClose", M),
                     "", MainBB);
  }

  ReturnInst::Create(C, Int0, MainBB);
}

} // end anonymous namespace

static void addMCAAnnotation(LLVMContext &C, BasicBlock &First,
                             BasicBlock &Last, const Twine &ID) {
  const char SideEffects[] = "~{memory},~{dirflag},~{fpsr},~{flags}";
  FunctionType *Ty = FunctionType::get(Type::getVoidTy(C), false);

  InlineAsm *StartAsm =
      InlineAsm::get(Ty, ("# LLVM-MCA-BEGIN " + ID).str(), SideEffects, true);
  InlineAsm *EndAsm =
      InlineAsm::get(Ty, ("# LLVM-MCA-END " + ID).str(), SideEffects, true);

  CallInst::Create(Ty, StartAsm, "", First.getFirstNonPHI());
  CallInst::Create(Ty, EndAsm, "", Last.getTerminator());
}

static BasicBlock *findSomeLatch(Function &F) {
  for (BasicBlock &BB : F) {
    if (BB.getTerminator()->getMetadata(LLVMContext::MD_loop))
      return &BB;
  }
  return nullptr;
}

static BasicBlock *findLatchOfVectorizedLoop(Function &F) {
  for (BasicBlock &BB : F) {
    const MDNode *MD = findOptionMDForLoopID(
        BB.getTerminator()->getMetadata(LLVMContext::MD_loop),
        "llvm.loop.isvectorized");
    if (!MD)
      continue;

    // Inspired by getOptionalBoolLoopAttribute from LoopInfo
    switch (MD->getNumOperands()) {
    case 1:
      return &BB;
    case 2:
      ConstantInt *IntMD =
          mdconst::extract_or_null<ConstantInt>(MD->getOperand(1).get());
      if (!IntMD || IntMD->getZExtValue())
        return &BB;
      continue;
    }
    llvm_unreachable("unexpected number of operands");
  }
  return nullptr;
}

static BasicBlock &getMCAStart(Function &F, BasicBlock &Latch) {
  // Sadly, there is no simple way access the LoopInfo here ...
  //
  // A latch must have a backedge to the header.  Looking at the basic block
  // order in the function, we assume the header to come first (otherwise the
  // MCA analysis won't work).
  BasicBlock *First = &Latch;
  Instruction *Term = Latch.getTerminator();
  for (unsigned Idx = 0, Num = Term->getNumSuccessors(); Idx < Num; ++Idx) {
    BasicBlock *SuccBB = Term->getSuccessor(Idx);
    if (SuccBB == &Latch)
      continue;

    // Since we cannot directly tell if `SuccBB` comes before `First`, we use
    // linear search.  As `Term.getNumSuccessors()` is usually 2 and one of
    // the successors is the latch itself, this should be quite fast in
    // practice.
    for (BasicBlock &BB : F.getBasicBlockList()) {
      if (&BB == First)
        break;
      if (&BB == SuccBB) {
        First = SuccBB;
        break;
      }
    }
  }
  return *First;
}

static bool containsVectorInstruction(Function::BasicBlockListType &BBs) {
  for (BasicBlock &BB : BBs) {
    for (Instruction &Inst : BB) {
      if (Inst.getType()->isVectorTy())
        return true;
    }
  }
  return false;
}

static void filterOutUnvectorized(
    SmallVectorImpl<ExplorativeLVPass::LoopFuncInfo> &LoopFuncs,
    bool AddMCAAnnotations) {
  for (unsigned I = 0, E = LoopFuncs.size(); I < E; ++I) {
    ExplorativeLVPass::LoopFuncInfo &Info = LoopFuncs[I];
    if (!Info.Func)
      continue;
    Function &F = *Info.Func;
    Function::BasicBlockListType &BBs = F.getBasicBlockList();

    BasicBlock *Latch = findSomeLatch(F);
    if (Latch) {
      if (Info.VF > 1 || Info.IF > 1)
        Latch = findLatchOfVectorizedLoop(F);
      if (Latch && (Info.VF == 1 || containsVectorInstruction(BBs))) {
        if (AddMCAAnnotations)
          addMCAAnnotation(F.getContext(), getMCAStart(F, *Latch), *Latch,
                           Twine(I));
        continue;
      }
    } else {
      // In this case the loop has (probably) been fully unrolled.  To find out
      // whether vectorization actually took place, we can only look at the
      // instruction types.
      // Here, we can only filter out loops with VF > 1.  If the SLP vectorizer
      // is enabled and IF > 1, the duplicated instructions from the
      // interleaving are typically merged into vector instructions, but we do
      // not want to rely on this.  This means that we do not filter out any
      // loops with VF = 1 and IF > 1, which possibly increases compile time a
      // little.  For the evaluation it is not a problem since we know how much
      // "work" was done.
      Info.FullUnroll = true;

      if (Info.VF == 1 || containsVectorInstruction(BBs)) {
        if (AddMCAAnnotations)
          addMCAAnnotation(F.getContext(), BBs.front(), BBs.back(), Twine(I));
        continue;
      }
    }

    LLVM_DEBUG(dbgs() << "XLV: Unvectorized with VF " << Info.VF << ", IF "
                      << Info.IF << ", UF " << Info.UF << '\n');
    F.eraseFromParent();
    Info.Func = nullptr;
  }
}

// The more complex approach, using the runtime estimation mechanisms
// already present in LLVM: llvm-mca
//
// Returns whether errors occured
template <typename FuncT>
static bool performMCACostCalc(std::string MCAPath, StringRef FileName,
                               const Twine &OutputStem, StringRef TargetCPU,
                               StringRef TargetTriple, FuncT Callback) {
  // Build the most specific command line we can generate from the info in TM
  Artifact MCAOutFile;
  MCAOutFile.create(OutputStem, "json");
  if (sys::ExecuteAndWait(
          MCAPath,
          {"llvm-mca", "--mcpu=" + TargetCPU.str(),
           "--mtriple=" + TargetTriple.str(), "--instruction-info=false",
           "--resource-pressure=false", "--json", FileName},
          None, {StringRef(""), MCAOutFile.Path.str(), StringRef("")}) != 0) {
    errs() << "XLV: executing llvm-mca failed\n";
    return true;
  }

  // Read JSON
  auto MCAOutBuf = MemoryBuffer::getFile(MCAOutFile.Path);
  if (!MCAOutBuf) {
    errs() << "XLV: could not read llvm-mca output\n";
    return true;
  }
  StringRef MCAOut = MCAOutBuf.get()->getBuffer();
  auto MCAJSON = json::parse(MCAOut);
  if (!MCAJSON) {
    errs() << "XLV: could not parse llvm-mca JSON\n";
    return true;
  }

  json::Object *Root = MCAJSON.get().getAsObject();
  assert(Root && "Root must be an object");
  json::Array *CodeRegions = Root->getArray("CodeRegions");
  assert(CodeRegions && "Root must contain an array 'CodeRegions'");
  for (json::Value &CodeRegionVal : *CodeRegions) {
    json::Object *CodeRegion = CodeRegionVal.getAsObject();
    assert(CodeRegion && "CodeRegion must be an object");
    StringRef FuncID = *CodeRegion->getString("Name");
    unsigned ID;
    if (FuncID.getAsInteger(10, ID))
      continue; // error

    json::Object *SummaryView = CodeRegion->getObject("SummaryView");
    assert(SummaryView && "CodeRegion must contain object 'SummaryView'");
    Optional<int64_t> TotalCycles = SummaryView->getInteger("TotalCycles");
    if (TotalCycles)
      Callback(ID, (double)TotalCycles.getValue());
  }

  return false;
}

// Add to the given pass manager everything we need to simulate our backend
// pipeline
// ~~> EmitAssemblyHelper::AddEmitPasses in clang's BackendUtil
static void initCodeGen(legacy::PassManager &PM, TargetMachine &TM,
                        ExplorativeLVPass *XLV, const TargetLibraryInfo &TLI,
                        raw_pwrite_stream &OS, CodeGenFileType CGFT) {
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
  CSVOutputContainer() : OS(XLVCSVOut, EC) {}
};

static void parseXLVForceFactors(
    StringMap<std::tuple<unsigned, unsigned, unsigned>> &ForcedFactors) {
  StringRef Remainder = StringRef(XLVForceFactors), LoopEntry, FacStr;
  while (!Remainder.empty()) {
    std::tie(LoopEntry, Remainder) = Remainder.split(';');

    unsigned Factors[3];
    for (int I = 2; I >= 0; --I) {
      std::tie(LoopEntry, FacStr) = LoopEntry.rsplit(',');
      if (FacStr.getAsInteger(10, Factors[I])) {
        errs() << "XLV: error parsing --xlv-force-factors\n";
        return;
      }
    }
    ForcedFactors.try_emplace(LoopEntry, Factors[0], Factors[1], Factors[2]);
  }
}

ExplorativeLVPass::ExplorativeLVPass(TargetMachine *TM,
                                     PipelineTuningOptions PTO,
                                     Optional<PGOOptions> PGOOpt)
    : TM(TM), PTO(PTO), PGOOpt(PGOOpt) {
  parseXLVForceFactors(ForcedFactors);
  if (!XLVArtifactsDir.empty()) {
    if (sys::fs::create_directories(XLVArtifactsDir))
      errs() << "XLV: creating artifacts dir failed\n";
  }
}

ExplorativeLVPass::~ExplorativeLVPass() {
  delete NullCGPipeline;
  delete OptPipeline;
  delete CSVOutput;
}

// Inspired by addStringMetadataToLoop from LoopUtils
static void setFactorMetadata(Loop &L, unsigned VF, unsigned IF, unsigned UF) {
  LLVMContext &Context = L.getHeader()->getContext();
  Type *I32Ty = Type::getInt32Ty(Context);

  // The first operand is reserved for the self-reference.
  SmallVector<Metadata *, 6> MDs(1);

  if (const MDNode *LoopID = L.getLoopID()) {
    for (const MDOperand *It = LoopID->op_begin() + 1, *End = LoopID->op_end();
         It != End; ++It) {
      MDNode *Node = cast<MDNode>(*It);
      // If it is of form key = value, try to parse it.
      if (Node->getNumOperands() == 2) {
        MDString *S = dyn_cast<MDString>(Node->getOperand(0));
        // Do not copy vectorize/interleave/unroll metadata
        if (S && (S->getString().startswith("llvm.loop.vectorize.") ||
                  S->getString().startswith("llvm.loop.interleave.") ||
                  S->getString().startswith("llvm.loop.unroll.")))
          continue;
      }

      MDs.push_back(Node);
    }
  }

  if (UF) {
    if (UF == UINT_MAX) {
      MDs.push_back(MDNode::get(
          Context, {MDString::get(Context, "llvm.loop.unroll.full")}));
    } else {
      MDs.push_back(MDNode::get(
          Context, {MDString::get(Context, "llvm.loop.unroll.count"),
                    ConstantAsMetadata::get(ConstantInt::get(I32Ty, UF))}));
    }
    if (VF > 1 || IF > 1) {
      // If we have a vector loop, we want the unroll count to be set only on
      // the vector loop.  To achieve this, we can use
      // "llvm.loop.vectorize.followup_vectorized".  When using this, however,
      // no other metadata is copied to the vector loop and
      // "llvm.loop.isvectorized" is not set.  So basically, we want to use MDs
      // as operands after "llvm.loop.vectorize.followup_vectorized".  To
      // achieve this, we can modify MDs a little:
      // Overwrite the first (reserved but by now uninitialized) element.
      MDs[0] =
          MDString::get(Context, "llvm.loop.vectorize.followup_vectorized");
      // Append the "llvm.loop.isvectorized" node.
      MDs.push_back(MDNode::get(
          Context, {MDString::get(Context, "llvm.loop.isvectorized"),
                    ConstantAsMetadata::get(ConstantInt::get(I32Ty, 1))}));
      // Replace the "llvm.loop.unroll.count" node by the newly created
      // "llvm.loop.vectorize.followup_vectorized" node.
      MDs[MDs.size() - 2] = MDNode::get(Context, MDs);
      // Finally, remove "llvm.loop.isvectorized".
      MDs.pop_back();
    }
  }

  MDs.push_back(MDNode::get(
      Context, {MDString::get(Context, "llvm.loop.vectorize.width"),
                ConstantAsMetadata::get(ConstantInt::get(I32Ty, VF))}));
  MDs.push_back(MDNode::get(
      Context, {MDString::get(Context, "llvm.loop.interleave.count"),
                ConstantAsMetadata::get(ConstantInt::get(I32Ty, IF))}));

  MDNode *NewLoopID = MDNode::get(Context, MDs);
  // Set operand 0 to refer to the loop id itself.
  NewLoopID->replaceOperandWith(0, NewLoopID);
  L.setLoopID(NewLoopID);
}

// returns whether an error occured
bool ExplorativeLVPass::processLoop(Function &F, Loop &L, unsigned LoopNo,
                                    ScalarEvolution &SE,
                                    TargetLibraryInfo &TLI) {
  assert(OptPipeline && "OptPipeline not initialized");

  LoopModuleBuilder Builder(F, L, LoopNo, SE, XLVMetric == Metric::Benchmark);
  Module *M = Builder.getModule();
  if (!M)
    return true;

  assert(LoopFuncInfoMap.empty());

  for (unsigned I = 0, E = LoopFuncInfos.size(); I < E; ++I) {
    LoopFuncInfo &Info = LoopFuncInfos[I];
    Info.reset();
    unsigned FullUnroll = Builder.getFullUnrollCount();
    unsigned LoopSize = Builder.getLoopSize() * Info.IF;
    if (Info.UF < UINT_MAX) {
      if (Info.UF > 0)
        LoopSize *= Info.UF;
    } else {
      if (FullUnroll > XLVMaxFullUnroll || FullUnroll <= XLVMaxUF)
        continue;
      LoopSize *= FullUnroll;
    }
    if (XLVMaxCodeSize > 0 && LoopSize > XLVMaxCodeSize) {
      LLVM_DEBUG(dbgs() << "XLV: not exploring VF " << Info.VF << ", IF "
                        << Info.IF << ", UF " << Info.UF
                        << " because of code size " << LoopSize << " > "
                        << XLVMaxCodeSize << '\n');
      continue;
    }
    Info.Func = Builder.buildLoopFunc(Info.VF, Info.IF, Info.UF);
    // Info.FullUnroll is set in filterOutUnvectorized()
    LoopFuncInfoMap.try_emplace(Info.Func, I);
  }

  if (LoopFuncInfoMap.empty())
    return true; // There are no functions to explore

  // Get a reference to the InputBuilder.  The ExplorativeLVPass in the
  // optimization pipeline will (try to) infer the arguments.
  InferredInputs = &Builder.getInferredInputsRef();

  // Run optimization pipeline
  OptPipeline->run(*M);

  // The arguments should be inferred exactly once.  To achieve this,
  // InferredInputs is cleared.
  assert(!InferredInputs && "Arguments have not been inferred");

  // Filter out functions that have not been vectorized and add annotations when
  // in MCA mode
  filterOutUnvectorized(LoopFuncInfos, XLVMetric == Metric::MCA);

  if (XLVMetric == Metric::Benchmark)
    // Build the main function -- no need to optimize it
    Builder.buildMainFuncBody(LoopFuncInfos);

  if (!XLVArtifactsDir.empty()) {
    // Print the optimized IR
    std::error_code EC;
    raw_fd_ostream OS((XLVArtifactsDir + sys::path::get_separator() +
                       "xlv-opt-" + F.getName() + "-" + Twine(LoopNo) + ".ll")
                          .str(),
                      EC);
    if (EC) {
      errs() << "XLV: could not open module IR file\n";
    } else {
      OS << *M;
    }
  }

  bool Errors = true;

  // Number of runs-only relevant in benchmark metric and for CSV/debug output
  uint64_t NRuns = 1;

  if (XLVMetric == Metric::InstCount) {
    NullCGPipeline->PM.run(*M);

    for (LoopFuncInfo &Info : LoopFuncInfos) {
      if (Info.Func) {
        assert(Info.UF == 1 && "Unrolling should be disabled");
        if (!std::isinf(Info.Costs)) {
          if (Info.FullUnroll) {
            unsigned FullUnroll = Builder.getFullUnrollCount();
            assert(FullUnroll > 0);
            Info.Costs /= FullUnroll;
          } else {
            Info.Costs /= Info.VF * Info.IF;
          }
        }
      }
    }
  } else if (XLVMetric == Metric::MCA) {
    int ASMFD;
    Artifact ASM;
    if (ASM.create("xlv-" + F.getName() + "-" + Twine(LoopNo), "s", ASMFD)) {
      dbgs() << "XLV: Failed to create assembly file\n";
      goto output_csv;
    }
    {
      raw_fd_ostream OS(ASMFD, true);
      legacy::PassManager CGPipeline;
      initCodeGen(CGPipeline, *TM, this, TLI, OS, CGFT_AssemblyFile);
      CGPipeline.run(*M);
    }

    unsigned FullUnroll = Builder.getFullUnrollCount();
    auto Callback = [this, FullUnroll](unsigned ID, double Costs) {
      if (!std::isinf(Costs)) {
        LoopFuncInfo &Info = LoopFuncInfos[ID];
        if (Info.FullUnroll) {
          assert(FullUnroll > 0);
          Costs /= FullUnroll;
        } else if (Info.UF >= 1) {
          Costs /= Info.VF * Info.IF * Info.UF;
        } else {
          LLVM_DEBUG(dbgs() << "XLV: Warning: The unroll count was not "
                               "considered when computing the costs\n");
          Costs /= Info.VF * Info.IF;
        }
        Info.Costs = Costs;
      }
    };

    // Perform the cost calculation after the pipeline is deleted to make sure
    // the assembly printer has finished.
    if (performMCACostCalc(Paths.mca(), ASM.Path,
                           "xlv-mca-" + F.getName() + "-" + Twine(LoopNo),
                           TM->getTargetCPU(), M->getTargetTriple(), Callback))
      goto output_csv; // error
  } else {
    assert(XLVMetric == Metric::Benchmark);

    // Run code generation pipeline
    Artifact Object;
    int ObjectFD;
    if (Object.create("xlv-" + F.getName() + "-" + Twine(LoopNo), "o",
                      ObjectFD)) {
      dbgs() << "XLV: Failed to create object file\n";
      goto output_csv;
    }
    {
      raw_fd_ostream OS(ObjectFD, true);
      legacy::PassManager CGPipeline;
      initCodeGen(CGPipeline, *TM, this, TLI, OS, CGFT_ObjectFile);
      CGPipeline.run(*M);
    }

    // Create an executable
    Artifact Exec;
    if (Exec.create("xlv-" + F.getName() + "-" + Twine(LoopNo), "")) {
      dbgs() << "XLV: Failed to create executable file\n";
      goto output_csv;
    }

    SmallVector<StringRef, 5> CCOpts;
    CCOpts.push_back("cc");
    CCOpts.push_back("-o");
    CCOpts.push_back(Exec.Path);
    CCOpts.push_back(Object.Path);
    if (!TM->isPositionIndependent())
      // Some systems appear to generate position independent executables by
      // default, but we do not necessarily generate position independent code
      // as well.
      CCOpts.push_back("-no-pie");
    if (XLVBenchmarkLikwid)
      CCOpts.push_back("-llikwid");

    if (sys::ExecuteAndWait(Paths.cc(), CCOpts) != 0) {
      errs() << "XLV: linking failed\n";
      goto output_csv;
    }

    // Count the functions that are benchmarked.  We use this when reading the
    // benchmarking output and for the timeout seconds.
    unsigned FuncCount = 0;
    for (LoopFuncInfo &Info : LoopFuncInfos) {
      if (Info.Func)
        ++FuncCount;
    }

    // Do the benchmarking
    // Timout: ceil(2 * (calib_secs + n * bench_secs)))
    unsigned TimeoutS = 1 + 2 * (FuncCount + 1) * XLVBenchmarkUSecs / 1000000;
    LLVM_DEBUG(dbgs() << "XLV: benchmarking for up to " << TimeoutS
                      << " s ...\n");
    Artifact BenchResFile;
    if (BenchResFile.create("xlv-out-" + F.getName() + "-" + Twine(LoopNo),
                            "txt")) {
      errs() << "XLV: failed to open benchmark result file\n";
      goto output_csv;
    }
    int RetCode;
    if (XLVBenchmarkPinCpu.empty()) {
      RetCode = sys::ExecuteAndWait(
          Exec.Path, {"explorative-lv"}, None,
          {StringRef(""), BenchResFile.Path.str(), StringRef("")}, TimeoutS);
    } else {
      RetCode = sys::ExecuteAndWait(
          Paths.taskset(), {"taskset", "-c", XLVBenchmarkPinCpu, Exec.Path},
          None, {StringRef(""), BenchResFile.Path.str(), StringRef("")},
          TimeoutS);
    }
    if (RetCode == -2) {
      errs() << "XLV: benchmarking failed/timed out\n";
      goto output_csv;
    }
    if (RetCode == -1) {
      errs() << "XLV: failed to execute\n";
      goto output_csv;
    }

    auto BenchResBuf = MemoryBuffer::getFile(BenchResFile.Path);
    if (!BenchResBuf) {
      errs() << "XLV: could not load benchmarking results\n";
      goto output_csv;
    }
    StringRef BenchRes = BenchResBuf.get()->getBuffer();

    // Read calibration info
    uint64_t CalibTime;
    if (BenchRes.consumeInteger(10, CalibTime) ||
        !BenchRes.consume_front(" ") || BenchRes.consumeInteger(10, NRuns)) {
      errs() << "XLV: malformed benchmarking results\n";
      goto output_csv;
    }
    LLVM_DEBUG(dbgs() << "XLV: calibration run took " << CalibTime
                      << " ns, executed each loop function "
                      << XLVBenchmarkLoopSize * NRuns << " times\n");

    for (unsigned I = 0; I < FuncCount; ++I) {
      BenchRes = BenchRes.split('\n').second;

      unsigned ID;
      uint64_t Costs;
      if (BenchRes.consumeInteger(10, ID) || !BenchRes.consume_front(" ") ||
          BenchRes.consumeInteger(10, Costs) || ID >= LoopFuncInfos.size()) {
        errs() << "XLV: malformed benchmarking results\n";
        goto output_csv;
      }

      LoopFuncInfos[ID].Costs = Costs;
    }
  }

  Errors = false;

  /* Get & set the best factors */ {
    double MinCosts = INFINITY;
    unsigned BestVF = 1, BestIF = 1, BestUF = 0;

    for (LoopFuncInfo &Info : LoopFuncInfos) {
      if (!Info.Func)
        continue; // The LoopFunction has been deleted

      if (std::isinf(Info.Costs)) {
        LLVM_DEBUG(dbgs() << "XLV: costs for VF " << Info.VF << ", IF "
                          << Info.IF << ", UF " << Info.UF << ": (invalid)\n");
        continue;
      }

      LLVM_DEBUG(dbgs() << "XLV: costs for VF " << Info.VF << ", IF " << Info.IF
                        << ", UF " << Info.UF << ": " << Info.Costs << '\n');

      if (Info.Costs > MinCosts)
        continue;
      MinCosts = Info.Costs;
      BestVF = Info.VF;
      BestIF = Info.IF;
      BestUF = Info.UF;
    }

    LLVM_DEBUG(dbgs() << "XLV: choosing VF " << BestVF << ", IF " << BestIF
                      << ", UF " << BestUF << '\n');
    setFactorMetadata(L, BestVF, BestIF, BestUF);
  }

output_csv:
  if (CSVOutput) {
    StringRef Func = F.getName();
    StringRef LoopName = L.getName();
    for (LoopFuncInfo &Info : LoopFuncInfos) {
      if (!Info.Func)
        continue; // The LoopFunction has been deleted

      // We use the following columns:
      // - function name
      // - loop number
      // - name of loop header basic block
      // - VF
      // - IF
      // - UF
      // - costs (empty if invalid)
      // - number of runs
      CSVOutput->OS << Func << ',' << LoopNo << ',' << LoopName << ','
                    << Info.VF << ',' << Info.IF << ',' << Info.UF << ',';
      if (!std::isinf(Info.Costs))
        CSVOutput->OS << Info.Costs;
      CSVOutput->OS << ',' << NRuns << '\n';
    }
  }

  return Errors;
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
  /// Global variables typically do not occur, because they require a `load`
  /// instruction, which is typically loop-invariant.  Also, scalar evolution
  /// analysis appears to fail on them.
  const Argument *InferredPtrStart = nullptr;

  /// End pointer value
  const Argument *InferredPtrEnd = nullptr;

  /// Difference (bytes) between Start and End pointer (if both are present)
  uint64_t InferredPtrDiff;

  SCEVBackedgeTakenAnalyzer(ScalarEvolution &SE, InputBuilder &InferredInputs)
      : SE(SE), Inferred(InferredInputs) {}

  bool isMulNeg(const SCEV *Expr) {
    const SCEVMulExpr *Mul = dyn_cast<SCEVMulExpr>(Expr);
    if (Mul == nullptr)
      return false;

    uint64_t Bits = SE.getTypeSizeInBits(Mul->getType());
    APInt Res(Bits, 1);

    for (const SCEV *Sub : Mul->operands()) {
      if (const SCEVConstant *Const = dyn_cast<SCEVConstant>(Sub))
        Res *= Const->getAPInt();
    }
    return Res.isNegative();
  }

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
    SmallVector<const SCEV *, 8> SubNormal;
    SmallVector<const SCEV *, 8> SubNeg;

    // Handle constant operands first
    for (const SCEV *Sub : Expr->operands()) {
      if (const SCEVConstant *Const = dyn_cast<SCEVConstant>(Sub)) {
        Res += Const->getAPInt();
      } else if (isMulNeg(Sub)) {
        SubNeg.push_back(Sub);
      } else {
        SubNormal.push_back(Sub);
      }
    }

    APInt DesiredNormal = Desired - Res;
    for (const SCEV *Sub : SubNormal) {
      // We hope that there is just one unknown in a "positive" position.
      // Otherwise we end up with more iterations than desired, but this should
      // not be too bad either.
      Optional<APInt> SubRes = visit(Sub, DesiredNormal);
      if (!SubRes)
        return None;
      Res += *SubRes;
    }
    Desired -= Res;

    for (const SCEV *Sub : SubNeg) {
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
      // This case should not occur.  Global variables typically do not occur,
      // because they require a `load` instruction, which is typically
      // loop-invariant.  Also, scalar evolution analysis appears to fail on
      // them.
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

// From ClangLinkerWrapper
static std::string getMainExecutable(const char *Name) {
  void *Ptr = (void *)(intptr_t)&getMainExecutable;
  auto COWPath = sys::fs::getMainExecutable(Name, Ptr);
  return sys::path::parent_path(COWPath).str();
}

bool ExplorativeLVPass::ProgramPaths::findAll() {
  if (Status == 1)
    return true;
  if (Status == -1)
    return false;

  Status = -1;
  if (XLVMetric == Metric::Benchmark) {
    auto CCPath = sys::findProgramByName("cc");
    if (!CCPath) {
      errs() << "XLV: could not find cc executable\n";
      return false;
    }
    CC = CCPath.get();

    if (!XLVBenchmarkPinCpu.empty()) {
      auto TasksetPath = sys::findProgramByName("taskset");
      if (!TasksetPath) {
        errs() << "XLV: could not find taskset executable\n";
        return false;
      }
      Taskset = TasksetPath.get();
    }
  } else if (XLVMetric == Metric::MCA) {
    // If we only compiled llvm-mca but didn't install it in PATH, we would not
    // find it.  `getMainExecutable()` is here to help.
    auto MCAPath =
        sys::findProgramByName("llvm-mca", {getMainExecutable("llvm-mca")});
    if (!MCAPath) {
      errs() << "XLV: could not find taskset executable\n";
      return false;
    }
    MCA = MCAPath.get();
  }

  Status = 1;
  return true;
}

// This is to be replaced by std::bit_width() when we have C++20
static unsigned pow2Count(unsigned I) {
  unsigned Count = 1;
  while (I >>= 1)
    ++Count;
  return Count;
}

PreservedAnalyses ExplorativeLVPass::run(Function &F,
                                         FunctionAnalysisManager &AM) {
  ExplorativeLVPass *MainPass = AM.getResult<ExplorativeLVAnalysis>(F).Pass;
  auto &LI = AM.getResult<LoopAnalysis>(F);
  auto &SE = AM.getResult<ScalarEvolutionAnalysis>(F);

  if (MainPass) { // "child" pass
    LoopFuncInfo *Factors = MainPass->getLoopFuncInfo(F);

    // Find loop
    auto LoopIt = LI.begin();
    if (LoopIt + 1 != LI.end()) {
      LLVM_DEBUG(dbgs() << "XLV: expected exactly one loop in loop function "
                        << F.getName() << '\n');
      return PreservedAnalyses::all();
    }
    Loop *L = *LoopIt;

    // Add VF/IF/UF
    setFactorMetadata(*L, Factors->VF, Factors->IF, Factors->UF);

    // Infer inputs (if desired, i. e. MainPass->InferredInputs is present)
    if (MainPass->InferredInputs) {
      inferInputs(F, *L, SE, *MainPass->InferredInputs);

      // The inputs should be inferred once only and not for every VF-IF
      // combination.
      MainPass->InferredInputs = nullptr;
    }

    return PreservedAnalyses::all();
  }

  if (!Paths.findAll()) {
    LLVM_DEBUG(dbgs() << "XLV: skipping " << F.getName()
                      << " because of missing program paths\n");
    return PreservedAnalyses::all();
  }

  // main/"parent" pass
  if (!XLVOnlyFunc.empty() && F.getName() != XLVOnlyFunc)
    return PreservedAnalyses::all();

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

  if (XLVMetric != Metric::Dummy) {
    // Initialize pipeline only if there is something to do.
    if (!OptPipeline)
      OptPipeline = new OptPipelineContainer(TM, PTO, PGOOpt, this, TLI);
    // If this is a code size evaluation, we can re-use the same backend
    // pipeline.
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

    if (LoopFuncInfos.size() == 0) {
      /* Generate factors we want to explore */
      unsigned VFCount = pow2Count(XLVMaxVF);
      unsigned IFCount = pow2Count(XLVMaxIF);
      if (XLVMetric == Metric::InstCount || XLVMaxUF == 0) {
        // For the InstCount metric, we only consider UF 1, everything else
        // should not interesting when counting instructions.
        unsigned UF = XLVMetric == Metric::InstCount ? 1 : 0;
        LoopFuncInfos.reserve(VFCount * IFCount);
        for (unsigned VF = 1; VF <= XLVMaxVF; VF *= 2) {
          for (unsigned IF = 1; IF <= XLVMaxIF; IF *= 2)
            LoopFuncInfos.emplace_back(VF, IF, UF);
        }
      } else {
        unsigned UFCount = XLVMaxUF + 1;
        LoopFuncInfos.reserve(VFCount * IFCount * UFCount);
        for (unsigned VF = 1; VF <= XLVMaxVF; VF *= 2) {
          for (unsigned IF = 1; IF <= XLVMaxIF; IF *= 2) {
            for (unsigned UF = 1; UF <= XLVMaxUF; ++UF)
              LoopFuncInfos.emplace_back(VF, IF, UF);
            LoopFuncInfos.emplace_back(VF, IF, UINT_MAX);
          }
        }
      }

      if (XLVRandomizeOrder) {
        std::random_device RD;
        std::default_random_engine RNG(RD());
        std::shuffle(LoopFuncInfos.begin(), LoopFuncInfos.end(), RNG);
      }
    }
  }

  unsigned LoopNo = 0;
  do {
    Loop *L = Worklist.pop_back_val();
    StringRef LoopName = L->getName();
    if (XLVOnlyLoop.empty() || LoopName == XLVOnlyLoop) {
      LLVM_DEBUG(dbgs() << "XLV: --- processing loop " << LoopNo << " '"
                        << LoopName << "' in " << F.getName() << " ---\n");
      auto It = ForcedFactors.find((F.getName() + Twine(',') + LoopName).str());
      if (It != ForcedFactors.end()) {
        unsigned VF, IF, UF;
        std::tie(VF, IF, UF) = It->second;
        LLVM_DEBUG(dbgs() << "XLV: enforcing VF " << VF << ", IF " << IF
                          << ", UF " << UF << '\n');
        setFactorMetadata(*L, VF, IF, UF);
      } else if (XLVMetric != Metric::Dummy) {
        if (processLoop(F, *L, LoopNo, SE, TLI)) {
          LLVM_DEBUG(dbgs() << "XLV: processing loop failed, letting the "
                               "AutoVectorizer determine VF and IF\n");
        }
        LoopFuncInfoMap.clear();
      }
    }
    ++LoopNo;
  } while (!Worklist.empty());

  return PreservedAnalyses::all();
}
