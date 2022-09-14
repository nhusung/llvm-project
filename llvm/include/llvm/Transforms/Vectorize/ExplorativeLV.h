//===- ExplorativeLV.h ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass is inserted in the optimization pipeline immediately before the
// loop vectorizer.  When enabled, it retrieves the innermost loops of the
// function (as done in LoopVectorize), compiles them ahead with different
// vectorization factors and evaluates the costs according to some metric.  The
// best option is chosen and vectorization hints are added accordingly.
//
// In a bit more detail: each loop is extracted into a new function in a fresh
// module.  Several copies of the loop function are created, one for each
// combination of vectorization, interleaving and optionally unrolling factor to
// explore.  These are sent through a "clone" of the optimization and code
// generation pipeline.  When they arrive at the ExplorativeLV pass in the
// "child" pipeline, the factors are set as metadata.  Furthermore, inputs for
// the loops functions are inferred if the benchmarking metric is in use.  After
// optimization, unvectorized loops (that should have been vectorized) are
// filtered out and annotations llvm-mca are added (if the mca metric is in
// use).  After code generation, the costs of the options are evaluated, either
// using the MachineCodeExplorer (inst-count metric, actually the last pass
// which is part of the code generation pipeline), llvm-mca or via benchmarking.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_EXPLORATIVELV_H
#define LLVM_TRANSFORMS_EXPLORATIVELV_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include <cassert>
#include <tuple>

namespace llvm {

struct MachineCodeExplorer;

class ExplorativeLVPass : public PassInfoMixin<ExplorativeLVPass> {
public:
  class InputBuilder;

  /// Available metrics
  ///
  /// `Disable` disables the pass entirely, `Dummy` does not disable it but does
  /// not peform any exploration. This can be useful to enforce vectorization
  /// factors for a specific loop.
  enum class Metric { Disable, Dummy, InstCount, MCA, Benchmark };

  friend struct MachineCodeExplorer;

  struct LoopFuncInfo {
    unsigned VF, IF, UF;
    bool FullUnroll = false;
    Function *Func = nullptr;
    double Costs = INFINITY;
    LoopFuncInfo(unsigned VF, unsigned IF, unsigned UF)
        : VF(VF), IF(IF), UF(UF) {}

    void reset() {
      FullUnroll = false;
      Func = nullptr;
      Costs = INFINITY;
    }
  };

private:
  struct OptPipelineContainer;
  struct NullCGPipelineContainer;

  // Things required to reconstruct the pipelines
  TargetMachine *TM;
  PipelineTuningOptions PTO;
  Optional<PGOOptions> PGOOpt;

  OptPipelineContainer *OptPipeline = nullptr;

  /// Code generation pipeline with no output
  /// This is used for instruction count mode.  Otherwise we need to rebuild
  /// the pipeline every time, since we cannot simply replace the output stream.
  NullCGPipelineContainer *NullCGPipeline = nullptr;

  bool processLoop(Function &F, Loop &L, unsigned LoopNo, ScalarEvolution &SE,
                   TargetLibraryInfo &TLI);

  /// For benchmarking mode: It is easier to infer the inputs in the "children"
  /// pipelines.  This is used to pass them back to the "parent" pipeline.
  InputBuilder *InferredInputs = nullptr;

  SmallVector<LoopFuncInfo, 0> LoopFuncInfos;
  DenseMap<Function *, unsigned> LoopFuncInfoMap;
  LoopFuncInfo *getLoopFuncInfo(const Function &F) {
    auto It = LoopFuncInfoMap.find(&F);
    return It == LoopFuncInfoMap.end() ? nullptr : &LoopFuncInfos[It->second];
  }

  struct CSVOutputContainer;
  CSVOutputContainer *CSVOutput = nullptr;

  StringMap<std::tuple<unsigned, unsigned, unsigned>> ForcedFactors;

  /// Used to retrieve paths for external programs once only.
  struct ProgramPaths {
    std::string cc() {
      assert(Status == 1 && !CC.empty());
      return CC;
    }
    std::string mca() {
      assert(Status == 1 && !MCA.empty());
      return MCA;
    }
    std::string taskset() {
      assert(Status == 1 && !Taskset.empty());
      return Taskset;
    }

    bool findAll();

  private:
    int Status = 0;
    std::string CC;
    std::string MCA;
    std::string Taskset;
  } Paths;

public:
  ExplorativeLVPass(TargetMachine *TM, PipelineTuningOptions PTO,
                    Optional<PGOOptions> PGOOpt);

  ExplorativeLVPass(ExplorativeLVPass &&Pass)
      : TM(Pass.TM), PTO(std::move(Pass.PTO)), PGOOpt(std::move(Pass.PGOOpt)),
        OptPipeline(Pass.OptPipeline), NullCGPipeline(Pass.NullCGPipeline),
        CSVOutput(Pass.CSVOutput), ForcedFactors(std::move(Pass.ForcedFactors)),
        Paths(std::move(Pass.Paths)) {
    assert(LoopFuncInfoMap.size() == 0 &&
           "LoopFuncInfoMap present during move");

    Pass.OptPipeline = nullptr;
    Pass.NullCGPipeline = nullptr;
    Pass.CSVOutput = nullptr;
  }

  ExplorativeLVPass(const ExplorativeLVPass &Pass) = delete;

  ~ExplorativeLVPass();

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_EXPLORATIVELV_H
