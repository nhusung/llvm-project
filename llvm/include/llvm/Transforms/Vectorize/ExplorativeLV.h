//===- ExplorativeLV.h ----------------------------------------------------===//
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

#ifndef LLVM_TRANSFORMS_EXPLORATIVELV_H
#define LLVM_TRANSFORMS_EXPLORATIVELV_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include <cassert>

namespace llvm {

struct MachineCodeExplorer;

class ExplorativeLVPass : public PassInfoMixin<ExplorativeLVPass> {
public:
  class InputBuilder;

  enum class Metric { Disable, InstCount, MCA, Benchmark };

  friend struct MachineCodeExplorer;
  static const uint64_t InvalidCosts = UINT64_MAX;

private:
  struct OptPipelineContainer;
  struct NullCGPipelineContainer;

  // Things required to reconstruct the pipelines
  TargetMachine *TM;
  PipelineTuningOptions PTO;
  Optional<PGOOptions> PGOOpt;

  OptPipelineContainer *OptPipeline = nullptr;

  // Code generation pipeline with no output
  // This is used for instruction count mode.  Otherwise we need to rebuild
  // the pipeline every time, since we cannot simply replace the output stream.
  NullCGPipelineContainer *NullCGPipeline = nullptr;

  bool processLoop(Function &F, Loop &L, ScalarEvolution &SE,
                   TargetLibraryInfo &TLI);

  // For benchmarking mode: It is easier to infer the inputs in the "children"
  // pipelines.  This is used to pass them back to the "parent" pipeline.
  InputBuilder *InferredInputs = nullptr;

  struct EvaluationInfo {
    unsigned VF;
    unsigned IF;
    uint64_t Costs;
  };
  DenseMap<Function *, EvaluationInfo> *EvaluationInfoMap = nullptr;
  EvaluationInfo *getEvalutaionInfo(const Function &F) {
    assert(EvaluationInfoMap && "No EvaluationInfoMap present");
    auto It = EvaluationInfoMap->find(&F);
    return It == EvaluationInfoMap->end() ? nullptr : &It->getSecond();
  }

public:
  ExplorativeLVPass(TargetMachine *TM, PipelineTuningOptions PTO,
                    Optional<PGOOptions> PGOOpt)
      : TM(TM), PTO(PTO), PGOOpt(PGOOpt) {}

  ExplorativeLVPass(ExplorativeLVPass &&Pass)
      : TM(Pass.TM), PTO(std::move(Pass.PTO)), PGOOpt(std::move(Pass.PGOOpt)),
        OptPipeline(Pass.OptPipeline), NullCGPipeline(Pass.NullCGPipeline) {
    assert(!EvaluationInfoMap && "EvaluationInfoMap present during move");
    assert(!InferredInputs && "InferredInputs present during move");

    Pass.OptPipeline = nullptr;
    Pass.NullCGPipeline = nullptr;
  }

  ExplorativeLVPass(const ExplorativeLVPass &Pass) = delete;

  ~ExplorativeLVPass();

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_EXPLORATIVELV_H
