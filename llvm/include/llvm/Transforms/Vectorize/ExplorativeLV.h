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

#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"

namespace llvm {

class ExplorativeLVPass : public PassInfoMixin<ExplorativeLVPass> {

public:
  ExplorativeLVPass(TargetMachine *TM, PipelineTuningOptions PTO,
                    Optional<PGOOptions> PGOOpt)
      : TM(TM), PTO(PTO), PGOOpt(PGOOpt) {}

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  static char ID; // Pass identification, replacement for typeid

  static StringRef name() { return "ExplorativeLV"; }

private:
  raw_pwrite_stream *init(legacy::PassManager &PM, int VecOps,
                          std::string AssemblyFileName);

  // FIXME: get an actual value from somewhere
  bool DebugLogging = false;

  // Things required to reconstruct the compilation pipeline
  TargetMachine *TM;
  PipelineTuningOptions PTO;
  Optional<PGOOptions> PGOOpt;
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_EXPLORATIVELV_H
