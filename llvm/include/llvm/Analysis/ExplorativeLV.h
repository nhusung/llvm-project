//===- ExplorativeLV.h - Helpers for ExplorativeLVPass ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helper analysis passes for explorative loop vectorization.  By default, their
// analysis result is just nullptr, however ExplorativeLVPass creates pipelines
// and in these sub-pipelines we can return a pointer to the "main"
// ExplorativeLVPass.  This permits passing results back.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_EXPLORATIVELV_H
#define LLVM_ANALYSIS_EXPLORATIVELV_H

#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"

namespace llvm {

class ExplorativeLVPass;

class ExplorativeLVAnalysis : public AnalysisInfoMixin<ExplorativeLVAnalysis> {
  friend AnalysisInfoMixin<ExplorativeLVAnalysis>;
  static AnalysisKey Key;

  ExplorativeLVPass *Pass = nullptr;

public:
  // We cannot directy return a pointer, so wrap it in a struct
  struct Result {
    ExplorativeLVPass *Pass;
  };

  ExplorativeLVAnalysis() = default;
  ExplorativeLVAnalysis(ExplorativeLVPass *Pass) : Pass(Pass) {}

  Result run(const Function &, FunctionAnalysisManager &) { return {Pass}; }
};

// Legacy Pass Manager (for code generation)

class ExplorativeLVHelper : public ImmutablePass {
  ExplorativeLVPass *Pass = nullptr;

public:
  static char ID;
  ExplorativeLVHelper();
  ExplorativeLVHelper(ExplorativeLVPass *MainPass);

  ExplorativeLVPass *getMainPass() { return Pass; }
};

} // namespace llvm

#endif // LLVM_ANALYSIS_EXPLORATIVELV_H
