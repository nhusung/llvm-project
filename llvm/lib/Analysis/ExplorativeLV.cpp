//===- ExplorativeLV.cpp - Helpers for ExplorativeLVPass --------*- C++ -*-===//
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

#include "llvm/Analysis/ExplorativeLV.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

AnalysisKey ExplorativeLVAnalysis::Key;

// Legacy Pass Manager

ExplorativeLVHelper::ExplorativeLVHelper() : ImmutablePass(ID) {
  initializeExplorativeLVHelperPass(*PassRegistry::getPassRegistry());
}

ExplorativeLVHelper::ExplorativeLVHelper(ExplorativeLVPass *MainPass)
    : ImmutablePass(ID), Pass(MainPass) {
  initializeExplorativeLVHelperPass(*PassRegistry::getPassRegistry());
}

char ExplorativeLVHelper::ID = 0;
INITIALIZE_PASS(ExplorativeLVHelper, "explorative-lv-helper",
                "Helper for Explorative Loop Vectorization", false, true)
