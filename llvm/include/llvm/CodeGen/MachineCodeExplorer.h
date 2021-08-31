//===- MachineCodeExplorer.h ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass contains functionality to analyse machine code and return a cost
// estimate, either based on the machine instruction count or on the total
// cycles predicted by llvm-mca, to the explorer instance
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Function.h"
#include "llvm/InitializePasses.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

struct MachineCodeExplorer : public MachineFunctionPass {
  static char ID; // Pass identification, replacement for typeid
  MachineCodeExplorer() : MachineFunctionPass(ID) {
    initializeMachineCodeExplorerPass(*PassRegistry::getPassRegistry());
  }
  int getCodeSizeEvaluation(MachineFunction &Fn, bool OnlyLoop);
  static int PerformMCACostCalc(std::string FileName, std::string TargetCPU,
                                std::string TargetTriple,
                                raw_pwrite_stream *OutStream);

  bool runOnMachineFunction(MachineFunction &F) override;

  bool IsMCARun = false;
  int *Result = nullptr;
  std::string FileName = "ExploredAssembly.tmp";
};