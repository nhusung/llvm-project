//===- MachineCodeExplorer.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass contains functionality to analyze machine code and return a cost
// estimate, based on the machine instruction count, to the explorer instance
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/ExplorativeLV.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/InitializePasses.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Vectorize/ExplorativeLV.h"

using namespace llvm;

#define DEBUG_TYPE "machine-code-explorer"

namespace llvm {

struct MachineCodeExplorer : public MachineFunctionPass {
  static char ID; // Pass identification, replacement for typeid
  MachineCodeExplorer() : MachineFunctionPass(ID) {
    initializeMachineCodeExplorerPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  bool runOnMachineFunction(MachineFunction &F) override;
};

extern cl::opt<bool> ExplorePlain;

} // namespace llvm

// The simple, code-size bound approach: count how
// many MachineInsts the function contains
static uint64_t getCodeSizeEvaluation(MachineFunction &Fn, unsigned VF) {
  if (VF > 1) {
    bool Vectorized = false;
    // If we expected to vectorize, only proceed if there is evidence
    // vectorization actually happened
    for (MachineBasicBlock &BB : Fn) {
      if (BB.getName().rfind("vector.") != std::string::npos) {
        Vectorized = true;
        break;
      }
    }
    if (!Vectorized)
      return ExplorativeLVPass::InvalidCosts;
  }

  if (ExplorePlain)
    return Fn.getInstructionCount();

  unsigned VectorCount = 0;
  unsigned ScalarCount = 0;
  for (MachineBasicBlock &BB : Fn) {
    // Collect loop blocks
    if (BB.getName().rfind("vector.") != std::string::npos) {
      VectorCount += BB.size();
    } else if ((BB.getName().rfind("while.") != std::string::npos) ||
               (BB.getName().rfind("for.") != std::string::npos)) {
      ScalarCount += BB.size();
    }
  }
  return (VectorCount != 0) ? VectorCount : ScalarCount;
}

bool MachineCodeExplorer::runOnMachineFunction(MachineFunction &MF) {
  ExplorativeLVPass *Pass = getAnalysis<ExplorativeLVHelper>().getMainPass();
  if (!Pass)
    return true;

  ExplorativeLVPass::EvaluationInfo *Result =
      Pass->getEvalutaionInfo(MF.getFunction());
  if (!Result)
    return true;

  Result->Costs = getCodeSizeEvaluation(MF, Result->VF);
  return true;
}

void MachineCodeExplorer::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<ExplorativeLVHelper>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

char MachineCodeExplorer::ID = 0;
char &llvm::MachineCodeExplorerID = MachineCodeExplorer::ID;

INITIALIZE_PASS_BEGIN(MachineCodeExplorer, "machine-code-explorer",
                      "Count MIs for explorative LV", false, false)
INITIALIZE_PASS_DEPENDENCY(ExplorativeLVHelper)
INITIALIZE_PASS_END(MachineCodeExplorer, "machine-code-explorer",
                    "Count MIs for explorative LV", false, false)

FunctionPass *llvm::createMachineCodeExplorer() {
  return new MachineCodeExplorer();
}
