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

#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/InitializePasses.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

#define DEBUG_TYPE "machine-code-explorer"

namespace {

struct MachineCodeExplorer : public MachineFunctionPass {
  static char ID; // Pass identification, replacement for typeid
  MachineCodeExplorer() : MachineFunctionPass(ID) {
    initializeMachineCodeExplorerPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &F) override;
};

} // end anonymous namespace

// The simple, code-size bound approach: count how
// many MachineInsts the function contains
static unsigned getCodeSizeEvaluation(MachineFunction &Fn, bool OnlyLoop) {

  if (!Fn.getFunction().getContext().isScalarExploration()) {
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
      return UINT_MAX;
  }

  if (!OnlyLoop)
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
  int Result = getCodeSizeEvaluation(
      MF, MF.getFunction().getContext().getMCInstCount() == 1);
  MF.getFunction().getContext().setMCInstCount(Result);
  return true;
}

char MachineCodeExplorer::ID = 0;
char &llvm::MachineCodeExplorerID = MachineCodeExplorer::ID;
INITIALIZE_PASS(MachineCodeExplorer, "machine-code-explorer",
                "Count MIs for explorative LV", false, false)
FunctionPass *llvm::createMachineCodeExplorer() {
  return new MachineCodeExplorer();
}
