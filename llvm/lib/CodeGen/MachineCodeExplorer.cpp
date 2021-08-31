//===- MachineCodeExplorer.cpp --------------------------------------------===//
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

#include "llvm/CodeGen/MachineCodeExplorer.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Function.h"
#include "llvm/InitializePasses.h"
#include "llvm/Target/TargetMachine.h"
#include <chrono>
#include <experimental/filesystem>
#include <thread>

using namespace llvm;

// The simple, code-size bound approach: count how
// many MachineInsts the function contains
int MachineCodeExplorer::getCodeSizeEvaluation(MachineFunction &Fn,
                                               bool OnlyLoop) {

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
      return -1;
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
  if (VectorCount != 0)
    return VectorCount;
  else
    return ScalarCount;
}

void WaitForPrinter(std::string FileName) {
  // https://solarianprogrammer.com/2019/01/13/cpp-17-filesystem-write-file-watcher-monitor/
  std::chrono::duration<int, std::milli> interval =
      std::chrono::milliseconds(500);
  // If the file is not there yet, wait for it to show up
  while (true) {
    if (std::experimental::filesystem::exists(FileName))
      break;
    std::this_thread::sleep_for(interval);
  }
  // Then wait for the printer to finish writing the assembly
  std::experimental::filesystem::file_time_type LastRead =
      std::experimental::filesystem::last_write_time(FileName);
  interval = std::chrono::milliseconds(500);
  while (true) {
    std::this_thread::sleep_for(interval);
    // Check whether there has been a change to the file; if yes, that means
    // the assembly printer hasn't finished yet
    if (std::experimental::filesystem::last_write_time(FileName) == LastRead)
      // Finished!
      break;
    LastRead = std::experimental::filesystem::last_write_time(FileName);
  }
}

// Executes the command line and returns the total output it sees
// Based on a Stackoverflow solution here:
// https://stackoverflow.com/questions/478898/how-do-i-execute-a-command
// -and-get-the-output-of-the-command-within-c-using-po
std::string exec(const char *cmd) {
  std::array<char, 128> buffer;
  std::string Res;
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
  assert(pipe &&
         "MachineCodeExplorer cannot open a command line to execute llvm-mca");
  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    Res += buffer.data();
  }
  return Res;
}

// The more complex approach, using the runtime estimation mechanisms
// already present in LLVM: llvm-mca
// (To be called from the pass that performs the exploration)
// FIXME: This only works when the compilation is executed from within
// the build folder of llvm-project
int MachineCodeExplorer::PerformMCACostCalc(std::string FileName,
                                            std::string TargetCPU,
                                            std::string TargetTriple,
                                            raw_pwrite_stream *OutStream) {
  // First, make sure to wait for the printer to finish, such that
  // the assembly file under inspection is complete
  WaitForPrinter(FileName);

  // Before running the analyser, reduce the code we look at to the actual
  // loop code. The script will try to find a vector loop first, but fall back
  // to the for/while loop if no vector loop is found
  // FIXME: This won't work on any architecture other than x86
  // Uncomment and change exec call if you really want to use it!
  /*std::string Command = "python ../llvm/lib/CodeGen/ExtractAssemblyLoop.py ";
  Command.append(FileName);
  Command.append(" ");
  Command.append(ReducedFileName);*/

  std::string ReducedFileName = "loop.tmp";
  // Build the most specific command line we can generate from
  // the info in TM
  // First, let's wait for a second to make sure printing the file has finished
  std::string CommandMCA = "bin/llvm-mca";
  CommandMCA.append(" --mcpu=" + TargetCPU);
  CommandMCA.append(" --mtriple=" + TargetTriple);
  CommandMCA.append(" --instruction-info=false");
  CommandMCA.append(" --resource-pressure=false ");
  CommandMCA.append(ReducedFileName);
  // Immediately reduce the output to the part we're interested in
  CommandMCA.append(" | grep 'Total Cycles' | cut -f2- -d:");

  // Retrieve result
  /*Command.append(" && ");
  Command.append(CommandMCA);
  std::string Output = exec(Command.c_str());*/
  std::string Output = exec(CommandMCA.c_str());

  // Transform result to a number
  int Result = std::atoi(Output.c_str());

  // And last, remove the tmp file
  std::string Remove = "rm " + FileName;
  exec(Remove.c_str());

  return Result;
}

bool MachineCodeExplorer::runOnMachineFunction(MachineFunction &MF) {
  // If this is no assembly exploration with llvm-mca, call the right
  // function. Otherwise, do nothing
  if (!IsMCARun) {
    int Result = getCodeSizeEvaluation(
        MF, MF.getFunction().getContext().getMCInstCount() == 1);
    MF.getFunction().getContext().setMCInstCount(Result);
  }
  return true;
}

char MachineCodeExplorer::ID = 0;
char &llvm::MachineCodeExplorerID = MachineCodeExplorer::ID;
INITIALIZE_PASS(MachineCodeExplorer, "explorative-costcalc",
                "Count MIs for explorative LV", false, false)
FunctionPass *llvm::createMachineCodeExplorer() {
  return new MachineCodeExplorer();
}
FunctionPass *llvm::createMachineCodeExplorer(int *ResultPtr,
                                              std::string File) {
  MachineCodeExplorer *MCE = new MachineCodeExplorer();
  MCE->IsMCARun = true;
  MCE->Result = ResultPtr;
  MCE->FileName = File;
  return MCE;
}
