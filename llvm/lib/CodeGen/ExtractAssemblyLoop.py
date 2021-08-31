
# Expected to be called from MachineCodeExplorer, giving an assembly file to
# look at as first argument, and as a second argument a name for the file that
# the extracted code should be saved in.
# This script will try to extract the loop contained in the assembly and write
# it to an output file for the MachineCodeExplorer to run llvm-mca on.
# Favoured is a vector loop, i.e., a loop that has annotations starting with
# vector. . If such blocks cannot be found, we are looking for the scalar for
# or while loop.

import sys

input_file = sys.argv[1]
output_file = sys.argv[2]

# we can't rely on the blocks appearing in a particular order, so in any case we
# need a list to collect the scalar loop in at least until we found the first
# block of a vector loop. Simply keep the blocks as a list of strings.
scalar_loop = []
vector_loop = []


with open(input_file, "r") as input:
  l = input.readline()
  while True:
    if l == "":
      break
    if l.startswith(".LBB") and "# %vector." in l:
      # read in lines as long as we don't see the beginning of the
      # next block
      vector_loop.append(l)
      while True:
        l = input.readline()
        if l == "" or l.startswith(".LBB") and not ("# %vector." in l):
          break
        vector_loop.append(l)
    elif l.startswith(".LBB") and ("# %for." in l or "# %while." in l):
      # read in lines as long as we don't see the beginning of the
      # next block
      scalar_loop.append(l)
      while True:
        l = input.readline()
        if l == "" or l.startswith(".LBB") and not ("# %for." in l or "# %while." in l):
          break
        scalar_loop.append(l)
    else:
      l = input.readline()
        
      

if scalar_loop == [] and vector_loop == []:
  print("Something went wrong, found no loop in input file!")
  sys.exit(1)

with open(output_file, "w") as output:
  if not vector_loop == []:
    for line in vector_loop:
      output.write(line)
  else:
    for line in scalar_loop:
      output.write(line)
