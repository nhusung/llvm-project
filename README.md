# About This Project

This is a fork of the [LLVM project](https://github.com/llvm/llvm-project) where we aim to to improve the choice of loop vectorization factors using an explorative approach.
The work started as a master's thesis “Determining the Perfect Vectorization Factor” by [Anna Welker](https://github.com/aWelker/llvm-project) at the Compiler Design Lab at Saarland University.
Because the results were not as good as expected, Nils Husung picked up this project in his bachelor's thesis “Improving the Choice of Vectorization Factors,” also at the Compiler Design Lab.

The main idea is to extract innermost loops into functions and compile them ahead via a clone of the optimization and the code generation pipeline.
This is done by the `ExplorativeLV` pass implemented in `llvm/lib/Transforms/Vectorize/ExplorativeLV.cpp`.
To evaluate the costs, there are three metrics available: a simple one focused on the instruction count (implementation in `llvm/lib/CodeGen/MachineCodeExplorer.cpp`), one based on [`llvm-mca`](https://www.llvm.org/docs/CommandGuide/llvm-mca.html) and a benchmarking metric.
The benchmarking metric automatically infers “valid” inputs for the extracted loops and benchmarks them.
The focus of the Bachelor's thesis was on this metric.
The MCA metric is currently a little buggy because it does not reliably isolate the assembly of the vectorized loop.

To enable our explorative loop vectorization (XLV) with the benchmarking metric, invoke `clang` as follows:

    clang -O3 -mllvm --xlv-metric=benchmark [<other flags>] <input>

If you are interested in the debugging output, add `-mllvm --debug-only=explorative-lv`.
Also remember to select the appropriate target (e.g. `-march=x86-64-v3` for machines with AVX2).
The ExplorativeLV pass has lots of configuration options.
If you have compiled the `opt` tool, you can use `opt --help-hidden` to see all the options.
These are prefixed with `--xlv`.
A particularly helpful option for debugging/evaluation purposes might be `--xlv-artifacts-dir=<dir>`, which places all build artifacts created by the ExplorativeLV pass (optimized IR of the loop functions, benchmarking executable or assembly code with for `llvm-mca`) in the given directory.

If you are interested in the evaluation of this tool, have a look at https://gitlab.cs.uni-saarland.de/s8nihusu/xlv-evaluation.

---

From here on follows the original readme:

# The LLVM Compiler Infrastructure

This directory and its sub-directories contain source code for LLVM,
a toolkit for the construction of highly optimized compilers,
optimizers, and run-time environments.

The README briefly describes how to get started with building LLVM.
For more information on how to contribute to the LLVM project, please
take a look at the
[Contributing to LLVM](https://llvm.org/docs/Contributing.html) guide.

## Getting Started with the LLVM System

Taken from https://llvm.org/docs/GettingStarted.html.

### Overview

Welcome to the LLVM project!

The LLVM project has multiple components. The core of the project is
itself called "LLVM". This contains all of the tools, libraries, and header
files needed to process intermediate representations and convert them into
object files.  Tools include an assembler, disassembler, bitcode analyzer, and
bitcode optimizer.  It also contains basic regression tests.

C-like languages use the [Clang](http://clang.llvm.org/) front end.  This
component compiles C, C++, Objective-C, and Objective-C++ code into LLVM bitcode
-- and from there into object files, using LLVM.

Other components include:
the [libc++ C++ standard library](https://libcxx.llvm.org),
the [LLD linker](https://lld.llvm.org), and more.

### Getting the Source Code and Building LLVM

The LLVM Getting Started documentation may be out of date.  The [Clang
Getting Started](http://clang.llvm.org/get_started.html) page might have more
accurate information.

This is an example work-flow and configuration to get and build the LLVM source:

1. Checkout LLVM (including related sub-projects like Clang):

     * ``git clone https://github.com/llvm/llvm-project.git``

     * Or, on windows, ``git clone --config core.autocrlf=false
    https://github.com/llvm/llvm-project.git``

2. Configure and build LLVM and Clang:

     * ``cd llvm-project``

     * ``cmake -S llvm -B build -G <generator> [options]``

        Some common build system generators are:

        * ``Ninja`` --- for generating [Ninja](https://ninja-build.org)
          build files. Most llvm developers use Ninja.
        * ``Unix Makefiles`` --- for generating make-compatible parallel makefiles.
        * ``Visual Studio`` --- for generating Visual Studio projects and
          solutions.
        * ``Xcode`` --- for generating Xcode projects.

        Some common options:

        * ``-DLLVM_ENABLE_PROJECTS='...'`` and ``-DLLVM_ENABLE_RUNTIMES='...'`` ---
          semicolon-separated list of the LLVM sub-projects and runtimes you'd like to
          additionally build. ``LLVM_ENABLE_PROJECTS`` can include any of: clang,
          clang-tools-extra, cross-project-tests, flang, libc, libclc, lld, lldb,
          mlir, openmp, polly, or pstl. ``LLVM_ENABLE_RUNTIMES`` can include any of
          libcxx, libcxxabi, libunwind, compiler-rt, libc or openmp. Some runtime
          projects can be specified either in ``LLVM_ENABLE_PROJECTS`` or in
          ``LLVM_ENABLE_RUNTIMES``.

          For example, to build LLVM, Clang, libcxx, and libcxxabi, use
          ``-DLLVM_ENABLE_PROJECTS="clang" -DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi"``.

        * ``-DCMAKE_INSTALL_PREFIX=directory`` --- Specify for *directory* the full
          path name of where you want the LLVM tools and libraries to be installed
          (default ``/usr/local``). Be careful if you install runtime libraries: if
          your system uses those provided by LLVM (like libc++ or libc++abi), you
          must not overwrite your system's copy of those libraries, since that
          could render your system unusable. In general, using something like
          ``/usr`` is not advised, but ``/usr/local`` is fine.

        * ``-DCMAKE_BUILD_TYPE=type`` --- Valid options for *type* are Debug,
          Release, RelWithDebInfo, and MinSizeRel. Default is Debug.

        * ``-DLLVM_ENABLE_ASSERTIONS=On`` --- Compile with assertion checks enabled
          (default is Yes for Debug builds, No for all other build types).

      * ``cmake --build build [-- [options] <target>]`` or your build system specified above
        directly.

        * The default target (i.e. ``ninja`` or ``make``) will build all of LLVM.

        * The ``check-all`` target (i.e. ``ninja check-all``) will run the
          regression tests to ensure everything is in working order.

        * CMake will generate targets for each tool and library, and most
          LLVM sub-projects generate their own ``check-<project>`` target.

        * Running a serial build will be **slow**.  To improve speed, try running a
          parallel build.  That's done by default in Ninja; for ``make``, use the option
          ``-j NNN``, where ``NNN`` is the number of parallel jobs, e.g. the number of
          CPUs you have.

      * For more information see [CMake](https://llvm.org/docs/CMake.html)

Consult the
[Getting Started with LLVM](https://llvm.org/docs/GettingStarted.html#getting-started-with-llvm)
page for detailed information on configuring and compiling LLVM. You can visit
[Directory Layout](https://llvm.org/docs/GettingStarted.html#directory-layout)
to learn about the layout of the source code tree.
