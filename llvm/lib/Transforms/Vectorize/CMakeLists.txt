add_llvm_component_library(LLVMVectorize
  ExplorativeLV.cpp
  LoadStoreVectorizer.cpp
  LoopVectorizationLegality.cpp
  LoopVectorize.cpp
  SLPVectorizer.cpp
  Vectorize.cpp
  VectorCombine.cpp
  VPlan.cpp
  VPlanHCFGBuilder.cpp
  VPlanPredicator.cpp
  VPlanSLP.cpp
  VPlanTransforms.cpp
  VPlanVerifier.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms/Vectorize

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Analysis
  CodeGen
  Core
  Passes
  Support
  TransformUtils
  )
