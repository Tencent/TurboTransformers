option(OPENMP_ENDABLE "Build OpenMP" ON)

if(OPENMP_ENDABLE)
  find_package(OpenMP)
  if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    message(STATUS "OpenMP FLAGS ${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
  else()
    message(WARNING "OpenMP is not supported by the compiler")
  endif()
else()
  message(STATUS "OpenMP is not used")
endif()


add_library(cpu_bert_embedding_op cpu_bert_embedding_op.cpp)
target_link_libraries(cpu_bert_embedding_op tensor math_function)

add_library(cpu_layernorm_op cpu_layernorm_op.cpp)
target_link_libraries(cpu_layernorm_op math_function)