pybind11_add_module(fast_transformers_cxx
        pybind.cpp)

target_link_libraries(fast_transformers_cxx PRIVATE blas profiler bert_embedding bert_attention bert_intermediate bert_output sequence_pool prepare_bert_masks)

SET(PY_PACKAGE_DIR ${CMAKE_CURRENT_BINARY_DIR}/pypackage)
file(GLOB_RECURSE PY_PROJ_FILES ${CMAKE_CURRENT_SOURCE_DIR}/fast_transformers/*.py)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in ${PY_PACKAGE_DIR}/setup.py @ONLY)

add_custom_command(OUTPUT ${PY_PACKAGE_DIR}/.timestamp
        COMMAND "${CMAKE_COMMAND}" -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/fast_transformers ${PY_PACKAGE_DIR}/fast_transformers
        COMMAND "${CMAKE_COMMAND}" -E copy
        "$<TARGET_FILE:fast_transformers_cxx>"
        "${PY_PACKAGE_DIR}/fast_transformers/$<TARGET_FILE_NAME:fast_transformers_cxx>"
        COMMAND env CC=${CMAKE_C_COMPILER} ${PYTHON_EXECUTABLE} setup.py bdist_wheel
        COMMAND ${CMAKE_COMMAND} -E touch ${PY_PACKAGE_DIR}/.timestamp
        WORKING_DIRECTORY ${PY_PACKAGE_DIR}
        DEPENDS fast_transformers_cxx ${PY_PROJ_FILES})
add_custom_target(fast_transformers ALL DEPENDS ${PY_PACKAGE_DIR}/.timestamp)


if (WITH_GPU)
  file(GLOB PY_TEST_FILES tests/gpu_*_test.py)
else ()
  file(GLOB PY_TEST_FILES tests/[^gpu]*_test.py)
endif ()

foreach (py_test_file ${PY_TEST_FILES})
    get_filename_component(py_test_trg_name ${py_test_file} NAME_WE)
    message(STATUS "Add python test ${py_test_file}")
    add_test(NAME
            ${py_test_trg_name}
            COMMAND
            ${CMAKE_COMMAND} -E env
            PYTHONPATH=${PY_PACKAGE_DIR}
            ${PYTHON_EXECUTABLE} ${py_test_file}
            WORKING_DIRECTORY
            ${CMAKE_CURRENT_BINARY_DIR})
endforeach ()
