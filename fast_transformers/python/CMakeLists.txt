pybind11_add_module(fast_transformers
        pybind.cpp)

target_link_libraries(fast_transformers PRIVATE bert_embedding)


file(GLOB PY_TEST_FILES tests/*_test.py)

foreach (py_test_file ${PY_TEST_FILES})
    get_filename_component(py_test_trg_name ${py_test_file} NAME_WE)
    message(STATUS "Add python test ${py_test_file}")
    add_test(NAME
            ${py_test_trg_name}
            COMMAND
            ${CMAKE_COMMAND} -E env
            PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}
                ${PYTHON_EXECUTABLE} ${py_test_file}
            WORKING_DIRECTORY
                ${CMAKE_CURRENT_SOURCE_DIR}/tests)
endforeach ()
