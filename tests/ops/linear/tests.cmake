add_library(ninfer_linear_test_support STATIC
  "${CMAKE_CURRENT_LIST_DIR}/linear_test_common.cpp")
ninfer_test_includes(ninfer_linear_test_support)
ninfer_op_oracle_options(ninfer_linear_test_support)
target_link_libraries(ninfer_linear_test_support PUBLIC ninfer_ops)

ninfer_add_op_test(ninfer_linear_q4_g32_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q4_g32_a16.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_q4_coarse_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q4_coarse_a16.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_q4_coarse_a8_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q4_coarse_a8.cpp"
  LIBRARIES ninfer_linear_test_support)

# The same binary with NINFER_Q4_EXACT: the exact route on the trunk shapes, deterministic.
add_test(NAME ninfer_linear_q4_coarse_a8_exact_test COMMAND ninfer_linear_q4_coarse_a8_test)
set_tests_properties(ninfer_linear_q4_coarse_a8_exact_test PROPERTIES
  ENVIRONMENT "NINFER_Q4_EXACT=1" SKIP_RETURN_CODE 77)

ninfer_add_op_test(ninfer_linear_q4_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q4_a16.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_q5_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q5_a16.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_q6_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q6_a16.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_q8_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q8_a16.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_nvfp4_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_nvfp4_a16.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_nvfp4_a4_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_nvfp4_a4.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_fp8_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_fp8_a16.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_gemma4_shapes_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gemma4_shapes.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_fp8_row_cublas_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_fp8_row_cublas.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_fp8_row_quantize_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_fp8_row_quantize.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_fp8_a8_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_fp8_a8.cpp"
  LIBRARIES ninfer_linear_test_support)

ninfer_add_op_test(ninfer_linear_bf16_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_bf16_a16.cpp"
  LIBRARIES ninfer_linear_test_support)
