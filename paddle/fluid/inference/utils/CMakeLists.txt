cc_library(benchmark SRCS benchmark.cc DEPS enforce)
cc_test(test_benchmark SRCS benchmark_tester.cc DEPS benchmark)

SET(GTEST_SOURCES_DIR ${THIRD_PARTY_PATH}/gtest)
SET(GTEST_INSTALL_DIR ${THIRD_PARTY_PATH}/install/gtest)
SET(GTEST_INCLUDE_DIR "${GTEST_INSTALL_DIR}/include" CACHE PATH "gtest include directory." FORCE)

include_directories(${GTEST_SOURCES_DIR})
include_directories(${GTEST_INSTALL_DIR})
include_directories(${GTEST_INCLUDE_DIR})
# get_filename_component(PARENT_DIR ../ ABSOLUTE)
# include_directories(${PARENT_DIR}/tests/api)

# set(QAT_PERF_DEPS paddle_inference_api paddle_fluid_api ir_pass_manager analysis_predictor tensor profiler lod_tensor memory paddle_gtest_main gtest gflags glog)
set(QAT_PERF_DEPS paddle_gtest_main lod_tensor memory gtest gflags glog)
# cc_library(qat_helper SRCS ${PARENT_DIR}/tests/api/tester_helper.h DEPS ${QAT_PERF_DEPS})
cc_binary(qat_perf_benchmark SRCS qat_image_benchmark.cc DEPS ${QAT_PERF_DEPS})

# get_property(os_dependency_modules GLOBAL PROPERTY OS_DEPENDENCY_MODULES)
# target_link_libraries(${TARGET_NAME} ${cc_test_DEPS} ${os_dependency_modules} paddle_gtest_main lod_tensor memory gtest gflags glog)
# add_dependencies(${TARGET_NAME} ${cc_test_DEPS} paddle_gtest_main lod_tensor memory gtest gflags glog)
