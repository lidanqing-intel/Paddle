cc_library(benchmark SRCS benchmark.cc DEPS enforce)
cc_test(test_benchmark SRCS benchmark_tester.cc DEPS benchmark)

set(QAT_PERF_DEPS paddle_inference_api paddle_fluid_api ir_pass_manager analysis_predictor tensor profiler)
cc_library(qat_helper SRCS test_helper.h DEPS ${QAT_PERF_DEPS})
cc_binary(qat_perf_benchmark SRCS qat_image_benchmark.cc DEPS qat_helper paddle_analysis_config)
