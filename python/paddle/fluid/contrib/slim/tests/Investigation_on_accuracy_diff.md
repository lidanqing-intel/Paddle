# Investigating accuracy issues in PaddlePaddle inference

## 0. Most frequent situations when accuracy issues occur
1. Accuracy regression in regular tests.
2. Introducing new hardware.
3. Developing new functionality (e.g. adding new operator kernel).
4. Testing with different environments, languages, data loaders 
    * e.g. python vs. C-API, training on python with PIL, inference on C++ with OpenCV
    * e.g. training on GPU, inference on CPU, with different kernel implementations on GPU and CPU. Use case: When working on `batch_norm`, the `sqrt` function implementation on GPU and native CPU had lower precision, whereas MKL-DNN implementation had much higher precision.

## 1. General accuracy investigation guidance
* Is full validation dataset used? testing on smaller datasets usually gives different and inaccurate results.
* Is the accuracy being calculated properly?
    * use case: 
        Accuracy issue with int8_mobilenet_ssd tester · Issue #18913 · PaddlePaddle/Paddle - a bug in the accuracy calculation algorithm.
* Is the issue present on another machine? If no:
    * make sure the very same revision is tested on both machines, with clean directory trees,
    * clean build directory,
    * compare the differences between the hardware
        * compare instruction set architecture (ISA) support by CPU (is AVX2 or AVX512 supported?),
        * which ISA is expected to be used by a particular operator? (e.g. usually AVX512 for DNNL-based conv2d or FC)
        * check if proper ISA is used by particular operator implementations
            * use GLOG to find out which ISA is used by the native operators,
            * use DNNL_VERBOSE to find out which ISA is used by DNNL-based operators,
        * check what is the difference between AVX512 and AVX2 in the final output,
    * compare the differences between environment configurations
        * are virtual machines / docker images used for testing?
            * They can lack support for CPU features (e.g. AVX512 support, run the `lscpu` command inside the VM/docker and outside of it and compare the supported features)
        * proper ISA support can be missing in older linux kernels (prior to 3.15),
        * libraries which PaddlePaddle depends on can be malfunctioning (e.g. after upgrade of system libraries),
    * use cases:
        * test_analyzer_resnet50 fails on E5-2660 v4 machine · Issue #18005 · PaddlePaddle/Paddle - difference in results between AVX2 and AVX512 ISA used + large magnitude of data values with very small accuracy diff threshold,
        * [MKL-DNN] Failure to run face model (demark) · Issue #18658 · PaddlePaddle/Paddle - after switching from machine with AVX2 to one with AVX512, the NCHW16C format was chosen (instead of NCHW for AVX2) by DNNL-based `tanh` operator kernel, then the `tanh`’s output, also in NCHW16C format was passed to the `fetch` operator which could not handle the NCHW16C format properly,
        * test_slim_int8_googlenet, test_analyzer_int8_googlenet etc fails on new 5117 machine · Issue #19505 · PaddlePaddle/Paddle - regression introduced by a commit + misconfigured machine (missing ISA support in the used CPU) + difference between AVX2 and AVX512 results + lack of relative error metric for data of large magnitude
* Is the issue present in the Release version and not in the Debug version (or vice versa)?
    * use printing debug messages in Release version,
    * if RelWithDebInfo version reproduces the issue try also a debugger (for debugging hints see below).
* Is the model being optimized in any way? If yes:
    * turn off the optimizations (e.g. passes); if that helped, find the offending pass turning the passes on one by one.
    * use cases investigated this way:
        * runtime_context_cache_pass breaks accuracy of FP32 C-API inference of ResNet50 and MobileNet-v1 · Issue #16609 · PaddlePaddle/Paddle,
        * The `fc_gru_fuse_pass` breaks accuracy in C-API CRNN-CTC inference. · Issue #14837 · PaddlePaddle/Paddle.
* Did the accuracy drop at some point in a branch history?
    * find the revision commit which introduced the regression, and then analyze the commit changes.
* Does a comparison check fail for different execution modes (e.g. MKL-DNN vs. native, python vs. C-API)?
    * find out which layer introduces a difference in outputs (e.g. starting from the top of the model and going down the tree, compare input or output variable’s tensors)
        * check the magnitude of differentiating outputs (ref vs. computed),
        * make sure a relative measure of difference between tensors is used for their comparison, i.e. if values of tensors reach 10000, requirement for their diff to fit into [-0.001, 0.001] range may be unrealizable, 
    * find out which kernel is used in the execution of the offending layer (e.g. add debug messages in the kernel’s `Compute()` methods or in a `GetExpectedKernelType()` method,
    * turn off MKL-DNN kernels for all the operators, then enable them one by one,
        * in mkldnn_placement_pass, by setting the pass’ `mkldnn_enabled_op_types` attribute, e.g. using AnalysisConfig class’ method `SetMKLDNNOp()`
        * or disable specific implementation from inside DNNL (see below).
    * use case:
        * conv_transpose_mkldnn has accuracy diff · Issue #17535 · PaddlePaddle/Paddle - comments in the issue show the steps of 
* Is a particular operator implementation the culprit? If yes:
    * modify a unit test to cover the case from the model and simplify it as long as it keeps failing, until you get the simplest case, easy to analyze manually,
    * write a simple program (can be a unit test) which applies the flawed operator kernel to prepared very simple input data, so that you could verify the correctness of the computations easily, even manually on a piece of paper,
        * use input tensors of very small dimensions,
        * initialize the input tensors with a patterned data (e.g. ascending integers for vectors, encoding rows and columns for matrices, like in [[11, 12], [21, 22]]),
    * add multiple test cases with various data patterns, specific failing cases may be a hint for where the bug in calculations is.
* What is the magnitude of the input tensors?
    * if the magnitude of the data is close to the data type’s max value, the precision may be lost during computation; make sure you understand IEEE floating point format and how precision is lost by performing binary operations on numbers that have large difference in magnitude.
* Does accuracy differ between single threaded and multiple threaded runs?
    * floating point calculation precision can depend on the order of operands and their magnitude.
* Enable printing logs using GLOG_v and/or DNNL_VERBOSE environment variables:
    * look for any unexpected messages,
    * look for nan/infs warnings in python output, turn on compiler numeric exceptions to see where the problem occurs (see below for hints on how to do that).
* Does INT8 accuracy fail (zero or lower than expected) and FP32 accuracy is good? If yes:
    * check if AVX512 VNNI is supported by the CPU (generally VNNI instructions handle INT8 operations much better when it comes to correctness),
    * turn off INT8 model optimizations (e.g. turn off various squashing procedures in the mkldnn_quantize_squash_pass); if that helps, turn them on one by one to find the offending one,
    * disable quantization of all operators and enable them one by one
        * you can use the `SetEnabledOpTypes()` method of the `MkldnnQuantizerConfig` class to enable quantization only for selected operators.
    * check the distribution of the FP32 data stored in the tensor to be quantized (quantization of data requires aliasing, which, depending on the distribution of tensor data, can have heavy impact on the accuracy, e.g. if 90 % of values in the tensor is in the range of [0, 10], and 5% falls into [1000, +] range, all the small values will be quantized into 0 or 1, losing the information on their variability),
        * make sure the scale used for quantization of the tensor matches the distribution of data in the tensor (e.g. if values in the tensor spread along the [-1.0, 1.0] range, using scale 0.125 for quantization is pointless, probably value close to 127 will be more appropriate, as the quantization will make the tensor data spread along the [-128, 127] full range of 8-bit integers).
    * if INT8 and FP32 kernels share the code:
        * make sure various input data formats (typically NCHW for FP32 and NHWC for INT8) are handled properly (e.g. it was the reason for accuracy loss by FC INT8 kernel in the Ernie model: NWC format could not be handled by DNNL inner product primitive inside the FC INT8 kernel)
        * check if weights have proper format for the input format, (for DNNL data formats see DNNL: Understanding Memory Formats)
        * check if input/output data has to be unsigned (e.g. after ReLU op) and whether that information is employed properly by quantize/dequantize operators (using quantization to signed INT8 when the data is unsigned very often reduces the accuracy)
    * otherwise, additionally:
        * analyze the INT8 kernel computation algorithm,
    * if INT8v2 quantization (post-training quantization) is applied:
        * make sure scaling factors are calculated using desired algorithm (usually KL for inputs/outputs, MAX per channel for weights),
        * accuracy depends heavily on the warmup batch size and warmup data (data used in a warmup phase to generate scaling factors for quantization),
        * modify the warmup batch size (usually 50 is enough, some models may give best accuracy with lower values, like 10, depending),
        * choose another set of input data for warmup,
    * if QATv2 procedure is applied and the QAT FP32 model achieves good accuracy:
        * save both FP32 and INT8 models using the `save_qat_model` test,
        * compare the accuracy you get using the FP32 and INT8,
        * starting from the first quantized operator compare its input and output tensors between QAT FP32 and INT8 model - they should contain very similar numbers kept as floats in the QAT and integers in the INT8 model,
        * for operators with weights (e.g. conv2d, fc) compare the weights tensors between QAT and INT8 model; their values should be very similar.
## 2. Disabling selected operator implementation inside DNNL
When we diagnose that there is a problem with a specific DNNL operator, it is possible to modify DNNL to remove some of implementations (for example AVX512) for the operator. As an example, this is how to do that for convolution.
1. Disable specific implementation of DNNL
    * Check in DNNL_VERBOSE which implementation is called
    * fork mkl-dnn and modify cpu_engine.cpp , but removing implementation you expect is faulty:
    ```
    INSTANCE(jit_avx512_common_convolution_fwd_t<f32>),
    INSTANCE(jit_avx512_common_convolution_bwd_data_t<f32>),
    //INSTANCE(jit_avx512_common_convolution_bwd_weights_t<f32>),   
    INSTANCE(jit_avx2_dw_convolution_fwd_t),
    INSTANCE(jit_avx2_dw_convolution_bwd_data_t),
    ```
2. Make your mkl-dnn available to PaddlePaddle (modify mkldnn.cmake).
3. Disable instruction set in DNNL. Modify the cpu_isa_traits.hpp file, replace true with false to disable some of the platforms:

   cpu_isa_traits.hpp:
    ```
    88 static inline bool mayiuse(const cpu_isa_t cpu_isa) {
    89     using namespace Xbyak::util;
    90
    91     switch (cpu_isa) {
    92     case sse42:
    93         return cpu.has(Cpu::tSSE42);
    94     case avx:
    95         return cpu.has(Cpu::tAVX);
    96     case avx2:
    97         return cpu.has(Cpu::tAVX2);
    98     case avx512_common:
    99         return cpu.has(Cpu::tAVX512F);
    100     case avx512_core:
    101         return true
    102             && cpu.has(Cpu::tAVX512F)
    103             && cpu.has(Cpu::tAVX512BW)
    104             && cpu.has(Cpu::tAVX512VL)
    105             && cpu.has(Cpu::tAVX512DQ);
    106     case avx512_core_vnni:
    107         return true
    108             && cpu.has(Cpu::tAVX512F)
    109             && cpu.has(Cpu::tAVX512BW)
    110             && cpu.has(Cpu::tAVX512VL)
    111             && cpu.has(Cpu::tAVX512DQ)
    112             && cpu.has(Cpu::tAVX512_VNNI);
    113     case avx512_mic:
    114         return true
    115             && cpu.has(Cpu::tAVX512F)
    116             && cpu.has(Cpu::tAVX512CD)
    117             && cpu.has(Cpu::tAVX512ER)
    118             && cpu.has(Cpu::tAVX512PF);
    119     case avx512_mic_4ops:
    120         return true
    121             && mayiuse(avx512_mic)
    122             && cpu.has(Cpu::tAVX512_4FMAPS)
    123             && cpu.has(Cpu::tAVX512_4VNNIW);
    124     case isa_any:
    125         return true;
    126     }
    127     return false;
    128 }
    ```
## 3. Basic logging
### 3.1 DNNL_VERBOSE description
(guide:[DNNL:Verbose Mode](https://intel.github.io/mkl-dnn/dev_guide_verbose.html))
After setting the DNNL_VERBOSE environment variable to a chosen level (0, 1 or 2), the DNNL library adds logging information to the standard output. The following information is provided:
* 1st line: DNNL library version and git hash,
* 2nd line: supported ISA and extensions,
* Subsequent lines:
eg. 
```
dnnl_verbose,exec,cpu,convolution,jit:avx2,forward_training,src_f32::blocked:aBcd8b:f0 wei_f32::blocked:ABcd8b8a:f0 bia_f32::blocked:a:f0 dst_f32::blocked:aBcd8b:f0,alg:convolution_direct,mb2_ic16oc16_ih7oh7kh5sh1dh0ph2_iw7ow7kw5sw1dw0pw2,0.026123
```
* Comments:
    * dnnl_verbose marker string
    * operation: create[:cache_hit], create[:cache_miss] or exec
    * engine kind: cpu or gpu
    * primitive name: convolution, reorder, sum, etc
    * primitive implementation
    * propagation: forward_training, forward_inference, or backward
    * information about input and output data types and formats
    * auxiliary information like algorithm name or number of inputs
    * a problem description in benchdnn format
    * execution time in milliseconds

### 3.2 GLOG_V
Please refer to [glog_v guide](http://rpg.ifi.uzh.ch/docs/glog.html)
* Add printing log messages (e.g. using `LOG(INFO) << “a message”;` in C-API, or `_logger.info(‘a message’)` in python)
    * a release build is enough, rebuilding it is usually faster,
    * use GLOG_v environment variable flag to print detailed information on Paddle’s internals usage and execution (see the guide http://rpg.ifi.uzh.ch/docs/glog.html)
    * use DNNL_VERBOSE (MKLDNN_VERBOSE) environment variable flag to print detailed information on DNNL primitives creation and execution (on the usage of the flag, see above),

## 4 Basic debugging techniques
Use a debugger (e.g. GDB, VS Code)
* interactive, catching exceptions, breakpoints, step by step execution, etc.
* allows for more thorough analysis of what is happening during the execution.
* a build with debugging symbols is required (`Debug` or `RelWithDebInfo`), rebuilding debug version is usually slower,

Some examples are as shown as follows (Tested with GDB 7.6.1)
### 4.1 Inspecting Tensor data
filter is available and initialized Tensor* with shape [64,3,7,7]
```
(gdb) p filter->data<float>()[0]@64*3*7*7
```
output:
```
(gdb) $20 = {0.159251779, -0.159962952, -0.182004049, 0.0299564917, .....}
```
### 4.2. Saving content Tensor to the file using builtin GDB python interpreter
bias is a 1D tensor of shape [64]
```
(gdb) pi open("mylog.txt","w").write(gdb.execute('print bias->data<float>()[0]@64', to_string=True))   
```
Note: GDB is inserting some control symbols that User removes before comparing buffers to each other
### 4.3. Automating debugging with scripting
#### Example 1 : Break into creation code of MKL-DNN conv , but after executing two operators
```
cgdb -x /home/jczaja/myscript.gdb --args  ./paddle/fluid/inference/tests/api/test_analyzer_image_classification --infer_model=/home/jczaja/paddle/build-debug/third_party/inference_demo/resnet50/model --disable_mkldnn_fc=true --gtest_filter=*profile_mkldnn
```
myscript.gdb:
```
start
b naive_executor.cc:58
c
c
c
d
b mkldnn_reuse.h:950
c
```
#### Example 2 : Start catching exceptions starting from NaiveExecutor::Run()
```
gdb -x /home/jczaja/myscript2.gdb --args  ./paddle/fluid/inference/tests/api/test_analyzer_image_classification --infer_model=/home/jczaja/paddle/build-debug/third_party/inference_demo/resnet50/model --disable_mkldnn_fc=true --gtest_filter=*profile_mkldnn
```
myscript2.gdb:
```
start
b paddle::framework::NaiveExecutor::Run()
c
catch throw
c
```
### 4.4 Catching NANs, overflow and underflow
1. Change source code of Paddle:
```
#include <fenv.h>
….
feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT & ~FE_UNDERFLOW & ~FE_OVERFLOW);
....
<Code for which exception will trigger on Floating point computations>
....
fedisableexcept(FE_ALL_EXCEPT & ~FE_INEXACT & ~FE_UNDERFLOW & ~FE_OVERFLOW);
```
2. Build it.
3. Debug. In GDB:
```
(gdb) start
(gdb) catch throw
(gdb) c
```
