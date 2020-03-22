# 调查PaddlePaddle推理中的精度差别问题

## 0. 最常出现精度问题的情况
1. 常规测试中的精度差。
2. 引入新的硬件。
3. 开发新功能（例如，添加新的运算符内核）。
4. 使用不同的环境，语言，数据加载进行测试
    * 例如 python与C-API，使用PIL进行python训练，使用OpenCV进行C++推理
    * 例如 在GPU上进行训练，在CPU上进行推理，并在GPU和CPU上使用不同的内核实现。 用例：在处理`batch_norm`时，在GPU和本机CPU上的`sqrt`函数实现的精度较低，而MKL-DNN实现的精度更高。
## 1. 通用精度问题调查指南
* 是否使用完整的验证数据集？在较小的数据集上进行测试通常会得出不同且不准确的结果。
* 是否正确计算了精度？
    * 用例：[#18913 Accuracy issue with int8_mobilenet_ssd tester](https://github.com/PaddlePaddle/Paddle/issues/18913) 精度计算算法中的错误。
* 问题出现在另一台机器上吗？如果不出现在另一台机器上：
    * 确保在两台计算机上测试了相同的修订版本，并且测试前清除了Build目录等
    * 清理build目录，
    * 比较硬件之间的差异
        * 比较CPU对指令集体系结构（ISA）的支持（是否支持AVX2或AVX512？），
        * 某个运算核应该使用哪种ISA？ （例如，如果使用基于DNNL的conv2d或fc,通常要用AVX512）
        * 检查特定的运算核是否使用了正确的ISA
            * 使用GLOG找出本机操作员使用的ISA，
            * 使用DNNL_VERBOSE来查找基于DNNL的运营商使用的ISA，
        * 检查AVX512和AVX2在最终输出中有什么区别
    * 比较环境配置之间的差异
        * 虚拟机/ docker映像是否用于测试？
            * 他们可能缺乏对CPU功能的支持（例如AVX512支持，在VM / docker内部和外部运行`lscpu`命令并比较支持的功能）
        * 较旧的Linux内核（3.15之前的版本）可能缺少适当的ISA支持，
        * PaddlePaddle依赖的库可能会发生故障（例如，在升级系统库之后），
    * 用例：
        * [#18005 test_analyzer_resnet50 fail on E5-2660 v4](https://github.com/PaddlePaddle/Paddle/issues/18005) Paddle-使用的AVX2和AVX512 ISA的结果之间的差异+非常小的精度差异在大量数据集上累积
        * [#18658 MKL-DNN Failure to run face model (demark)](https://github.com/PaddlePaddle/Paddle/issues/18658)  PaddlePaddle / Paddle-从具有AVX2的机器切换到具有AVX512的机器后，基于DNNL的代码选择了NCHW16C格式（而不是AVX2的NCHW） tanh运算符内核，然后以同样NCHW16C格式输出，传递给无法正确处理NCHW16C格式的fetch运算符，
        * [#19505 test_slim_int8_googlenet，test_analyzer_int8_googlenet fail on 5117](https://github.com/PaddlePaddle/Paddle/issues/19505) PaddlePaddle / Paddle-由提交+配置错误的计算机引入的回归（在使用的CPU中缺少ISA支持）+ AVX2和AVX512结果之间的差异+缺少相对误差度量大数据
* 该问题是否存在于发行版中，而不是在调试版中（或者反之）？
    * 在发行版中使用打印调试消息，
    * 如果RelWithDebInfo版本重现此问题，请尝试使用调试器（有关调试提示，请参见下文）。
* 是否以某些方式对模型进行了优化？如是：
    * 关闭优化（例如passes）；如果有帮助，通过一个一个打开pass，找到那个问题pass。
    * 以这种方式发现bug的案例：
        * runtime_context_cache_pass破坏了ResNet50和MobileNet-v1的FP32 C-API推断的准确性·问题＃16609·PaddlePaddle / Paddle，
        * `fc_gru_fuse_pass`破坏了C-API CRNN-CTC推断的准确性。 ·问题＃14837·PaddlePaddle / Paddle。
* 是否在提交历史的某一个commit造成了精度下降？
    * 查找引入了精度下降的commit，然后分析这个commit。
* 对于不同的执行模式（例如MKL-DNN与native，python与C-API），比较检查是否失败？
    * 找出哪一层会在输出上造成差异（例如，从模型的顶部开始并沿着树向下移动，比较输入或输出变量的张量）
        * 检查差分输出的大小（参考版本与当下版本），
        * 确保使用张量之间的差异的相对度量进行比较即可，即，如果张量的值达到10000，则其差异必须满足[-0.001，0.001]范围的要求可能无法实现，
    * 找出该问题层（layer）使用了什么内核（例如，在内核的Compute（）方法或GetExpectedKernelType（）方法中添加调试消息，
    * 关闭所有运算符的MKL-DNN内核，然后逐一启用它们，
        * 在mkldnn_placement_pass中，通过设置pass的`mkldnn_enabled_op_types`属性，例如使用AnalysisConfig类的方法`SetMKLDNNOp（）`
        * 或从DNNL内部禁用特定的实现（请参见下文）。
    * 用例：
        * [#17535 conv_transpose_mkldnn accuracy diff](https://github.com/PaddlePaddle/Paddle/issues/18658)·PaddlePaddle / Paddle-问题中的注释显示了以下步骤
* 特定的operator是罪魁祸首吗？如是：
    * 修改单元测试以覆盖模型中的案例，并在持续失败的情况下对其进行简化，直到获得最简单的案例，易于手动分析，
    * 编写一个简单的程序（可以是单元测试），该程序将有缺陷的运算符内核应用于非常简单的输入数据，以便您甚至可以在一张纸上手动地轻松验证计算的正确性，
        * 使用尺寸非常小的输入张量，
        * 使用模式化的数据初始化输入张量（例如，使用升序整数，编码矩阵的行和列，例如[[11，12]，[21，22]]），
    * 添加具有各种数据模式的多个测试用例，特定的失败用例可能会提示计算中的错误所在。
* 输入张量的大小是多少？
    * 如果数据的大小接近数据类型的最大值，则计算过程中可能会损失精度；请确保您了解IEEE浮点格式，以及为何对大小差异较大的数执行二元运算容易损失精度。
* 单线程和多线程运行的精度是否有所不同？
    * 浮点计算精度可以取决于二元运算操作数的前后顺序及其大小。
* 使用GLOG_v和/或DNNL_VERBOSE环境变量启用打印日志：
    * 寻找任何异常消息，
    * 在python输出中寻找 nan / infs警告，打开编译器数字异常以查看出现问题的位置（有关如何执行此操作的提示，请参见下文）。
* INT8精度是否出问题（零或比预期低）而同时FP32精度良好？如果是：
    * 检查CPU是否支持AVX512 VNNI（通常，VNNI指令在正确性方面能更好地处理INT8操作），
    * 关闭INT8模型优化（例如，关闭`mkldnn_quantize_squash_pass`中的各种压缩程序）；如果有帮助，请一一打开，找到有问题的主要pass，
    * 禁用所有运算符的量化,然后再一一启用
        * 您可以使用`MkldnnQuantizerConfig`类的`SetEnabledOpTypes()`方法来仅对选定的运算符启用量化。
    * 检查要量化的张量中存储的FP32数据的分布（数据量化需要混叠，这取决于张量数据的分布，可能会对精度产生重大影响，例如，如果张量中90％的值是在[0，10]的范围内，并且5％落在[1000，+]范围内，所有较小的值将被量化为0或1，从而丢失了有关其可变性的信息），
        * 确保用于张量量化的标度与张量中的数据分布相匹配（例如，如果张量的值沿[-1.0，1.0]范围扩展，则使用标度0.125量化是没有意义的，可能接近127会更合适，因为量化会使张量数据沿着[-128，127] 8位整数的整个范围扩展）。
    * 如果INT8和FP32内核共享代码：
        * 确保正确处理了各种输入数据格式（对于FP32通常为NCHW，对于INT8为NHWC）（例如，这是Ernie模型中FC INT8内核准确性下降的原因：DNNL内部产品原语无法处理NWC格式） FC INT8内核）
        * 检查权重是否具有适合输入格式的格式，（有关DNNL数据格式，请参见DNNL：了解内存格式）
        * 检查是否必须对输入/输出数据进行无符号化（例如在ReLU op之后）以及量化/去量化运算符是否正确地使用了该信息（当数据无符号化时经常使用量化来对有符号的INT8进行量化会降低准确性）
    * 另外：
        * 分析INT8内核计算算法，
    * 如果应用INT8v2量化（训练后量化）：
        * 确保使用所需算法计算比例因子（通常是KL用于输入/输出，权重为每通道MAX），
        * 准确性在很大程度上取决于预热批次的大小和预热数据（在预热阶段用于生成量化因子以进行量化的数据），
        * 修改预热批处理的大小（通常50个就足够了，某些模型可能会以较低的值（例如10，视情况而定）提供最佳的精度），
        * 选择另一组输入数据进行预热，
    * 如果应用了QATv2程序并且QAT FP32模型达到了良好的准确性：
        * 使用`save_qat_model`测试保存FP32和INT8模型，
        * 比较您使用FP32和INT8获得的精度，
        * 从第一个量化运算符开始，比较其QAT FP32和INT8模型之间的输入和输出张量-它们应包含非常相似的数字，如QAT中的浮点数和INT8模型中的整数，
        * 对于具有权重的运算符（例如conv2d，fc），比较QAT和INT8模型之间的权重张量；它们的值应该非常相似。
## 2. 在DNNL中禁用选定的运算符实现
当我们发现是特定的DNNL运算符存在问题时，可以修改DNNL以删除该运算符的某些实现（例如AVX512）。下面以卷积为例进行操作。
1. 禁用DNNL的某些特定实现
    * 使用DNNL_VERBOSE查看调用哪个实现
    * Folk mkl-dnn并修改cpu_engine.cpp，但是删除您觉得是错误的实现：
    ```
    INSTANCE(jit_avx512_common_convolution_fwd_t<f32>),
    INSTANCE(jit_avx512_common_convolution_bwd_data_t<f32>),
    // INSTANCE(jit_avx512_common_convolution_bwd_weights_t<f32>),   
    INSTANCE(jit_avx2_dw_convolution_fwd_t),
    INSTANCE(jit_avx2_dw_convolution_bwd_data_t),
    ```
2. 使您的mkl-dnn可用于PaddlePaddle(修改mkldnn.cmake)。
3. 禁用DNNL中指令集。修改cpu_isa_traits.hpp文件，将true替换为false以禁用某些平台：
   cpu_isa_traits.hpp：
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
## 3. 基本日志记录
### 3.1 DNNL_VERBOSE说明
（指南：[DNNL：详细文档](https://intel.github.io/mkl-dnn/dev_guide_verbose.html)
将DNNL_VERBOSE环境变量设置为选定级别（0、1或2）后，DNNL库将日志记录信息添加到标准输出中。提供以下信息：
* 第一行：DNNL库版本和git hash，
* 第二行：受支持的ISA和扩展，
* 后续行如下
    ```
    dnnl_verbose，exec，cpu，convolution，jit：avx2，forward_training，src_f32 :: blocked：aBcd8b：f0 wei_f32 :: blocked：ABcd8b8a：f0 bia_f32 :: blocked：a：f0 dst_f32 :: blocked：aBcdgb： ，mb2_ic16oc16_ih7oh7kh5sh1dh0ph2_iw7ow7kw5sw1dw0pw2,0.026123
    ```
* 注释：
    * dnnl_verbose标记字符串
    * 操作：create [：cache_hit]，create [：cache_miss]或exec
    * engine类型：cpu或gpu
    * kernel名称：convolution, reorder, sum, 等
    * kernel实现
    * propagation：forward_training，forward_inference或backward
    * 有关输入和输出数据类型和格式的信息
    * 辅助信息，例如算法名称或输入数量
    * Benchednn格式
    * 执行时间（以毫秒为单位）

### 3.2 GLOG_V
请参考[glog_v指南](http://rpg.ifi.uzh.ch/docs/glog.html)
* 添加打印日志消息（例如，在C-API中使用`LOG(INFO) << “a message”;`，在python中使用`_logger.info(‘a message’)`）
    * release版本就足够了，不需要debug版，重建通常更快
    * 使用GLOG_v环境变量标志来打印有关Paddle内部使用和执行的详细信息（请参阅指南http://rpg.ifi.uzh.ch/docs/glog.html）
    * 使用DNNL_VERBOSE（MKLDNN_VERBOSE）环境变量标志来打印有关DNNL基元创建和执行的详细信息（有关标志的用法，请参见上文）

## 4 基本调试技巧
使用调试器（例如GDB，VS-code）特点
* 交互式，捕获异常，设置断点，逐步执行等。
* 允许对执行期间发生的事情进行更彻底的分析。
* 需要带有调试符号的构建（`Debug`或`RelWithDebInfo`），重建调试版本通常较慢，

一些示例如下所示（已在GDB 7.6.1上进行了测试）
### 4.1 检查Tensor数据
* filter用初始化了形状为[64,3,7,7]的Tensor
    ```
    (gdb) p filter->data<float>()[0]@64*3*7*7
    ```
    输出：
    ```
    (gdb) $20 = {0.159251779, -0.159962952, -0.182004049, 0.0299564917, .....}
    ```
### 4.2 使用内置的GDB python解释器将内容Tensor保存到文件中
* bias是形状的一维张量[64]
    ```
    (gdb) pi open("mylog.txt","w").write(gdb.execute('print bias->data<float>()[0]@64', to_string=True))   
    ```
    注意：GDB会插入一些控制符号，然后再相互比较缓冲区
### 4.3 输出任何你想要的格式
* 您可能想以十六进制打印数字或以十进制打印指针。或者您可能希望以任何数据类型查看内存中某个地址处的数据。比如以INT8格式输出某段地址处数据。为此，请在打印值时指定输出格式。
可以参考[参考文档](https://sourceware.org/gdb/onlinedocs/gdb/Output-Formats.html)
    ```
    (gdb) set print pretty
    (gdb) p *output
    (gdb) p/d (int8_t*)(output->holder_)@5*128*768
    ```
### 4.4 使用脚本自动化调试
1. 示例1：在执行两个conv运算符之后,打断点进入MKL-DNN conv的代码
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
2. 示例2：从NaiveExecutor :: Run（）开始捕获异常
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
### 4.5 捕获NAN，上溢和下溢
1. 更改Paddle的源代码：
    ```
    #include <fenv.h>
    ….
    feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT & ~FE_UNDERFLOW & ~FE_OVERFLOW);
    ....
    <Code for which exception will trigger on Floating point computations>
    ....
    fedisableexcept(FE_ALL_EXCEPT & ~FE_INEXACT & ~FE_UNDERFLOW & ~FE_OVERFLOW);
    ```
2. 构建编译。
3. 调试。在GDB中：
    ```
    (gdb) start
    (gdb) catch throw
    (gdb) c
    ```
