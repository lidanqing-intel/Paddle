/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <mkldnn/include/mkldnn_types.h>
#include <memory>
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/conv_op.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/mkldnn_helper.h"
#include "paddle/fluid/platform/variant.h"
#include "paddle/fluid/framework/data_layout_transform.h"

namespace paddle {
namespace operators {
using framework::DataLayout;
using Tensor = framework::Tensor;
using framework::LoDTensor;
using framework::DDim;
using framework::ExecutionContext;
using platform::MKLDNNDeviceContext;
using platform::to_void_cast;
using platform::GetMKLDNNFormat;
using platform::MKLDNNGetDataType;
using mkldnn::memory;
using mkldnn::convolution_forward;
using mkldnn::primitive;
using mkldnn::stream;
using mkldnn::prop_kind;
#ifdef PADDLE_WITH_MKLDNN
using MKLDNNFormat = mkldnn::memory::format;
using MKLDNNDataType = mkldnn::memory::data_type;

static std::vector<int> GetTensorDims(const Tensor* tensor) {
  auto dims = framework::vectorize2int(tensor->dims());
  return dims;
}

static std::vector<int> ComputeWeightsDims(const ExecutionContext& ctx,
                                           const Tensor* weights, int groups,
                                           bool is_conv3d) {
  std::vector<int> weights_tz = GetTensorDims(weights);
  int g = std::max(groups, 1);
  if (g > 1) {
    // convert [out_n, in_n, c, h, w] to [g, out_n/g, in_n, c, h, w]
    // convert [out_n, in_n, h, w] to [g, out_n/g, in_n, h, w]
    weights_tz.push_back(0);
    std::rotate(weights_tz.begin(), weights_tz.end() - 1, weights_tz.end());
    weights_tz[0] = groups;
    weights_tz[1] = weights_tz[1] / groups;
  }
  return weights_tz;
}

template <typename T_in, typename T_w, typename T_out>
class ConvPrimitiveFactory {
 public:
  explicit ConvPrimitiveFactory(const mkldnn::engine& engine) : engine_(engine) {}

  convolution_forward CreateConvPrimitive(
      const LoDTensor* input, const Tensor* weights,
      const Tensor* bias, const std::vector<int>& strides,
      const std::vector<int>& paddings, const int groups, bool is_conv3d, bool fuse_relu,
      bool fuse_brelu, bool fuse_residual_conn, const Tensor* residual_param,
      LoDTensor* output, bool is_test, const ExecutionContext& ctx, bool is_int8) {
    if (conv_prim_) {
      UpdateDataPointers(ctx, output, input, residual_param);
      return *conv_prim_; // This is returned and all reorder are returned
    }

    const T_in* input_data = input->data<T_in>();
    const T_w* weights_data = weights->data<T_w>();

    /**
     * user weights_md and user_src_md
    */ 
    auto weights_tz = ComputeWeightsDims(ctx, weights, groups, is_conv3d);
    auto weights_format = GetWeightsFormat(weights->format, groups, is_conv3d);
    auto user_weights_md = CreateMemDescriptor<T_w>(weights_tz, weights_format);
    auto user_src_md = CreateMemDescriptor<T_in>(input);

    /**
     * following weight_md and src_md and dst_md bias_md_p are only used for constructing conv_prim_desc. No other usages any more 
     */
    weights_format = mkldnn::memory::format::any;
    auto src_tz = GetTensorDims(input);
    auto chosen_memory_format = GetChosenFormat(ctx, src_tz.size(), groups, is_conv3d);
    auto src_md = CreateMemDescriptor<T_in>(src_tz, chosen_memory_format);
    auto weights_md = CreateMemDescriptor<T_w>(weights_tz, weights_format);
    std::vector<int> dst_tz = GetTensorDims(output);
    auto dst_md = CreateMemDescriptor<T_out>(dst_tz, chosen_memory_format);
    std::shared_ptr<memory::desc> bias_md_p;
    if (bias) {
      auto bias_tz = paddle::framework::vectorize2int(bias->dims());
      auto bias_md_p = std::make_shared<memory::desc>(
          platform::MKLDNNMemDesc(bias_tz, platform::MKLDNNGetDataType<T_in>()),
          memory::format::x);
    }
    auto conv_prim_desc = CreateConvPrimDesc(
        src_md, weights_md, bias_md_p, dst_md, strides, paddings, fuse_relu,
        fuse_residual_conn, fuse_brelu, is_test, groups, weights_tz, is_int8);
 
    /**
     * Create raw input_ and weights memory, no reoder yet
     */
    input_ = CreateMemory(user_src_md, to_void_cast<T_in>(input_data));
    weights_ = CreateMemory(user_weights_md, to_void_cast<T_w>(weights_data));

    /**
     * 1. input reorder
     */
    // TODO(lidanqing) here yes there is some problems
    auto dst_desc = CreateMemDescriptor<T_out>(output, memory::format::any);
    conv_prim_ =
        CreateConvPrimitive(*input_, *weights_, dst_desc, conv_prim_desc, bias,
                            residual_param, output, ctx);
    return *conv_prim_;
  }

  mkldnn::memory CreateUserResidualMemory(const Tensor* residual_param) {
    auto residual_data_tz = GetTensorDims(residual_param);
    auto residual_data_type =
        paddle::framework::ToMKLDNNDataType(residual_param->type());
    auto residual_param_data = residual_param->data<T_out>();
    auto user_residual_md = platform::MKLDNNMemDesc(
        residual_data_tz, residual_data_type, residual_param->format());
    auto user_residual_memory_p =
        CreateMemory(user_residual_md, to_void_cast<T_in>(residual_param_data));
    return user_residual_memory_p;
  }

  mkldnn::memory Reorder(const memory& src_mem, const memory& dst_mem) {
    auto reorder = mkldnn::reorder(src_mem, dst_mem);
    stream(stream::kind::eager).submit({reorder}).wait();
    return dst_mem;
  }

  mkldnn::memory Reorder(const memory::desc& src_desc,
                         const memory::desc& dst_desc, const void* src_data) {
    auto src_mem = memory({src_desc, engine_}, const_cast<void*>(src_data));
    auto dst_mem = memory({dst_desc, engine_});
    auto reorder = mkldnn::reorder(src_mem, dst_mem);
    stream(stream::kind::eager).submit({reorder}).wait();
    return dst_mem;
  }

  // // quantize
  // mkldnn::memory Reorder(const memory& src_mem,
  //                        const memory::primitive_desc& dst_pd,
  //                        const std::vector<float>& scale_data) {
  //   mkldnn::memory dst_mem = mkldnn::memory(dst_pd);  // TODO(lidanqing) waiting
  //                                                     // for review.
  //   mkldnn::primitive_attr attributes;
  //   int mask = CreateMask(0, scale_data.size() > 1);
  //   attributes.set_output_scales(mask, scale_data);
  //   auto reorder =
  //       mkldnn::reorder(mkldnn::reorder::primitive_desc(
  //                           src_mem.get_primitive_desc(), dst_pd, attributes),
  //                       src_mem, dst_mem);
  //   stream(stream::kind::eager).submit({reorder}).wait();
  //   return dst_mem;
  // }

  // quantize 
  mkldnn::memory QuantizeReorderMemory(const memory::primitive_desc& dst_mem_pd, const memory& src_mem, const std::vector<float>& scale_data, int mask){

    mkldnn::memory dst_mem = mkldnn::memory(dst_mem_pd);  // Yes here is correct because dst_mem_pd is memory::primitive_desc, not memory::desc
    mkldnn::primitive_attr attri;  // attribute for int8 weights and bias data reorder.
    attri.set_output_scales(mask, scale_data);
    auto reorder =
        mkldnn::reorder(mkldnn::reorder::primitive_desc(
                            src_mem.get_primitive_desc(), dst_mem_pd, attri),
                        src_mem, dst_mem);
    stream(stream::kind::eager).submit({reorder}).wait();
    return dst_mem;
  }

/**
 *   std::shared_ptr<mkldnn::memory> AcquireMemory(
      mkldnn::memory::primitive_desc& mpd,       // NOLINT
      mkldnn::memory::primitive_desc& user_mpd,  // NOLINT
      const std::shared_ptr<mkldnn::memory> user_memory_p,
      const std::string& suffix,
      std::vector<mkldnn::primitive>& pipeline,  // NOLINT
      bool is_persistent = false, bool is_INT8 = false,
      std::vector<float> scale_data = {1.0f}, int mask = 0) {
    // create reorder primitive if the input format is not the preferred one
    auto local_key = key_ + suffix;
    auto key_reorder_p = key_ + suffix + "reorder_p";

    auto target_memory_p =
        std::static_pointer_cast<mkldnn::memory>(dev_ctx_.GetBlob(local_key));
    if (target_memory_p == nullptr) {
      target_memory_p = user_memory_p;
      std::shared_ptr<mkldnn::primitive> reorder_p;
      if (mpd != user_mpd) {
        target_memory_p = std::make_shared<mkldnn::memory>(mpd);
        std::shared_ptr<mkldnn::reorder> reorder_p;
        if (is_INT8) {
          mkldnn::primitive_attr
              attri;  // attribute for int8 weights and bias data reorder.
          attri.set_output_scales(mask, scale_data);

          auto reorder_pd = std::shared_ptr<mkldnn::reorder::primitive_desc>(
              new mkldnn::reorder::primitive_desc(user_mpd, mpd, attri));
          reorder_p = std::shared_ptr<mkldnn::reorder>(new mkldnn::reorder(
              *reorder_pd, *user_memory_p, *target_memory_p));
        } else {
          reorder_p = std::make_shared<mkldnn::reorder>(*user_memory_p,
                                                        *target_memory_p);
        }
        dev_ctx_.SetBlob(key_reorder_p, reorder_p);
        pipeline.push_back(*reorder_p);
      }
      dev_ctx_.SetBlob(local_key, target_memory_p);
    } else if (!is_persistent) {
      // Make reorder if needed
      auto reorder_p = std::static_pointer_cast<mkldnn::reorder>(
          dev_ctx_.GetBlob(key_reorder_p));
      if (reorder_p != nullptr) {
        pipeline.push_back(*reorder_p);
      }
    }
    return target_memory_p;
  }
 */

  template <typename T>
  static mkldnn::memory::desc CreateMemDescriptor(const std::vector<int>& dims,
                                                  memory::format format) {
    return platform::MKLDNNMemDesc(dims, platform::MKLDNNGetDataType<T>(),
                                   format);
  }

  template <typename T>
  static mkldnn::memory::desc CreateMemDescriptor(Tensor* tensor) {
    auto dims = GetTensorDims(tensor);
    auto format = tensor->format();
    return platform::MKLDNNMemDesc(dims, platform::MKLDNNGetDataType<T>(),
                                   format);
  }

  //What role the Create play. creat memory and when update only use pointer. Put reorder and scale into the queue

 private:
  void UpdateDataPointers(const ExecutionContext& ctx, Tensor* out,
                          const Tensor* in, const Tensor* residual_param) {
    input_->set_data_handle(const_cast<T_in*>(in->data<T_in>()));
    if (residual_param) { // TODO residual may really need to create new memory. Otherwise, the output_ actually has changed type, output_ after created is reordered
      output_->set_data_handle(const_cast<T_out*>(residual_param->data<T_out>())); // what if the format is not correct? It seems reoder is needed. TODO(lidanqing) so the residual format must be included too
    } else {
      output_->set_data_handle(out->mutable_data<T_out>(ctx.GetPlace()));
      if (out->format() == memory::format::format_undef) {
        auto output_format = output_->get_primitive_desc().desc().data.format;
        out->set_format((memory::format)output_format);
      }
    }
  }

  inline mkldnn::memory::format GetWeightsFormat(mkldnn::memory::format format,
                                                 int groups, bool is_conv3d) {
    format =
        (groups == 1) ? format : (is_conv3d ? mkldnn::memory::format::goidhw
                                            : mkldnn::memory::format::goihw);
    return format;
  }

  inline mkldnn::memory::format GetChosenFormat(const ExecutionContext& ctx, size_t src_tz_size,
                                                int groups, bool is_conv3d) {
    std::string data_format = ctx.Attr<std::string>("data_format");
    auto chosen_memory_format =
        platform::data_format_to_memory_format(data_format);

    if (chosen_memory_format != mkldnn::memory::format::any && is_conv3d) {
      chosen_memory_format =
          platform::MKLDNNFormatForSize(src_tz_size, chosen_memory_format);
    }
    return chosen_memory_format;
  }

  template <typename T>
  mkldnn::memory CreateMemory(const mkldnn::memory::desc& desc,
                              const Tensor* tensor) {
    return CreateMemory(desc, tensor->data<T>());
  }

  mkldnn::memory CreateMemory(const mkldnn::memory::desc& desc,
                              const void* data) {
    return memory({desc, engine_}, const_cast<void*>(data));
    //reference of mkldnn_reuse:  mem_p = std::make_shared<mkldnn::memory>(
    //      mkldnn::memory::primitive_desc{md, engine_}, ptr);
  }

  // checked
  std::vector<float> ComputeBiasScales(const ExecutionContext& ctx) {
    auto scale_in_data = ctx.Attr<float>("Scale_in");
    auto scale_weights_data = ctx.Attr<std::vector<float>>("Scale_weights");
    int count =
        is_multi_channel
            ? (g > 1 ? (weights_tz)[1] * (weights_tz)[0] : (weights_tz)[0])
            : 1;
    std::vector<float> scale_bias_data(count);

#pragma omp parallel for if (count > 1)
    for (size_t i = 0; i < count ; i++) {
      if (scale_weights_data[i] == 0.0)
        scale_bias_data[i] = 1.0f;
      else
        scale_bias_data[i] = scale_in_data * scale_weights_data[i];
    }
    return scale_bias_data;
  }

  //TODO I dont quantize residual data. Only put residual_data into the output and the postop appendsum will solve this
  // template <typename T>
  // void QuantizeResidual(const ExecutionContext& ctx) {
  //   auto quantized_desc = output_->get_primitive_desc().desc();
  //   quantized_desc.data.data_type =
  //       (mkldnn_data_type_t)platform::MKLDNNGetDataType<T>();
  //   output_ = Reorder(*output_, {quantized_desc, engine_},
  //                     ctx.Attr<std::vector<float>>("Scale_in_eltwise"));
  // }

  mkldnn::memory QuantizeBias(
      const convolution_forward::primitive_desc& conv_prim_desc,
      const ExecutionContext& ctx) {
    auto bias_scales = ComputeBiasScales(ctx);
    int mask_reorder = is_multi_channel ? 1 << 0 : 1; // 0000 0010 : 0000 0001
    
    bias_ = QuantizeReorderMemory(conv_prim_desc.bias_primitive_desc(), *bias_, bias_scales, mask_reorder);
    return bias_;
  }

  //checked
  template <typename T>
  void QuantizeWeights(const convolution_forward::primitive_desc& conv_prim_desc, const ExecutionContext& ctx) {
    // TODO I feel here should be conv_pd_, but Michal's version is as follow: auto dst_quantized_desc = weights_->get_primitive_desc().desc();
    auto scale_weights_data = ctx.Attr<std::vector<float>>("Scale_weights");
    int mask_reorder =
          is_multi_channel ? ((g != 1) ? (1 << 1) + (1 << 0) : 1 << 0) : 0;  // 0000 0011: 0000 0001 : 0000 0000
    /**
     *     /**
     *  CreateMask(0, channel);
     *   int CreateMask(int slice_dimension, bool is_multi_channel_quantizied) {
    return is_multi_channel_quantizied ? 1 << slice_dimension : 0;
  }*/
     
    quantized_desc.data.data_type =
        (mkldnn_data_type_t)platform::MKLDNNGetDataType<T>();
    
    // TODO return value is weights_ or *weights_。 Here above it is not reorder it is actually quantize
    *weights_ = QuantizeReorderMemory(conv_prim_desc.weights_primitive_desc(), *weights_, scale_weights_data, mask_reorder);
  }

  std::vector<float> ComputeOutputShiftScale(const ExecutionContext& ctx, const int groups, std::vector<int>& weights_tz) {
    auto scale_in_data = ctx.Attr<float>("Scale_in");
    auto scale_weights_data = ctx.Attr<std::vector<float>>("Scale_weights");
    auto scale_out_data = ctx.Attr<bool>("force_fp32_output")
                              ? 1.0f
                              : ctx.Attr<float>("Scale_out");

    bool is_multi_channel = scale_weights_data.size() > 1;
    int count =
        is_multi_channel
            ? (groups > 1 ? (weights_tz)[1] * (weights_tz)[0] : (weights_tz)[0])
            : 1;
    std::vector<float> output_shift_scale(count);

#pragma omp parallel for
    for (size_t i = 0; i < count; i++) {
      if (scale_weights_data[i] == 0.0) {
        output_shift_scale[i] = scale_out_data;
      } else {
        output_shift_scale[i] =
            static_cast<float>(static_cast<double>(scale_out_data) /
                               (static_cast<double>(scale_in_data) *
                                static_cast<double>(scale_weights_data[i])));
      }
    }
    return output_shift_scale;
  }

  float ComputeInt8SumScale(const ExecutionContext ctx) {
      
    // To make the output in the scale of scale_out, we need to scale the residual with [scale_out_data / scale_in_eltwise_data] and this should be put directly in the output initialization
    // If residual does not exists. Then sum_scale is 0
    auto scale_out_data = ctx.Attr<bool>("force_fp32_output")
                              ? 1.0f
                              : ctx.Attr<float>("Scale_out");
    auto scale_in_eltwise_data = ctx.Attr<float>("Scale_in_eltwise");
    sum_scale = ctx.Attr<bool>("fuse_residual_connection")
                          ? scale_out_data / scale_in_eltwise_data
                          : 1.0f;
    return sum_scale;
  }

  mkldnn::primitive_attr CreatePostOps(const ExecutionContext& ctx, const int groups, std::vector<int>& weights_tz, bool is_int8) {
    mkldnn::primitive_attr attributes;
    mkldnn::post_ops post_operations;
    auto output_shift_scale = ComputeOutputShiftScale(ctx, groups, weights_tz);

    /**
     *   int CreateMask(int slice_dimension, bool is_multi_channel_quantizied) {
    return is_multi_channel_quantizied ? 1 << slice_dimension : 0;
    }*/
    // int mask = CreateMask(1, output_shift_scale.size() > 1); // 0000 0010: 0000 0000
    // attributes.set_output_scales(mask, output_shift_scale);
    if (output_shift_scale.size() > 0) {
      int mask = output_shift_scale.size() > 1 ? 1 << 1 : 0; // 0000 0010: 0000 0000
      conv_attr.set_output_scales(mask, output_shift_scale);
    }

    // Fusion with Elementwise layer relies on adding a sum post-operation with
    // the scale parameter. It is assumed that when fuse_residual_connection is
    // true, the output tensor contains the data coming from residual
    // connection. The result of this post_op is:
    // Output = scale * Output + Conv_Out.
    if (ctx.Attr<bool>("fuse_residual_conn")) {
      auto sum_scale = is_int8 ?  ComputeInt8SumScale(ctx) : 1.0f;
      post_operations.append_sum(sum_scale); 
    }

    // Fusion with ReLU layer is executed through the PostOps feature. Create a
    // PostOps object and configure it to execute an eltwise relu operation.
    if (ctx.Attr<bool>("fuse_relu")) {
      constexpr float scale = 1.0f;
      constexpr float negative_slope = 0.0f;
      constexpr float placeholder = 1.0f;  // beta
      post_operations.append_eltwise(scale, mkldnn::algorithm::eltwise_relu,
                                     negative_slope, placeholder);
    }

    if (ctx.Attr<bool>("fuse_brelu")) {
      float fuse_brelu_threshold = ctx.Attr<float>("fuse_brelu_threshold");
      constexpr float scale = 1.0f;
      constexpr float placeholder = 0.0f;
      post_operations.append_eltwise(scale,
                                     mkldnn::algorithm::eltwise_bounded_relu,
                                     fuse_brelu_threshold, placeholder);
    }
    attributes.set_post_ops(post_operations);
    return attributes;
  }

  mkldnn::convolution_forward::primitive_desc CreateConvPrimDesc(
      const ExecutionContext& ctx, const mkldnn::memory::desc& input_desc,
      const mkldnn::memory::desc& weights_desc,
      boost::optional<const mkldnn::memory::desc&> bias_desc,
      const mkldnn::memory::desc& dst_desc, const std::vector<int>& strides,
      const std::vector<int>& paddings, const bool fuse_relu,
      const bool fuse_residual_conn, const bool fuse_brelu,
      const bool is_test, const int groups, std::vector<int> weights_tz, bool is_int8) {
    const auto attrs = CreatePostOps(ctx, groups, weights_tz, is_int8);
    static std::mutex acquire_barrier;
    std::lock_guard<std::mutex> block_threads_until_finish_this_job(
        acquire_barrier);

    auto fwd_prop_kind = is_test ? mkldnn::prop_kind::forward_inference
                                 : mkldnn::prop_kind::forward_training;
    mkldnn::memory::dims stride_dims = strides;
    mkldnn::memory::dims padding_dims = paddings;
    auto conv_desc =
        (bias_desc != nullptr)
            ? convolution_forward::desc(
                  fwd_prop_kind, mkldnn::algorithm::convolution_direct,
                  input_desc, weights_desc, *bias_desc, dst_desc, stride_dims,
                  padding_dims, mkldnn::padding_kind::zero)
            : convolution_forward::desc(
                  fwd_prop_kind, mkldnn::algorithm::convolution_direct,
                  input_desc, weights_desc, dst_desc, stride_dims, padding_dims,
                  mkldnn::padding_kind::zero);
    auto conv_prim_desc =
        convolution_forward::primitive_desc(conv_desc, attrs, engine_);
    return conv_prim_desc;
  }

  convolution_forward CreateConvPrimitive(
      const memory& user_src_memory, const memory& user_weights_memory,
      const memory::desc& dst_desc,
      const mkldnn::convolution_forward::primitive_desc& conv_prim_desc,
      const Tensor* bias, const Tensor* residual_param, const Tensor* output,
      const ExecutionContext& ctx) {
    // all reorders should be put here because the conv_prim returned here will be put into the submit, and all others follow this
    input_ = Reorder(user_src_memory.get_primitive_desc(),
                     conv_prim_desc.src_primitive_desc(), user_src_memory);
    // TODO(lidanqing) Should weights reorder from the begining before quantization? It seems for INT8 it is not. According to liaoli's code, it is like this
    if (!is_int8){
      weights_ = Reorder(user_weights_memory.get_primitive_desc(),  conv_prim_desc.weights_primitive_desc(), user_weights_memory);
    }
    else{
      QuantizeWeights(conv_prim_desc, ctx);
    }
    
    CreateDstMemory(conv_prim_desc, ctx, residual_param);
    // until here original untill 544

    if (bias) {
      auto bias_desc = CreateMemDescriptor<T_w>(bias);
      bias_ = CreateMemory<T_w>(bias_desc, bias);
      if(is_int8){
        bias_ = QuantizeBias(conv_prim_desc, ctx);
      }
      return convolution_forward(conv_prim_desc, input_, weights_,
                                 bias_, output_);
    } else {
      return convolution_forward(conv_prim_desc, input_, weights_,
                                 output_);
    }
    // push primitive to stream and wait until it's executed
    //  pipeline.push_back(*conv_p);
    //Creating end. The ptr==null conv_prim doesn not exist situation end here
  }

  // This is still the most important
  // if residual_param exists, set pointer of residual_data memory as output_.
  // If output->dst format != residua format, reorder will be executed. If
  // INT8 is applied, sum_scale is executed too.
  mkldnn::memory CreateDstMemory(
      const mkldnn::convolution_forward::primitive_desc& conv_prim_desc,
      const ExecutionContext& ctx, Tensor* residual_param) {
    auto fetched_dst_format =
        conv_prim_desc.dst_primitive_desc().desc().data.format;
    auto fetched_dst_memory_size =
        conv_prim_desc.dst_primitive_desc().get_size();

    if (residual_param) {
      output_ = CreateUserResidualMemory(residual_param);
      if (residual_param->format() != fetched_dst_format) {
        auto residual_dt = platform::MKLDNNGetDataType<T_out>();
        auto residual_data_tz = GetTensorDims(residual_param);
        auto user_residual_md = platform::MKLDNNMemDesc(
               residual_data_tz, residual_dt, residual_param->format());
        output_ = Reorder( //TODO in liaoli's code this part looks does not need to do reorder. But there is pipe, so strange
            conv_prim_desc.dst_primitive_desc(), output_);
      }
    } else{  // create memory
      auto output_data =
          output->mutable_data<T_out>(ctx.GetPlace(), fetched_dst_memory_size);
      auto output_ = std::make_shared<mkldnn::memory>(
          conv_prim_desc.dst_primitive_desc(), output_data);
    }
    return output_;
  }

  template <typename T>
  mkldnn::memory SetDstMemory(
      const framework::ExecutionContext& ctx, framework::Tensor* output,
      const std::shared_ptr<ConvMKLDNNHandler>& handler) {
    T* output_data =
        output->mutable_data<T>(ctx.GetPlace(), handler->GetDstMemorySize());
    mkldnn::memory dst_memory_p =
        handler->AcquireDstMemoryFromPrimitive(to_void_cast<T>(output_data));
    return dst_memory_p;
  }

 private:
  const mkldnn::engine& engine_;
  boost::optional<memory> bias_;
  boost::optional<memory> input_;
  boost::optional<memory> output_;
  boost::optional<memory> weights_;
  boost::optional<mkldnn::convolution_forward> conv_prim_;
};

template <typename T_in, typename T_w, typename T_out>
static std::shared_ptr<ConvPrimitiveFactory<T_in, T_w, T_out>> GetConvPrimitiveFactory(
    const MKLDNNDeviceContext& dev_ctx, const std::string key,
    const mkldnn::engine& mkldnn_engine) {
  auto prim_creator =
      std::static_pointer_cast<ConvPrimitiveFactory<T_in, T_w, T_out>>(
          dev_ctx.GetBlob(key));
  if (prim_creator == nullptr) {
    prim_creator =
        std::make_shared<ConvPrimitiveFactory<T_in, T_w, T_out>>(mkldnn_engine);
    dev_ctx.SetBlob(key, prim_creator);
  }
  return prim_creator;
}

// dst_dt =
// paddle::framework::ToMKLDNNDataType(framework::DataTypeTrait<float>::DataType);
static MKLDNNDataType getDstType(bool is_int8, bool force_fp32_output, bool fuse_relu, bool fuse_brelu,
                                       bool fuse_residual_conn,
                                       const Tensor* residual_param) {
  auto dst_dt = MKLDNNDataType::f32;  // uint8_t, int8_t, float
  if (is_int8) {
    auto dst_dt = (fuse_relu || fuse_brelu) ? MKLDNNDataType::u8 : MKLDNNDataType::s8;

    if (force_fp32_output) {
      dst_dt = MKLDNNDataType::f32;
    }
    if (fuse_residual_conn && residual_param) {
      auto residual_dt = framework::ToMKLDNNDataType(residual_param->type());
      if (dst_dt != residual_dt) dst_dt = residual_dt;
    }
  }
  return dst_dt;
}
static std::string GetHash(const mkldnn::memory::dims& input_dims,  // NOLINT
                           const mkldnn::memory::data_type src_dt,
                           const mkldnn::memory::format& format,
                           const mkldnn::memory::dims& weights_dims,  // NOLINT
                           const bool& fuse_relu,                     // NOLINT
                           const bool& fuse_brelu,                    // NOLINT
                           const bool& fuse_residual_conn,
                           std::vector<int>& strides,    // NOLINT
                           std::vector<int>& paddings,   // NOLINT
                           std::vector<int>& dilations,  // NOLINT
                           int groups, const std::string& suffix) {
  auto dims2str = [](const mkldnn::memory::dims& operand_dims) {
    std::string str = "";
    for (size_t i = 0; i < operand_dims.size(); ++i) {
      str += std::to_string(operand_dims[i]) + "-";
    }
    return str;
  };

  auto vec2str = [](const std::vector<int>& vec) {
    std::string str = "";
    for (size_t i = 0; i < vec.size(); ++i) {
      str += std::to_string(vec[i]) + "-";
    }
    return str;
  };

  return dims2str(input_dims) + std::to_string(src_dt) +
         std::to_string(format) + 
         dims2str(weights_dims) + std::to_string(fuse_relu) +
         std::to_string(fuse_brelu) + std::to_string(fuse_residual_conn) +
         vec2str(strides) + vec2str(paddings) + vec2str(dilations) +
         std::to_string(groups) + suffix;
}

template <typename T_in, typename T_w>
class ConvMKLDNNOpKernel : public framework::OpKernel<T_in> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_cpu_place(ctx.GetPlace()),
                   "It must use CPUPlace.");
    auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    auto& mkldnn_engine = dev_ctx.GetEngine();

    auto is_test = ctx.Attr<bool>("is_test");

    auto input = ctx.Input<LoDTensor>("Input");
    auto weights = ctx.Input<Tensor>("Filter");
    auto bias = ctx.Input<Tensor>("Bias");
    auto output = ctx.Output<LoDTensor>("Out");

    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");
    int groups = ctx.Attr<int>("groups");
    bool fuse_brelu = false;
    float fuse_brelu_threshold = 6.0;
    bool fuse_relu = ctx.Attr<bool>("fuse_relu");
    bool fuse_residual_conn = ctx.Attr<bool>("fuse_residual_connection");
    bool force_fp32_output = ctx.Attr<bool>("force_fp32_output");
    auto residual_param = ctx.Input<Tensor>("ResidualData");

    bool is_conv3d = strides.size() == 3U;
    if (!is_conv3d) {
      fuse_brelu = ctx.Attr<bool>("fuse_brelu");
      fuse_brelu_threshold = ctx.Attr<float>("fuse_brelu_threshold");
    }

    PADDLE_ENFORCE(
        is_conv3d
            ? dilations.size() == 3 && dilations[0] == 1 && dilations[1] == 1 &&
                  dilations[2] == 1
            : dilations.size() == 2 && dilations[0] == 1 && dilations[1] == 1,
        "dilation in convolution is not implemented yet");

    if (fuse_residual_conn) {
      PADDLE_ENFORCE(
          residual_param != nullptr,
          "Provide data if you want MKLDNN conv+elementwise_add fusion");
      PADDLE_ENFORCE_EQ(output->dims(), residual_param->dims(),
                        "Output and elementwise parameter need to have the "
                        "same dimension sizes");
    }
    constexpr bool is_int8 =
        std::is_same<T_in, int8_t>::value || std::is_same<T_in, uint8_t>::value;
    
    if ( is_int8 ){
      PADDLE_ENFORCE(
          is_conv3d == false,
          "is_conv3d is not enabled when is_int8 is true");
    }
    auto src_dt = platform::MKLDNNGetDataType<T_in>();
    auto src_format = input->format();

    auto src_tz = GetTensorDims(input);
    auto weights_tz = ComputeWeightsDims(ctx, weights, groups, is_conv3d);
    std::string key = GetHash(
        src_tz, src_dt, src_format, weights_tz, fuse_relu, fuse_brelu,
        fuse_residual_conn, strides, paddings, dilations, groups,
        ctx.op().Input("Input") + ctx.op().Input("Filter"));

    auto dst_typename = getDstType(is_int8, force_fp32_output, fuse_relu, fuse_brelu, fuse_residual_conn, residual_param);
 
    //std::shared_ptr<mkldnn::convolution_forward> conv_p;
    if (dst_typename == MKLDNNDataType::f32){
      auto conv = GetConvPrimitiveFactory<T_in, T_w, float>(dev_ctx, key,
                                                    mkldnn_engine)
            ->CreateConvPrimitive(input, weights, bias, strides, paddings,
                                  groups, is_conv3d, fuse_relu, fuse_brelu,
                                  fuse_residual_conn, residual_param, output, is_test, ctx, is_int8);
      
    
    }
    else if (dst_typename == MKLDNNDataType::u8){
      auto  conv = GetConvPrimitiveFactory<T_in, T_w, uint8_t>(dev_ctx, key,
                                                    mkldnn_engine)
            ->CreateConvPrimitive(input, weights, bias, strides, paddings,
                                  groups, is_conv3d, fuse_relu, fuse_brelu,
                                  fuse_residual_conn, residual_param, output, is_test, ctx, is_int8);
    }
    else if(dst_typename == MKLDNNDataType::s8){
      auto conv = GetConvPrimitiveFactory<T_in, T_w, int8_t>(dev_ctx, key,
                                                    mkldnn_engine)
            ->CreateConvPrimitive(input, weights, bias, strides, paddings,
                                  groups, is_conv3d, fuse_relu, fuse_brelu,
                                  fuse_residual_conn, residual_param, output, is_test, ctx, is_int8);
      
    }
    stream(stream::kind::eager).submit({conv}).wait();
    output->set_layout(DataLayout::kMKLDNN);
  }
};
}  // namespace operators
}  // namespace paddle

/**
 * T_w is the type of weights and bias. In convolution they are always set as float
 * T_in is the type of input
 * dst output type 
 * residual_param will be decided during running
 */
namespace ops = paddle::operators;
REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(conv2d, MKLDNN, ::paddle::platform::CPUPlace,
                                    FP32, ops::kConvMKLDNNFP32,
                                    ops::ConvMKLDNNOpKernel<float, float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(conv2d, MKLDNN, ::paddle::platform::CPUPlace,
                                    U8, ops::kConvMKLDNNINT8,
                                    ops::ConvMKLDNNOpKernel<uint8_t, float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(conv2d, MKLDNN, ::paddle::platform::CPUPlace,
                                    S8, ops::kConvMKLDNNINT8,
                                    ops::ConvMKLDNNOpKernel<int8_t, float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(conv3d, MKLDNN, ::paddle::platform::CPUPlace, 
                                    FP32, ops::kConvMKLDNNFP32,
                                    ops::ConvMKLDNNOpKernel<float, float>);

// REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(conv2d_grad, MKLDNN,
//                                     ::paddle::platform::CPUPlace, FP32,
//                                     ops::kConvMKLDNNFP32,
//                                     ops::ConvMKLDNNGradOpKernel<float>);

// REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(conv3d_grad, MKLDNN,
//                                     ::paddle::platform::CPUPlace, FP32,
//                                     ops::kConvMKLDNNFP32,
//                                     ops::ConvMKLDNNGradOpKernel<float>);
