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
#include "boost/optional.hpp"
#include "paddle/fluid/framework/data_layout_transform.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/operators/conv_op.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/mkldnn_helper.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"
#include "paddle/fluid/platform/variant.h"

namespace paddle {
namespace operators {
using framework::DataLayout;
using framework::Tensor;
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

static std::vector<int> GetWeightsTz(const ExecutionContext& ctx,
                                     const Tensor* weights, int groups,
                                     bool is_conv3d) {
  std::vector<int> weights_tz = framework::vectorize2int(weights->dims());
  int g = std::max(groups, 1);
  if (g > 1) {  
    // is_conv3d = true, [out_n, in_n, c, h, w] to [g, out_n/g, in_n, c, h, w]
    // is_conv3d = false, [out_n, in_n, h, w] to [g, out_n/g, in_n, h, w]
    weights_tz.push_back(0);
    std::rotate(weights_tz.begin(), weights_tz.end() - 1, weights_tz.end());
    weights_tz[0] = groups;
    weights_tz[1] = weights_tz[1] / groups;
  }
  return weights_tz;
}
// fp32 format by default mkdnn_oihw. INT8 format by default mkldnn_nchw. throw
// error
static mkldnn::memory::format GetWeightsFormat(memory::format format,
                                               int groups, bool is_conv3d,
                                               bool is_int8 = false) {
  if (!is_int8)
    return (groups == 1) ? format : (is_conv3d ? memory::format::goidhw
                                               : memory::format::goihw);
  else
    return (groups == 1) ? mkldnn::memory::format::oihw
                         : mkldnn::memory::format::goihw;
}

template <typename T_in, typename T_w, typename T_out>
class ConvPrimitiveFactory {
 public:
  explicit ConvPrimitiveFactory(const mkldnn::engine& engine)
      : engine_(engine) {}

  convolution_forward AcquireConvPrimitive(
      const LoDTensor* input, const Tensor* weights, const Tensor* bias,
      const std::vector<int>& strides, const std::vector<int>& paddings,
      const int groups, bool is_conv3d, bool fuse_relu, bool fuse_brelu,
      const float fuse_brelu_threshold, bool fuse_residual_conn,
      const Tensor* residual_param, LoDTensor* output, bool is_test,
      const ExecutionContext& ctx, bool is_int8) {
    if (conv_prim_) {
      UpdateDataPointers(ctx, output, input, residual_param);
      if (is_int8 && fuse_residual_conn && (fuse_relu || fuse_brelu) &&
          platform::MKLDNNGetDataType<T_out>() == memory::data_type::s8) {
        output->mutable_data<uint8_t>(ctx.GetPlace());
      }
      return *conv_prim_;
    }

    auto weights_tz = GetWeightsTz(ctx, weights, groups, is_conv3d);
    auto weights_format =
        GetWeightsFormat(weights->format(), groups, is_conv3d, is_int8);
    auto user_weights_md =
        CreateMemDescriptor<T_w>({weights_tz}, weights_format);
    auto user_src_md = CreateMemDescriptor<T_in>(input, input->format());

    // weight_md_p, src_md, dst_md and bias_md_p to construct conv_prim_desc_
    auto weights_format_any = memory::format::any;
    auto src_tz = framework::vectorize2int(input->dims());
    auto chosen_memory_format =
        GetChosenFormat(ctx, src_tz.size(), groups, is_conv3d, is_int8);
    auto src_md = CreateMemDescriptor<T_in>({src_tz}, chosen_memory_format);
    auto weights_dt =
        is_int8 ? memory::data_type::s8 : platform::MKLDNNGetDataType<T_w>();
    auto weights_md = is_int8 ? platform::MKLDNNMemDesc(weights_tz, weights_dt,
                                                        chosen_memory_format)
                              : platform::MKLDNNMemDesc(weights_tz, weights_dt,
                                                        weights_format_any);
    std::shared_ptr<mkldnn::memory::desc> bias_md_p;
    if (bias) {
      auto bias_dt =
          is_int8 ? memory::data_type::s32 : platform::MKLDNNGetDataType<T_w>();
      auto bias_tz = paddle::framework::vectorize2int(bias->dims());
      bias_md_p = std::make_shared<mkldnn::memory::desc>(
          platform::MKLDNNMemDesc(bias_tz, bias_dt, mkldnn::memory::format::x));
    }
    std::vector<int> dst_tz = paddle::framework::vectorize2int(output->dims());
    auto dst_md = CreateMemDescriptor<T_out>({dst_tz}, chosen_memory_format);
    conv_prim_desc_ = CreateConvPrimDesc(
        ctx, src_md, weights_md, bias_md_p, dst_md, strides, paddings,
        fuse_relu, fuse_residual_conn, fuse_brelu, fuse_brelu_threshold,
        is_test, groups, weights_tz, is_int8);

    user_src_p = CreateMemory(user_src_md, input->data<T_in>());
    input_p = AcquireMemory(conv_prim_desc_->src_primitive_desc(),
                           user_src_p->get_primitive_desc(), user_src_p);
    user_weights_p =
        CreateMemory(user_weights_md,
                     weights->data<T_w>());  // to_void_cast<T_w>(weights_pdata)
    weights_p = ReorderQuantizeWeights(
        ctx, groups, is_int8);  // is_persistent = true, is_test = true
    CreateDstMemory(ctx, residual_param, output);

    if (bias) {
      auto user_bias_md =
          CreateMemDescriptor<T_w>(bias, mkldnn::memory::format::x);
      user_bias_p = CreateMemory(user_bias_md, bias->data<T_w>());
      bias_ = QuantizeBias(ctx, groups, weights_tz, is_int8);
      conv_prim_ = convolution_forward(*conv_prim_desc_, *input_, *weights_p,
                                       *bias_, *output_);
    } else {
      conv_prim_ =
          convolution_forward(*conv_prim_desc_, *input_p, *weights_p, *output_);
    }

    if (is_int8 && fuse_residual_conn && (fuse_relu || fuse_brelu) &&
        platform::MKLDNNGetDataType<T_out>() == memory::data_type::s8) {
      output->mutable_data<uint8_t>(ctx.GetPlace());
    }
    output->set_layout(DataLayout::kMKLDNN);
    output->set_format(GetMKLDNNFormat(*output_));
    return *conv_prim_;
  }

  std::shared_ptr<mkldnn::memory> AcquireMemory(
      const mkldnn::memory::primitive_desc& mpd,       // NOLINT
      const mkldnn::memory::primitive_desc& user_mpd,  // NOLINT
      const shared_ptr<mkldnn::memory>& user_memory, std::vector<float> scale_data = {1.0f},
      int mask = 0, bool is_INT8 = false) {
    // create reorder primitive if the input format is not the preferred one
    mkldnn::memory dst_mem_p = std::make_shared<mkldnn::memory>(mpd);
    if (mpd != user_mpd) {
      if (is_INT8) {
        mkldnn::primitive_attr attri;  // attribute for int8 weights and bias
        attri.set_output_scales(mask, scale_data);
        mkldnn::reorder::primitive_desc reorder_pd =
            mkldnn::reorder::primitive_desc(user_mpd, mpd, attri);
        mkldnn::reorder reorder =
            mkldnn::reorder(reorder_pd, *user_memory, *dst_mem_p);
        stream(stream::kind::eager).submit({reorder}).wait();
      } else {
        auto reorder = mkldnn::reorder(*user_memory, *dst_mem_p);
        stream(stream::kind::eager).submit({reorder}).wait();
      }
    } else {
      dst_mem_p = user_memory;
    }
    return dst_mem_p;
  }

  std::shared_ptr<mkldnn::memory> AcquireMemory(
      const std::shared_ptr<mkldnn::memory>& user_memory_p,
      const std::shared_ptr<mkldnn::memory>& target_memory_p) {
    if user_memory_p->get_primitive_desc() != target_memory_p->get_primitive_desc(){
      auto reorder_p =
          std::make_shared<mkldnn::reorder>(*user_memory_p, *target_memory_p);
      stream(stream::kind::eager).submit({*reorder_p}).wait();
    }
    else{
      target_memory_p = user_memory_p;
    }
    return target_memory_p;
  }

  template <typename T>
  mkldnn::memory::desc CreateMemDescriptor(const std::vector<int>& dims,
                                           memory::format format) {
    return platform::MKLDNNMemDesc(dims, platform::MKLDNNGetDataType<T>(),
                                   format);
  }

  template <typename T>
  mkldnn::memory::desc CreateMemDescriptor(const Tensor* tensor,
                                           memory::format format) {
    auto dims = framework::vectorize2int(tensor->dims());
    return CreateMemDescriptor<T>({dims}, format);
  }

 private:
  void UpdateDataPointers(const ExecutionContext& ctx, Tensor* out,
                          const Tensor* in, const Tensor* residual_param) {
    user_src_p->set_data_handle(const_cast<T_in*>(in->data<T_in>()));
    input_p = AcquireMemory(conv_prim_desc_->src_primitive_desc(),
                           user_src_p->get_primitive_desc(), *user_src_p);
    auto fetched_dst_format =
        conv_prim_desc_->dst_primitive_desc().desc().data.format;
    size_t fetched_dst_size = conv_prim_desc_->dst_primitive_desc().get_size();

    if (residual_param) {
      residual_->set_data_handle(
          const_cast<T_out*>(residual_param->data<T_out>()));
      if (residual_param->format() != fetched_dst_format) {
        auto output_data =
            out->mutable_data<T_out>(ctx.GetPlace(), fetched_dst_size);
        output_->set_data_handle(output_data);
        // *output_ = AcquireMemory(conv_prim_desc_->dst_primitive_desc(), //
        // This part should not be reorder withconv_prim_desc but with target
        // memory and src memory
        //                         residual_->get_primitive_desc(), *residual_);
        auto reorder = mkldnn::reorder(*residual_, *output_);
        stream(stream::kind::eager).submit({reorder}).wait();
      } else {
        out->ShareDataWith(*residual_param);
        auto output_data = out->mutable_data<T_out>(ctx.GetPlace());
        output_->set_data_handle(output_data);
      }
    } else {
      auto output_data =
          out->mutable_data<T_out>(ctx.GetPlace(), fetched_dst_size);
      output_->set_data_handle(output_data);
    }
    out->mutable_data<T_out>(ctx.GetPlace());
    out->set_layout(DataLayout::kMKLDNN);
    out->set_format((memory::format)fetched_dst_format);
  }

  void CreateDstMemory(const ExecutionContext& ctx,
                       const Tensor* residual_param, Tensor* out) {
    auto fetched_dst_format =
        conv_prim_desc_->dst_primitive_desc().desc().data.format;
    auto fetched_dst_memory_size =
        conv_prim_desc_->dst_primitive_desc().get_size();

    if (residual_param) {
      auto residual_tz = framework::vectorize2int(residual_param->dims());
      auto residual_md = platform::MKLDNNMemDesc(
          residual_tz, framework::ToMKLDNNDataType(residual_param->type()),
          residual_param->format());
      residual_ = CreateMemory(
          residual_md,
          residual_param->data<T_out>());  // Note: If the residual_param
      // exist, T_out = residual_param->type()
      if (residual_param->format() != fetched_dst_format) {
        auto output_data =
            out->mutable_data<T_out>(ctx.GetPlace(), fetched_dst_memory_size);
        output_ = mkldnn::memory(conv_prim_desc_->dst_primitive_desc(),
                                 to_void_cast<T_out>(output_data));
        // output_ = AcquireMemory(conv_prim_desc_->dst_primitive_desc(),
        //                         residual_->get_primitive_desc(), *residual_);
        //                         // creat new output_data
        auto reorder = mkldnn::reorder(*residual_, *output_);
        stream(stream::kind::eager).submit({reorder}).wait();
      } else {
        out->ShareDataWith(*residual_param);
        auto output_data = out->mutable_data<T_out>(ctx.GetPlace());
        output_ = mkldnn::memory(conv_prim_desc_->dst_primitive_desc(),
                                 to_void_cast<T_out>(output_data));
      }
    } else {
      auto output_data =
          out->mutable_data<T_out>(ctx.GetPlace(), fetched_dst_memory_size);
      output_ = mkldnn::memory(conv_prim_desc_->dst_primitive_desc(),
                               to_void_cast<T_out>(output_data));
    }
    out->set_layout(DataLayout::kMKLDNN);
    out->set_format((memory::format)fetched_dst_format);
  }

  inline memory::format GetChosenFormat(const ExecutionContext& ctx,
                                        size_t src_tz_size, int groups,
                                        bool is_conv3d, bool is_int8) {
    std::string data_format = ctx.Attr<std::string>("data_format");
    auto chosen_memory_format =
        platform::data_format_to_memory_format(data_format);
    if (chosen_memory_format != memory::format::any && is_conv3d && !is_int8) {
      chosen_memory_format =
          platform::MKLDNNFormatForSize(src_tz_size, chosen_memory_format);
    }
    return chosen_memory_format;
  }

  template <typename T>
  std::shared_ptr<mkldnn::memory> CreateMemory(const mkldnn::memory::desc& desc,
                              const Tensor* tensor) {
    return CreateMemory(desc, tensor->data<T>());
  }

  std::shared_ptr<mkldnn::memory> CreateMemory(const mkldnn::memory::desc& desc,
                              const void* data) {
    auto memory_p = std::make_shared<mkldnn::memory>({desc, engine_}, const_cast<void*>(data));
    return memory_p;
  }

  std::vector<float> ComputeBiasScales(const ExecutionContext& ctx,
                                       const int groups,
                                       const std::vector<int>& weights_tz) {
    auto scale_in_data = ctx.Attr<float>("Scale_in");
    auto scale_weights_data = ctx.Attr<std::vector<float>>("Scale_weights");
    bool is_multi_channel = scale_weights_data.size() > 1;
    int count =
        is_multi_channel
            ? (groups > 1 ? (weights_tz)[1] * (weights_tz)[0] : (weights_tz)[0])
            : 1;
    std::vector<float> scale_bias_data(count);
#pragma omp parallel for if (count > 1)
    for (int i = 0; i < count; i++) {
      if (scale_weights_data[i] == 0.0)
        scale_bias_data[i] = 1.0f;
      else
        scale_bias_data[i] = scale_in_data * scale_weights_data[i];
    }
    return scale_bias_data;
  }

  std::shared_ptr<mkldnn::memory> QuantizeBias(const ExecutionContext& ctx, const int groups,
                              const std::vector<int>& weights_tz,
                              bool is_int8) {
    if (is_int8) {
      auto bias_scales = ComputeBiasScales(ctx, groups, weights_tz);
      auto scale_weights_data = ctx.Attr<std::vector<float>>("Scale_weights");
      bool is_multi_channel = scale_weights_data.size() > 1;
      int mask_reorder =
          is_multi_channel ? 1 << 0 : 1;  // 0000 0010 : 0000 0001
      bias_ = AcquireMemory(conv_prim_desc_->bias_primitive_desc(),
                            user_bias_p->get_primitive_desc(), *user_bias_p,
                            bias_scales, mask_reorder, is_int8);
    } else {
      bias_ = user_bias_p;
    }
    return bias_;
  }

  std::shared_ptr<mkldnn::memory> ReorderQuantizeWeights(const ExecutionContext& ctx,
                                        const int groups, bool is_int8) {
    if (is_int8) {
      auto scale_weights_data = ctx.Attr<std::vector<float>>("Scale_weights");
      bool is_multi_channel = scale_weights_data.size() > 1;
      int mask_reorder = is_multi_channel
                             ? ((groups != 1) ? (1 << 1) + (1 << 0) : 1 << 0)
                             : 0;  // 0000 0011: 0000 0001 : 0000 0000
      weights_p =
          AcquireMemory(conv_prim_desc_->weights_primitive_desc(),
                        user_weights_p->get_primitive_desc(), user_weights_p,
                        scale_weights_data, mask_reorder, is_int8);
    } else {
      weights_p =
          AcquireMemory(conv_prim_desc_->weights_primitive_desc(),
                        user_weights_p->get_primitive_desc(), user_weights_p);
    }
    return weights_p;
  }

  std::vector<float> ComputeOutputShiftScale(
      const ExecutionContext& ctx, const int groups,
      const std::vector<int>& weights_tz) {
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
    for (int i = 0; i < count; i++) {
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
    // To make the output with the scale of scale_out, scale the residual with
    // [scale_out_data / scale_in_eltwise_data]
    auto scale_out_data = ctx.Attr<bool>("force_fp32_output")
                              ? 1.0f
                              : ctx.Attr<float>("Scale_out");
    auto scale_in_eltwise_data = ctx.Attr<float>("Scale_in_eltwise");
    auto sum_scale = ctx.Attr<bool>("fuse_residual_connection")
                         ? scale_out_data / scale_in_eltwise_data
                         : 1.0f;
    return sum_scale;
  }

  mkldnn::primitive_attr CreatePostOps(
      const ExecutionContext& ctx, const int groups,
      const std::vector<int>& weights_tz, bool is_int8, bool fuse_relu,
      bool fuse_brelu, float fuse_brelu_threshold, bool fuse_residual_conn) {
    mkldnn::primitive_attr attributes;
    mkldnn::post_ops post_operations;
    float sum_scale = 1.0f;
    if (is_int8) {
      auto output_shift_scale =
          ComputeOutputShiftScale(ctx, groups, weights_tz);
      sum_scale = ComputeInt8SumScale(ctx);
      int mask =
          output_shift_scale.size() > 1 ? 1 << 1 : 0;  // 0000 0010 : 0000 0000
      attributes.set_output_scales(mask, output_shift_scale);
    }

    // Fusion with Elementwise layer relies on adding a sum post-operation with
    // the scale parameter. It is assumed that when fuse_residual_connection is
    // true, the output tensor contains the data coming from residual
    // connection. The result of this post_op is:
    // Output = scale * Output + Conv_Out.
    if (fuse_residual_conn) {
      post_operations.append_sum(sum_scale);
    }

    // Fusion with ReLU layer is executed through the PostOps feature. Create a
    // PostOps object and configure it to execute an eltwise relu operation.
    if (fuse_relu) {
      constexpr float scale = 1.0f;
      constexpr float negative_slope = 0.0f;
      constexpr float placeholder = 1.0f;  // beta
      post_operations.append_eltwise(scale, mkldnn::algorithm::eltwise_relu,
                                     negative_slope, placeholder);
    }

    if (fuse_brelu) {
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
      const mkldnn::memory::desc weights_md,
      const std::shared_ptr<mkldnn::memory::desc> bias_md_p,
      const mkldnn::memory::desc& dst_desc, const std::vector<int>& strides,
      const std::vector<int>& paddings, const bool fuse_relu,
      const bool fuse_residual_conn, const bool fuse_brelu,
      const float fuse_brelu_threshold, const bool is_test, const int groups,
      const std::vector<int>& weights_tz, bool is_int8) {
    const auto attrs =
        CreatePostOps(ctx, groups, weights_tz, is_int8, fuse_relu, fuse_brelu,
                      fuse_brelu_threshold, fuse_residual_conn);
    static std::mutex acquire_barrier;
    std::lock_guard<std::mutex> block_threads_until_finish_this_job(
        acquire_barrier);

    auto fwd_prop_kind = is_test ? mkldnn::prop_kind::forward_inference
                                 : mkldnn::prop_kind::forward_training;
    mkldnn::memory::dims stride_dims = {strides[0], strides[1]};
    mkldnn::memory::dims padding_dims = {paddings[0], paddings[1]};
    auto conv_desc =
        (bias_md_p != nullptr)
            ? mkldnn::convolution_forward::desc(
                  fwd_prop_kind, mkldnn::algorithm::convolution_direct,
                  input_desc, weights_md, (*bias_md_p), dst_desc, stride_dims,
                  padding_dims, padding_dims, mkldnn::padding_kind::zero)
            : mkldnn::convolution_forward::desc(
                  fwd_prop_kind, mkldnn::algorithm::convolution_direct,
                  input_desc, weights_md, dst_desc, stride_dims, padding_dims,
                  padding_dims, mkldnn::padding_kind::zero);
    return convolution_forward::primitive_desc(conv_desc, attrs, engine_);
  }

 private:
  const mkldnn::engine& engine_;
  // TODO(lidanqing) consider shared_ptr to avoid memory copy

  std::shared_ptr<memory> bias_;
  std::shared_ptr<memory> user_bias_p;
  std::shared_ptr<memory> input_p;
  std::shared_ptr<memory> user_src_p;
  std::shared_ptr<memory> output_;
  std::shared_ptr<memory> weights_;
  std::shared_ptr<memory> user_weights_p;
  std::shared_ptr<memory> residual_;
  std::shared_ptr<mkldnn::convolution_forward> conv_prim_;
  std::shared_ptr<mkldnn::convolution_forward::primitive_desc> conv_prim_desc_;
};

// T_w is the type of user_weights and user_bias. In convolution they are both
// float
// T_in is the type of user_input. T_out will be decided
template <typename T_in, typename T_w, typename T_out>
static std::shared_ptr<ConvPrimitiveFactory<T_in, T_w, T_out>>
GetConvPrimitiveFactory(const MKLDNNDeviceContext& dev_ctx,
                        const std::string key,
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

static mkldnn::memory::data_type GetDstType(bool is_int8,
                                            bool force_fp32_output,
                                            bool fuse_relu, bool fuse_brelu,
                                            bool fuse_residual_conn,
                                            const Tensor* residual_param) {
  auto dst_dt = mkldnn::memory::data_type::f32;  // uint8_t, int8_t, float
  if (is_int8) {
    dst_dt = (fuse_relu || fuse_brelu) ? mkldnn::memory::data_type::u8
                                       : mkldnn::memory::data_type::s8;
    if (force_fp32_output) {
      dst_dt = mkldnn::memory::data_type::f32;
    }
    if (fuse_residual_conn && residual_param) {
      auto residual_dt = framework::ToMKLDNNDataType(residual_param->type());
      if (dst_dt != residual_dt) dst_dt = residual_dt;
    }
  }
  return dst_dt;
}
static std::string GetHash(const mkldnn::memory::dims& input_dims,    // NOLINT
                           const mkldnn::memory::dims& weights_dims,  // NOLINT
                           const mkldnn::memory::data_type src_dt,    // NOLINT
                           const memory::format& format,              // NOLINT
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

  return dims2str(input_dims) + std::to_string(src_dt) + "-" +
         std::to_string(format) + "-" + dims2str(weights_dims) +
         std::to_string(fuse_relu) + "-" + std::to_string(fuse_brelu) + "-" +
         std::to_string(fuse_residual_conn) + "-" + vec2str(strides) +
         vec2str(paddings) + vec2str(dilations) + std::to_string(groups) + "-" +
         suffix;
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
    auto output = ctx.Output<LoDTensor>("Output");

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

    if (is_int8) {
      PADDLE_ENFORCE(is_conv3d == false,
                     "is_conv3d is not enabled when is_int8 is true");
    }
    auto src_dt = platform::MKLDNNGetDataType<T_in>();
    auto src_format = input->format();
    auto src_tz = framework::vectorize2int(input->dims());
    auto weights_tz = GetWeightsTz(ctx, weights, groups, is_conv3d);
    std::string key =
        GetHash(src_tz, weights_tz, src_dt, src_format, fuse_relu, fuse_brelu,
                fuse_residual_conn, strides, paddings, dilations, groups,
                ctx.op().Input("Input") + ctx.op().Input("Filter"));
    auto dst_typename =
        GetDstType(is_int8, force_fp32_output, fuse_relu, fuse_brelu,
                   fuse_residual_conn, residual_param);

    std::shared_ptr<mkldnn::convolution_forward> conv_p;
    if (dst_typename == mkldnn::memory::data_type::f32) {
      conv_p = std::make_shared<mkldnn::convolution_forward>(
          GetConvPrimitiveFactory<T_in, T_w, float>(dev_ctx, key, mkldnn_engine)
              ->AcquireConvPrimitive(input, weights, bias, strides, paddings,
                                     groups, is_conv3d, fuse_relu, fuse_brelu,
                                     fuse_brelu_threshold, fuse_residual_conn,
                                     residual_param, output, is_test, ctx,
                                     is_int8));
    } else if (dst_typename == mkldnn::memory::data_type::u8) {
      conv_p = std::make_shared<mkldnn::convolution_forward>(
          GetConvPrimitiveFactory<T_in, T_w, uint8_t>(dev_ctx, key,
                                                      mkldnn_engine)
              ->AcquireConvPrimitive(input, weights, bias, strides, paddings,
                                     groups, is_conv3d, fuse_relu, fuse_brelu,
                                     fuse_brelu_threshold, fuse_residual_conn,
                                     residual_param, output, is_test, ctx,
                                     is_int8));
    } else if (dst_typename == mkldnn::memory::data_type::s8) {
      conv_p = std::make_shared<mkldnn::convolution_forward>(
          GetConvPrimitiveFactory<T_in, T_w, int8_t>(dev_ctx, key,
                                                     mkldnn_engine)
              ->AcquireConvPrimitive(input, weights, bias, strides, paddings,
                                     groups, is_conv3d, fuse_relu, fuse_brelu,
                                     fuse_brelu_threshold, fuse_residual_conn,
                                     residual_param, output, is_test, ctx,
                                     is_int8));
    }
    stream(stream::kind::eager).submit({*conv_p}).wait();
    // output->set_layout(DataLayout::kMKLDNN);
  }
};

template <typename T>
class ConvMKLDNNGradOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(paddle::platform::is_cpu_place(ctx.GetPlace()),
                   "It must use CPUPlace.");

    auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    const Tensor* input = ctx.Input<Tensor>("Input");
    const Tensor* filter = ctx.Input<Tensor>("Filter");
    const Tensor* output_grad =
        ctx.Input<Tensor>(framework::GradVarName("Output"));
    Tensor* input_grad = ctx.Output<Tensor>(framework::GradVarName("Input"));
    Tensor* filter_grad = ctx.Output<Tensor>(framework::GradVarName("Filter"));

    PADDLE_ENFORCE(input->layout() == DataLayout::kMKLDNN &&
                       input->format() != memory::format::format_undef,
                   "Wrong layout/format set for Input tensor");
    PADDLE_ENFORCE(filter->layout() == DataLayout::kMKLDNN &&
                       filter->format() != memory::format::format_undef,
                   "Wrong layout/format set for Filter tensor");
    PADDLE_ENFORCE(output_grad->layout() == DataLayout::kMKLDNN &&
                       output_grad->format() != memory::format::format_undef,
                   "Wrong layout/format set for output_grad tensor");

    PADDLE_ENFORCE(
        !ctx.Attr<bool>("is_test"),
        "is_test attribute should be set to False in training phase.");

    if (!input_grad && !filter_grad) return;

    std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");
    int groups = ctx.Attr<int>("groups");

    bool is_conv3d = strides.size() == 3U;
    const T* input_data = input->data<T>();
    const T* filter_data = filter->data<T>();
    const T* output_grad_data = output_grad->data<T>();
    T* input_grad_data = nullptr;
    T* filter_grad_data = nullptr;

    std::vector<int> src_tz = paddle::framework::vectorize2int(input->dims());

    int g = std::max(groups, 1);
    std::vector<int> weights_tz = GetWeightsTz(ctx, filter, g, is_conv3d);
    std::vector<int> dst_tz =
        paddle::framework::vectorize2int(output_grad->dims());
    bool fuse_relu = ctx.Attr<bool>("fuse_relu");
    bool fuse_brelu = false;
    if (!is_conv3d) {
      fuse_brelu = ctx.Attr<bool>("fuse_brelu");
    }
    auto src_format = input->format();
    mkldnn::memory::format weights_format =
        GetWeightsFormat(filter->format(), g, is_conv3d);

    // Get an unique name from "argument" name of "input" and "Filter" variable
    // as well as attributes of primitive to be created
    // This name will be used as key when saving info into device context
    const std::string key = platform::ConvMKLDNNHandler::GetHash(
        src_tz, weights_tz, fuse_relu, fuse_brelu, strides, paddings, dilations,
        groups, ctx.op().Input("Input") + ctx.op().Input("Filter"));

    const std::string key_conv_pd = key + "@conv_pd";
    std::vector<primitive> pipeline;

    // Create user memory descriptors
    auto user_src_md = platform::MKLDNNMemDesc(
        {src_tz}, platform::MKLDNNGetDataType<T>(), src_format);
    auto user_weights_md = platform::MKLDNNMemDesc(
        {weights_tz}, platform::MKLDNNGetDataType<T>(), weights_format);
    auto user_diff_dst_md = platform::MKLDNNMemDesc(
        {dst_tz}, platform::MKLDNNGetDataType<T>(), output_grad->format());

    /* create memory descriptor for conv backward without specified format
     * ('any') which lets a primitive (conv backward in this case) choose
     * the memory format preferred for best performance
     */
    std::string data_format = ctx.Attr<std::string>("data_format");
    auto chosen_memory_format =
        platform::data_format_to_memory_format(data_format);

    weights_format = mkldnn::memory::format::any;
    // Check the format for user's special output
    if (chosen_memory_format != mkldnn::memory::format::any) {
      if (is_conv3d) {
        chosen_memory_format =
            platform::MKLDNNFormatForSize(src_tz.size(), chosen_memory_format);
      }
    }

    auto src_md = platform::MKLDNNMemDesc(
        src_tz, platform::MKLDNNGetDataType<T>(), chosen_memory_format);
    auto diff_src_md = platform::MKLDNNMemDesc(
        src_tz, platform::MKLDNNGetDataType<T>(), chosen_memory_format);
    auto weights_md = platform::MKLDNNMemDesc(
        weights_tz, platform::MKLDNNGetDataType<T>(), weights_format);
    auto diff_weights_md = platform::MKLDNNMemDesc(
        weights_tz, platform::MKLDNNGetDataType<T>(), weights_format);
    auto diff_dst_md = platform::MKLDNNMemDesc(
        dst_tz, platform::MKLDNNGetDataType<T>(), chosen_memory_format);

    // Retrieve conv_pd from device context
    auto conv_pd =
        std::static_pointer_cast<mkldnn::convolution_forward::primitive_desc>(
            dev_ctx.GetBlob(key_conv_pd));
    PADDLE_ENFORCE(conv_pd != nullptr,
                   "Fail to find conv_pd in device context");

    // create backward convolution weights primitive descriptor
    auto conv_bwd_weights_desc = mkldnn::convolution_backward_weights::desc(
        mkldnn::convolution_direct, src_md, diff_weights_md, diff_dst_md,
        strides, paddings, paddings, mkldnn::padding_kind::zero);
    auto conv_bwd_weights_pd =
        std::make_shared<mkldnn::convolution_backward_weights::primitive_desc>(
            conv_bwd_weights_desc, mkldnn_engine, *conv_pd);

    // create backward convolution data primitive descriptor
    auto conv_bwd_data_desc = mkldnn::convolution_backward_data::desc(
        mkldnn::convolution_direct, diff_src_md, weights_md, diff_dst_md,
        strides, paddings, paddings, mkldnn::padding_kind::zero);
    auto conv_bwd_data_pd =
        std::make_shared<mkldnn::convolution_backward_data::primitive_desc>(
            conv_bwd_data_desc, mkldnn_engine, *conv_pd);

    platform::ConvMKLDNNHandler handler(conv_pd, conv_bwd_data_pd,
                                        conv_bwd_weights_pd, dev_ctx,
                                        mkldnn_engine, key);

    // create mkldnn memory from input tensors (data/weights)
    auto user_src_memory_p =
        handler.AcquireSrcMemory(user_src_md, to_void_cast<T>(input_data));
    auto user_weights_memory_p = handler.AcquireWeightsMemory(
        user_weights_md, to_void_cast<T>(filter_data));
    auto user_diff_dst_memory_p = handler.AcquireDiffDstMemory(
        user_diff_dst_md, to_void_cast<T>(output_grad_data));

    // create backward conv primitive for weights
    if (filter_grad) {
      auto src_memory_p = handler.AcquireSrcMemoryFromWeightsPrimitive(
          user_src_memory_p, pipeline);

      auto diff_dst_memory_4filter_p =
          handler.AcquireDiffDstMemoryFromWeightsPrimitive(
              user_diff_dst_memory_p, pipeline);

      const size_t size = handler.GetDiffWeightsMemorySize();
      filter_grad_data = filter_grad->mutable_data<T>(ctx.GetPlace(), size);

      auto diff_weights_memory_p =
          handler.AcquireDiffWeightsMemoryFromWeightsPrimitive(
              reinterpret_cast<void*>(filter_grad_data));

      auto conv_bwd_weights_p = handler.AcquireConvolutionBackwardWeights(
          src_memory_p, diff_dst_memory_4filter_p, diff_weights_memory_p);

      // push primitive to stream and wait until it's executed
      pipeline.push_back(*conv_bwd_weights_p);

      filter_grad->set_layout(DataLayout::kMKLDNN);
      filter_grad->set_format(GetMKLDNNFormat(*diff_weights_memory_p));
    }

    if (input_grad) {
      auto weights_memory_p = handler.AcquireWeightsMemoryFromDataPrimitive(
          user_weights_memory_p, pipeline);

      auto diff_dst_memory_4data_p =
          handler.AcquireDiffDstMemoryFromDataPrimitive(user_diff_dst_memory_p,
                                                        pipeline);

      const size_t size = handler.GetDiffSourceMemorySize();
      input_grad_data = input_grad->mutable_data<T>(ctx.GetPlace(), size);

      auto diff_src_memory_p = handler.AcquireDiffSrcMemoryFromDataPrimitive(
          reinterpret_cast<void*>(input_grad_data));

      auto conv_bwd_data_p = handler.AcquireConvolutionBackwardData(
          diff_dst_memory_4data_p, weights_memory_p, diff_src_memory_p);

      pipeline.push_back(*conv_bwd_data_p);

      input_grad->set_layout(DataLayout::kMKLDNN);
      input_grad->set_format(GetMKLDNNFormat(*diff_src_memory_p));
    }
    stream(stream::kind::eager).submit(pipeline).wait();
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(conv2d, MKLDNN,
                                    ::paddle::platform::CPUPlace, FP32,
                                    ops::kConvMKLDNNFP32,
                                    ops::ConvMKLDNNOpKernel<float, float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(conv2d, MKLDNN,
                                    ::paddle::platform::CPUPlace, U8,
                                    ops::kConvMKLDNNINT8,
                                    ops::ConvMKLDNNOpKernel<uint8_t, float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(conv2d, MKLDNN,
                                    ::paddle::platform::CPUPlace, S8,
                                    ops::kConvMKLDNNINT8,
                                    ops::ConvMKLDNNOpKernel<int8_t, float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(conv3d, MKLDNN,
                                    ::paddle::platform::CPUPlace, FP32,
                                    ops::kConvMKLDNNFP32,
                                    ops::ConvMKLDNNOpKernel<float, float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(conv2d_grad, MKLDNN,
                                    ::paddle::platform::CPUPlace, FP32,
                                    ops::kConvMKLDNNFP32,
                                    ops::ConvMKLDNNGradOpKernel<float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(conv3d_grad, MKLDNN,
                                    ::paddle::platform::CPUPlace, FP32,
                                    ops::kConvMKLDNNFP32,
                                    ops::ConvMKLDNNGradOpKernel<float>);
