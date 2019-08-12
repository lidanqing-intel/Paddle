#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import unittest
import numpy as np
from paddle.fluid.tests.unittests.op_test import OpTest
import paddle.fluid.core as core
from paddle.fluid.op import Operator
from paddle.fluid.tests.unittests.test_elementwise_mul_op import *
from paddle.fluid.tests.unittests.test_conv2d_op import conv2d_forward_naive
from paddle.fluid.tests.unittests.mkldnn.mkldnn_op_test import __assert_close


# For UT coverage, add conv2d and elementwise-mul into this test so that nchw16C could be automatically chosen when mkldnn-kernel is enabled
class TestElementwiseMulMKLDNNOp_BroadcastNCHW16c(ElementwiseMulOp):
    def setUp(self):
        self.op_type = "elementwise_mul"
        self.dtype = np.float32
        self.init_dtype()
        self.init_kernel_type()
        self.init_axis()
        self._cpu_only = True
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.groups = 1
        self.input_size = [1, 3, 4, 4]  # NCHW
        self.filter_size = [16, 3, 3, 3]
        self.dilations = False
        self.use_cudnn = False
        self.data_format = "NCHW"
        self.fuse_relu_before_depthwise_conv = False
        self.fuse_relu_before_depthwise_conv = False
        self.exhaustive_search = False
        self.input = np.random.rand(1, 3, 4, 4).astype(self.dtype)
        self.filter = np.random.rand(16, 3, 3, 3).astype(self.dtype)

        conv2d_param = {
            'stride': self.stride,
            'pad': self.pad,
            'dilation': self.dilations
        }

        conv_out, _, _, _, _ = conv2d_forward_naive(
            self.input, self.filter, self.groups, conv2d_param)  #[1, 16, 2, 2]
        self.conv_output = conv_out
        self.elt_mul_y_size = [1, 16, 1, 1]
        self.elt_mul_y = np.random.rand(1, 16, 1, 1).astype(self.dtype)
        self.elt_mul_output = self.conv_output * self.elt_mul_y  # the result dimension is 1*16*2*2

        self.fetch_list = ["conv_output", "elt_mul_output"]

    def init_kernel_type(self):
        self.use_mkldnn = True

    def init_axis(self):
        self.axis = 0

    def test_check_output(self):
        print("SHOCK!!! THIS HAS TO BE SHOWN UP")
        ground_truth = {
            "input": self.input,
            "filter": self.filter,
            "conv_output": self.conv_output,
            "elt_mul_y": self.elt_mul_y,
            "elt_mul_output": self.elt_mul_output
        }
        program = fluid.Program()
        with fluid.program_guard(program):
            block = program.global_block()
            for name in ground_truth:
                block.create_var(
                    name=name, dtype="float32", shape=ground_truth[name].shape)
            bn_op = block.append_op(
                type="conv2d",
                inputs={
                    "Input": block.var('input'),
                    'Filter': block.var('filter')
                },
                outputs={"Conv_Output": block.var('conv_output')},
                attrs={
                    'strides': self.stride,
                    'paddings': self.pad,
                    'groups': self.groups,
                    'dilations': self.dilations,
                    'use_cudnn': self.use_cudnn,
                    'use_mkldnn': self.use_mkldnn,
                    'data_format': self.data_format,
                    'fuse_relu_before_depthwise_conv':
                    self.fuse_relu_before_depthwise_conv,
                    'exhaustive_search': self.exhaustive_search
                })
            elementwise_mul_op = block.append_op(
                type="conv2d",
                inputs={
                    'X': block.var('conv_output'),
                    'Y': block.var('elt_mul_y'),
                },
                outputs={"Output": block.var('elt_mul_output'), },
                attrs={
                    'use_cudnn': self.use_cudnn,
                    'use_mkldnn': self.use_mkldnn,
                    'data_format': self.data_format
                })
            out = exe.run(program,
                          feed={
                              name: ground_truth[name]
                              for name in ["input", "filter", "elt_mul_y"]
                          },
                          fetch_list=self.fetch_list)

            for id, name in enumerate(self.fetch_list):
                print("ref is ", ground_truth[name], "real out is ", out[id],
                      "name is ", name)
                __assert_close(self, ground_truth[name], out[id], name)

    def test_check_grad_normal(self):
        pass

    def test_check_grad_ingore_x(self):
        pass

    def test_check_grad_ingore_y(self):
        pass


# # TODO(LeoZhao-Intel): re-enable this case
# # https://github.com/PaddlePaddle/Paddle/issues/16764
# @unittest.skip("Not supported well on avx2.")
# class TestElementwiseMulMKLDNNOp_BroadcastNCHW16c(ElementwiseMulOp):
#     def init_input_output(self):
#         x = np.random.rand(1, 16, 2, 2).astype(self.dtype)
#         self.x = x.transpose(0, 2, 3, 1).reshape(1, 16, 2, 2)
#         self.y = np.random.rand(1, 16).astype(self.dtype)

#         self.out = x * self.y.reshape(1, 16, 1, 1)
#         self.out = self.out.transpose(0, 2, 3, 1).reshape(1, 16, 2, 2)

#     def setUp(self):
#         super(TestElementwiseMulMKLDNNOp_BroadcastNCHW16c, self).setUp()
#         self.attrs["x_data_format"] = "nchw16c"
#         self.attrs["y_data_format"] = "nc"
#         self._cpu_only = True

#     def init_kernel_type(self):
#         self.use_mkldnn = True

#     def init_axis(self):
#         self.axis = 0

#     #unit test does not require the output format must be mkldnn format. If use_mkldnn is true
#     def test_check_output(self):
#         self.check_output()

#     def test_check_grad_normal(self):
#         pass

#     def test_check_grad_ingore_x(self):
#         pass

#     def test_check_grad_ingore_y(self):
#         pass

# @unittest.skip(
#     "Not implemented yet.")  # TODO(mgallus): enable when implemented.
# class TestElementwiseMulMKLDNNOp_BroadcastNCHW8c(ElementwiseMulOp):
#     def init_input_output(self):
#         x = np.random.rand(1, 8, 2, 2).astype(self.dtype)
#         self.x = x.transpose(0, 2, 3, 1).reshape(1, 8, 2, 2)
#         self.y = np.random.rand(1, 8).astype(self.dtype)

#         self.out = x * self.y.reshape(1, 8, 1, 1)
#         self.out = self.out.transpose(0, 2, 3, 1).reshape(1, 8, 2, 2)

#     def setUp(self):
#         super(TestElementwiseMulMKLDNNOp_BroadcastNCHW8c, self).setUp()
#         self.attrs["x_data_format"] = "nchw8c"
#         self.attrs["y_data_format"] = "nc"
#         self._cpu_only = True

#     def init_kernel_type(self):
#         self.use_mkldnn = True

#     def init_axis(self):
#         self.axis = 0

#     def test_check_grad_normal(self):
#         pass

#     def test_check_grad_ingore_x(self):
#         pass

#     def test_check_grad_ingore_y(self):
#         pass


class TestElementwiseMulMKLDNNOp_FallbackNCHW(ElementwiseMulOp):
    def init_input_output(self):
        self.x = np.random.rand(1, 16, 2, 2).astype(self.dtype)
        self.y = np.random.rand(1, 16).astype(self.dtype)

        self.out = self.x * self.y.reshape(1, 16, 1, 1)

    def init_kernel_type(self):
        self.use_mkldnn = True

    def init_axis(self):
        self.axis = 0

    def test_check_grad_normal(self):
        pass

    def test_check_grad_ingore_x(self):
        pass

    def test_check_grad_ingore_y(self):
        pass


class TestElementwiseMulMKLDNNOp_FallbackNCHW16C(ElementwiseMulOp):
    def init_input_output(self):
        x = np.random.rand(1, 16, 2, 2).astype(self.dtype)
        self.x = x.transpose(0, 2, 3, 1).reshape(1, 16, 2, 2)
        y = np.random.rand(1, 16, 2, 2).astype(self.dtype)
        self.y = y.transpose(0, 2, 3, 1).reshape(1, 16, 2, 2)

        self.out = self.x * self.y

    def setUp(self):
        super(TestElementwiseMulMKLDNNOp_FallbackNCHW16C, self).setUp()
        self.attrs["x_data_format"] = "nchw16c"
        self.attrs["y_data_format"] = "nchw16c"
        self._cpu_only = True

    def init_kernel_type(self):
        self.use_mkldnn = True

    def init_axis(self):
        self.axis = 0

    def test_check_grad_normal(self):
        pass

    def test_check_grad_ingore_x(self):
        pass

    def test_check_grad_ingore_y(self):
        pass


class TestElementwiseMulMKLDNNOp_FallbackNoReorders(ElementwiseMulOp):
    def init_input_output(self):
        x = np.random.rand(1, 16, 2, 2).astype(self.dtype)
        self.x = x.transpose(0, 2, 3, 1).reshape(1, 16, 2, 2)
        y = np.random.rand(1, 16, 2, 2).astype(self.dtype)
        self.y = y.transpose(0, 2, 3, 1).reshape(1, 16, 2, 2)

        self.out = self.x * self.y

    def setUp(self):
        super(TestElementwiseMulMKLDNNOp_FallbackNoReorders, self).setUp()
        self.attrs["x_data_format"] = "nchw16c"
        self.attrs["y_data_format"] = "nchw16c"
        self._cpu_only = True

    def init_kernel_type(self):
        self.use_mkldnn = True

    def init_axis(self):
        self.axis = 0

    def test_check_grad_normal(self):
        pass

    def test_check_grad_ingore_x(self):
        pass

    def test_check_grad_ingore_y(self):
        pass


class TestElementwiseMulMKLDNNOp_FallbackWithReorder1(ElementwiseMulOp):
    def init_input_output(self):
        self.x = np.random.rand(1, 16, 2, 2).astype(self.dtype)
        y = np.random.rand(1, 16, 2, 2).astype(self.dtype)
        self.y = y.transpose(0, 2, 3, 1).reshape(1, 16, 2, 2)

        self.out = self.x * y

    def setUp(self):
        super(TestElementwiseMulMKLDNNOp_FallbackWithReorder1, self).setUp()
        self.attrs["x_data_format"] = "nchw"
        self.attrs["y_data_format"] = "nchw16c"
        self._cpu_only = True

    def init_kernel_type(self):
        self.use_mkldnn = True

    def init_axis(self):
        self.axis = 0

    def test_check_grad_normal(self):
        pass

    def test_check_grad_ingore_x(self):
        pass

    def test_check_grad_ingore_y(self):
        pass


class TestElementwiseMulMKLDNNOp_FallbackWithReorder2(ElementwiseMulOp):
    def init_input_output(self):
        self.y = np.random.rand(1, 16, 2, 2).astype(self.dtype)
        x = np.random.rand(1, 16, 2, 2).astype(self.dtype)
        self.x = x.transpose(0, 2, 3, 1).reshape(1, 16, 2, 2)

        self.out = x * self.y

    def setUp(self):
        super(TestElementwiseMulMKLDNNOp_FallbackWithReorder2, self).setUp()
        self.attrs["x_data_format"] = "nchw16c"
        self.attrs["y_data_format"] = "nchw"
        self._cpu_only = True

    def init_kernel_type(self):
        self.use_mkldnn = True

    def init_axis(self):
        self.axis = 0

    def test_check_grad_normal(self):
        pass

    def test_check_grad_ingore_x(self):
        pass

    def test_check_grad_ingore_y(self):
        pass


class TestElementwiseMulMKLDNNOp_FallbackNoReorders2(ElementwiseMulOp):
    def init_input_output(self):
        self.x = np.random.rand(1, 16).astype(self.dtype)
        self.y = np.random.rand(1, 16).astype(self.dtype)

        self.out = self.x * self.y

    def setUp(self):
        super(TestElementwiseMulMKLDNNOp_FallbackNoReorders2, self).setUp()
        self.attrs["x_data_format"] = "nc"
        self.attrs["y_data_format"] = "nc"
        self._cpu_only = True

    def init_kernel_type(self):
        self.use_mkldnn = True

    def init_axis(self):
        self.axis = 0

    def test_check_grad_normal(self):
        pass

    def test_check_grad_ingore_x(self):
        pass

    def test_check_grad_ingore_y(self):
        pass


if __name__ == '__main__':
    unittest.main()
