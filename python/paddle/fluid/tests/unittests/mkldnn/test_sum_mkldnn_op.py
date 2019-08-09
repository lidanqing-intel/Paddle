# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import unittest
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.backward import append_backward
from paddle.fluid.executor import Executor
from paddle.fluid.framework import Program, OpProtoHolder, Variable
import paddle.fluid.core as core
from paddle.fluid.tests.unittests.test_sum_op import TestSumOp
from paddle.fluid.tests.unittests.op_test import OpTest
from paddle.fluid.op import Operator

# Maybe this inplace is not registered even. Because original does not have this. It seems I need to find from the registered ones first. That get_all_ops did not register this


def create_or_get_tensor(scope, var_name, var, place):
    tensor = scope.var(var_name).get_tensor()
    if var is not None:
        assert isinstance(var, np.ndarray)
        tensor.set_recursive_sequence_lengths([])
        tensor.set(var, place)
    return tensor


# class TestMKLDNN1(TestSumOp):
#     def init_kernel_type(self):
#         self.use_mkldnn = True

#     def setUp(self):
#         self.op_type = "sum"
#         self.init_kernel_type()
#         self.use_mkldnn = False
#         self.init_kernel_type()
#         x0 = np.random.random((3, 4)).astype(self.dtype)
#         x1 = np.random.random((3, 4)).astype(self.dtype)
#         x2 = np.random.random((3, 4)).astype(self.dtype)
#         self.inputs = {"X": [("x0", x0), ("x1", x1), ("x2", x2)]}
#         y = x0 + x1 + x2
#         self.outputs = {'Out': y}
#         self.attrs = {'use_mkldnn': self.use_mkldnn}

# class TestMKLDNN(TestSumOp):

#     def init_kernel_type(self):
#         self.use_mkldnn = True

#     def setUp(self):
#         self.op_type = "sum"
#         self.dtype = np.float32
#         self.use_mkldnn = False
#         self.init_kernel_type()
#         x0 = np.random.random((3, 4)).astype(self.dtype)
#         x1 = np.random.random((3, 4)).astype(self.dtype)
#         x2 = np.random.random((3, 4)).astype(self.dtype)
#         self.inputs = {"X": [("x0", x0), ("x1", x1), ("x2", x2)]}
#         y = x0 + x1 + x2
#         self.outputs = {'Out': y}
#         self.attrs = {'use_mkldnn': self.use_mkldnn}

#     def test_check_output(self):
#         pass
#         # self.check_output()

#     def test_check_grad(self):
#         pass
#         # self.check_grad(['x0'], 'Out')


class TestSelectedRowsSumMKLDNNOp(OpTest):
    def setUp(self):
        self.op_type = "sum"
        self.use_mkldnn = True
        # self.init_kernel_type()
        self.attrs = {'use_mkldnn': self.use_mkldnn}
        # x0 = np.random.random((3, 4)).astype(self.dtype)
        # x1 = np.random.random((3, 4)).astype(self.dtype)
        # x2 = np.random.random((3, 4)).astype(self.dtype)
        # self.inputs = {"X": [("x0", x0), ("x1", x1), ("x2", x2)]}
        # y = x0 + x1 + x2
        # self.outputs = {'Out': y}

        # place = core.CPUPlace()
        # x0 = np.random.random((3, 4)).astype(self.dtype)
        # x1 = np.random.random((3, 4)).astype(self.dtype)
        # y =x0 + x1
        # var_dict = {'x0': x0, 'x1': x1, 'y' : y}
        # var_names = list(var_dict.keys())
        # ground_truth = {name: var_dict[name] for name in var_names}

        # program = fluid.Program()

        # with fluid.program_guard(program):
        #     block = program.global_block()
        #     for name in ground_truth:
        #         block.create_var(
        #             name=name, dtype=np.float32, shape=ground_truth[name].shape)
        #     op = block.append_op(
        #         type=self.op_type,
        #         inputs= {'X': [block.var("x0"), block.var("x1")] },
        #         outputs={'Out': block.var("y")},
        #         attrs={'use_mkldnn': True})

        #     # Generate backward op_desc
        #     grad_op_desc_list, op_grad_to_var = core.get_grad_op_desc(op.desc,
        #                                                             set(), [])
        #     grad_op_desc = grad_op_desc_list[0]
        #     new_op_desc = block.desc.append_op()
        #     new_op_desc.copy_from(grad_op_desc)
        #     for var_name in grad_op_desc.output_arg_names():
        #         block.desc.var(var_name.encode('ascii'))
        #     grad_op_desc.infer_var_type(block.desc)
        #     grad_op_desc.infer_shape(block.desc)
        #     for arg in grad_op_desc.output_arg_names():
        #         grad_var = block.desc.find_var(arg.encode('ascii'))
        #         grad_var.set_dtype(core.VarDesc.VarType.FP32)

        #     exe = fluid.Executor(place)
        #     out = exe.run(
        #         program,
        #         feed={'X'},
        #         fetch_list=['Out'])

    def test_check_output(self):
        pass

    def check_output_with_place(self):
        place = core.CPUPlace()
        scope = core.Scope()
        x0 = np.random.random((3, 4)).astype(self.dtype)
        x1 = np.random.random((3, 4)).astype(self.dtype)

        # create input
        x0_tensor = create_or_get_tensor(scope, "x0",
                                         OpTest.np_dtype_to_fluid_dtype(x0),
                                         place)
        x1_tensor = create_or_get_tensor(scope, "x1",
                                         OpTest.np_dtype_to_fluid_dtype(x1),
                                         place)

        # create output
        y_tensor = create_or_get_tensor(scope, "x0", None, place)
        print("USE_MKLDNN IS SET TO ", self.use_mkldnn)
        batch_norm_op = Operator(
            "sum",
            # inputs
            X=["x0", "x1"],
            # outputs
            Y="x0",
            # attrs
            is_test=True,
            # data_layout=data_layout,
            use_mkldnn=self.use_mkldnn)

        batch_norm_op.run(scope, place)

    def test_check_grad(self):
        pass

    def test_check_grad_with_place(self):
        pass


# def test_cpu_inplace(self):
#     place = core.CPUPlace()
#     def __assert_close(tensor, np_array, msg, atol=1e-4):
#         self.assertTrue(
#             np.allclose(
#                 np.array(tensor), np_array, atol=atol), msg)
#     self.create_lod_tensor(scope, place, "x1")
#     print(self.use_mkldnn)
#     print("outid is ", id(out))
#     out_name = "x1"
#     # create and run sum operator
#     #         op = block.append_op(
#     # type=op_type,
#     # inputs={'X': block.var('x'), },
#     # outputs={'Out': block.var('out')},
#     # attrs={'use_mkldnn': True})

#     sum_op = Operator("sum", X=["x1"], Out=out_name, use_mkldnn=self.use_mkldnn)
#     # cur_block.append_op(type="sum",
#     #                     inputs={"X": [var1, var2, var3]},
#     #                     outputs={"Out": [var1]})
#     print("attrs has been changed to Attr")
#     sum_op.run(scope, place)
#     x0 = np.random.random((3, 4)).astype(self.dtype)
#     x1 = np.random.random((3, 4)).astype(self.dtype)
#     var_dict = {'x0': x0, 'x1':x1}
#     var_names = list(var_dict.keys())
#     ground_truth = {name: var_dict[name] for name in var_names}

#     program = fluid.Program()

#     with fluid.program_guard(program):
#         block = program.global_block()
#         for name in ground_truth:
#             block.create_var(
#                 name=name, dtype=np.float32, shape=ground_truth[name].shape)
#         op = block.append_op(
#             type=self.op_type,
#             inputs= {'X': [block.var("x0"), block.var("x1")] },
#             outputs={'Out': block.var("x0")},
#             attrs={'use_mkldnn': True})

#         # Generate backward op_desc
#         grad_op_desc_list, op_grad_to_var = core.get_grad_op_desc(op.desc,
#                                                                 set(), [])
#         grad_op_desc = grad_op_desc_list[0]
#         new_op_desc = block.desc.append_op()
#         new_op_desc.copy_from(grad_op_desc)
#         for var_name in grad_op_desc.output_arg_names():
#             block.desc.var(var_name.encode('ascii'))
#         grad_op_desc.infer_var_type(block.desc)
#         grad_op_desc.infer_shape(block.desc)
#         for arg in grad_op_desc.output_arg_names():
#             grad_var = block.desc.find_var(arg.encode('ascii'))
#             grad_var.set_dtype(core.VarDesc.VarType.FP32)

#         exe = fluid.Executor(place)
#         out = exe.run(
#             program,
#             feed={'X'},
#             fetch_list=['Out'])
#     out_t = np.array(out)
#     print(out_t)
#     self.assertTrue(
#         np.allclose(
#             np.array(0.33), out_t, atol=1e-4), 'y == out')

if __name__ == '__main__':
    unittest.main()
