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
from paddle.fluid.tests.unittests.test_sum_op import TestSumOp, TestSelectedRowsSumOp, TestLoDTensorAndSelectedRowsOp
import paddle.fluid.core as core

class TestMKLDNN(TestSumOp):
    def init_kernel_type(self):
        self.use_mkldnn = True
        self.attrs = {'_cpu_only': True}

class TestMKLDNNInplaceSumOp(TestSumOp):
    def setUp(self):
        self.op_type = "sum"
        self.init_kernel_type()
        self.use_mkldnn = False
        self.init_kernel_type()
        x0 = np.random.random((3, 4)).astype(self.dtype)
        
        #self.inputs = {"X": [("x0", x0), ("x1", x1), ("x2", x2)]}
        #y = x0 + x1 + x2
        #self.outputs = {'Out': y}
        
        print("Start")
        print(id(x0))
        # xtemp = [x0] # Here when you add xtemp, it is a new variable. print(id(x0))
        self.inputs = {"X": [("x0", x0)]}
        self.outputs = {"Out": x0 }
        print(id(x0))
        print("End")
        self.attrs = {'use_mkldnn': self.use_mkldnn}

    def init_kernel_type(self):
        self.use_mkldnn = True

class TestMKLDNNSelectedRowsSumOp(TestSelectedRowsSumOp):
    def init_kernel_type(self):
        self.use_mkldnn = True
        self.attrs = {'_cpu_only': True}

# class TestMKLDNNLoDTensorAndSelectedRowsOp(TestLoDTensorAndSelectedRowsOp):
#     def init_kernel_type(self):
#         self.use_mkldnn = True
#          self.attrs = {'_cpu_only': True}

# class TestWithInplace(TestSelectedRowsSumOp):
   
#     def test_w_is_selected_rows(self):
#         places = [core.CPUPlace()]
#         # if core.is_compiled_with_cuda():
#         #     places.append(core.CUDAPlace(0))
#         for place in places:
#             for inplace in [True]:
#                 self.check_with_place(place, inplace)

#     def init_kernel_type(self):
#         self.use_mkldnn = True
#         self.attrs = {'_cpu_only': True}

if __name__ == '__main__':
    unittest.main()
