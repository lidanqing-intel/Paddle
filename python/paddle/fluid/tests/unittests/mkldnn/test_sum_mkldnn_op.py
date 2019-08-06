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

from paddle.fluid.tests.unittests.test_sum_op import TestSumOp


class TestMKLDNN(TestSumOp):
    def init_kernel_type(self):
        print("!!!!!!MARK!!! self.use_mkldnn is set to TRUE")
        self.use_mkldnn = True


class TestSelectedRowsSumMKLDNNOp(OpTest):
    def setUp(self):
        self.op_type = "sum"
        self.init_kernel_type()
        self.use_mkldnn = True
        # self.init_kernel_type()
        self.attrs = {'use_mkldnn': self.use_mkldnn}
        # x0 = np.random.random((3, 4)).astype(self.dtype)
        # x1 = np.random.random((3, 4)).astype(self.dtype)
        # x2 = np.random.random((3, 4)).astype(self.dtype)
        # self.inputs = {"X": [("x0", x0), ("x1", x1), ("x2", x2)]}
        # y = x0 + x1 + x2
        # self.outputs = {'Out': y}
    def init_kernel_type(self):
        self.use_mkldnn = True

    def test_cpu_inplace(self):
        place = core.CPUPlace()
        for inplace in [True]:
            self.check_with_place(place, inplace)

    def create_lod_tensor(self, scope, place, var_name):
        var = scope.var(var_name)
        w_tensor = var.get_tensor()
        # w = np.random.random((3, 4)).astype(self.dtype)
        w = 0.33
        w_tensor.set(w, place)
        print("w_tensor id is ", id(w_tensor))
        return var

    def check_with_place(self, place, inplace):
        def __assert_close(tensor, np_array, msg, atol=1e-4):
            self.assertTrue(
                np.allclose(
                    np.array(tensor), np_array, atol=atol), msg)

        scope = core.Scope()
        if inplace:
            self.create_lod_tensor(scope, place, "x1")
            # self.create_lod_tensor(scope, place, "x2")
            out = scope.var("x1").get_tensor()
            print(self.use_mkldnn)
            print("outid is ", id(out))
            out_name = "x1"
            # create and run sum operator
            sum_op = Operator("sum", X=["x1"], Out=out_name)
            sum_op.run(scope, place)
            out_t = np.array(out)
            print(out_t)
            self.assertTrue(
                np.allclose(
                    np.array(0.33), out_t, atol=1e-4), 'y == out')


if __name__ == '__main__':
    unittest.main()
