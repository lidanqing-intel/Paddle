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

import unittest
import numpy as np
import math
# from paddle.fluid.tests.unittests.op_test import OpTest
from paddle.fluid.tests.unittests.test_fusion_gru_op import TestFusionGRUOp
from paddle.fluid.tests.unittests.op_test import OpTest
from paddle.fluid.tests.unittests.test_gru_op import gru
from paddle.fluid.tests.unittests.test_fusion_lstm_op import fc, ACTIVATION

# class TestFusionGRUMKLDNNOp(TestFusionGRUOp):
#     def setUp(self):
#         TestFusionGRUOp.setUp(self)
#         self.attrs['use_mkldnn'] = True


def fusion_gru(
        x,  # T x M
        lod,  # 1 x N
        h0,  # N x D
        wx,  # M x 3D
        wh,  # D x 3D
        bias,  # 1 x 3D
        is_reverse,
        act_state,
        act_gate):
    return gru(fc(x, wx, bias),
               lod,
               h0,
               wh,
               np.zeros(
                   (1, wh.shape[1]), dtype='float32'),
               is_reverse,
               act_state,
               act_gate)


class TestFusionGRUMKLDNNOp(OpTest):
    def set_confs(self):
        pass

    def setUp(self):
        self.op_type = "fusion_gru"
        self.lod = [
            [2, 4, 3]
        ]  # T is the total length of the whole batch of sentences length. The lod will separate the long sequence to a batch of small sentences
        self.M = 3  # M is character feature size, like channel size. It may come from some directory
        self.D = 5  # D is the H0[0], it it like the first dimension of the output
        self.is_reverse = False
        self.with_h0 = True
        self.with_bias = True
        self.act_state = 'tanh'
        self.act_gate = 'sigmoid'
        self._cpu_only = True
        self.set_confs()

        T = sum(self.lod[0])
        N = len(self.lod[0])

        x = np.random.rand(T, self.M).astype('float32')
        wx = np.random.rand(self.M, 3 * self.D).astype('float32')
        wh = np.random.rand(self.D, 3 * self.D).astype('float32')
        bias = np.random.rand(
            1, 3 * self.D).astype('float32') if self.with_bias else np.zeros(
                (1, 3 * self.D), dtype='float32')
        h0 = np.random.rand(
            N, self.D).astype('float32') if self.with_h0 else np.zeros(
                (N, self.D), dtype='float32')

        _, _, _, hidden = fusion_gru(
            x, self.lod, h0, wx, wh, bias, self.is_reverse,
            ACTIVATION[self.act_state], ACTIVATION[self.act_gate])

        self.inputs = {'X': (x, self.lod), 'WeightX': wx, 'WeightH': wh}

        if self.with_bias:
            self.inputs['Bias'] = bias

        if self.with_h0:
            self.inputs['H0'] = h0

        self.outputs = {'Hidden': (hidden, self.lod)}

        self.attrs = {
            'activation': self.act_state,
            'gate_activation': self.act_gate,
            'is_reverse': self.is_reverse,
            'use_mkldnn': True
        }

    def test_check_output(self):
        # for use_seq in {True, False}:
        #     self.attrs['use_seq'] = use_seq
        self.check_output(check_dygraph=False)


# class TestGRUOp(OpTest):
#     lod = [[2, 4, 5]]
#     batch_size = sum(lod[0])
#     frame_size = 5
#     feature_size = 8
#     activate = ACTIVATION

#     @staticmethod
#     def seq_to_batch(lod, is_reverse):
#         idx_in_seq_list = []
#         seq_lens = lod[0]
#         seq_starts = [0]
#         for i in range(len(seq_lens)):
#             seq_starts.append(seq_starts[-1] + seq_lens[i])
#         sorted_seqs = sorted(
#             range(len(seq_lens)), lambda x, y: seq_lens[y] - seq_lens[x])
#         num_batch = seq_lens[sorted_seqs[0]]
#         for batch_idx in range(num_batch):
#             idx_in_seq = []
#             for i in range(len(seq_lens)):
#                 if seq_lens[sorted_seqs[i]] <= batch_idx:
#                     break
#                 idx = (seq_starts[sorted_seqs[i] + 1] - 1 - batch_idx
#                        ) if is_reverse else (
#                            seq_starts[sorted_seqs[i]] + batch_idx)
#                 idx_in_seq.append(idx)
#             idx_in_seq_list.append(idx_in_seq)
#         return idx_in_seq_list, sorted_seqs

#     def gru_step(self, x, h_p, w, b):
#         batch_size = x.shape[0]
#         frame_size = w.shape[0]
#         g = x + np.tile(b, (batch_size, 1))
#         w_u_r = w.flatten()[:frame_size * frame_size * 2].reshape(
#             (frame_size, frame_size * 2))
#         u_r = self.activate[self.attrs['gate_activation']](np.dot(
#             h_p, w_u_r) + g[:, :frame_size * 2])

#         u = u_r[:, :frame_size]
#         r = u_r[:, frame_size:frame_size * 2]
#         r_h_p = r * h_p
#         w_c = w.flatten()[frame_size * frame_size * 2:].reshape(
#             (frame_size, frame_size))
#         c = self.activate[self.attrs['activation']](np.dot(r_h_p, w_c) +
#                                                     g[:, frame_size * 2:])
#         g = np.hstack((u_r, c))
#         h = u * h_p + (1 - u) * c
#         return g, r_h_p, h

#     def gru(self):
#         input, lod = self.inputs['X']
#         wx = self.inputs['WeightX']
#         wh = self.inputs['WeightH']

#         b = self.inputs['Bias'] if self.inputs.has_key('Bias') else np.zeros(
#             (1, self.frame_size * 3))
#         hidden = self.outputs['Hidden']
#         idx_in_seq_list = self.idx_in_seq_list
#         h_p = self.inputs['H0'][self.sorted_seqs] if self.inputs.has_key(
#             'H0') else np.zeros((len(idx_in_seq_list[0]), self.frame_size))
#         num_batch = len(idx_in_seq_list)
#         end_idx = 0
#         for batch_idx in range(num_batch):
#             x = input[idx_in_seq_list[batch_idx]]
#             x_gru = np.dot(x, wx)
#             g, r_h_p, h = self.gru_step(x_gru, h_p, wh, b)
#             if batch_idx < (num_batch - 1):
#                 h_p = h[:len(idx_in_seq_list[batch_idx + 1])]
#             start_idx = end_idx
#             end_idx = start_idx + len(idx_in_seq_list[batch_idx])
#             hidden[idx_in_seq_list[batch_idx]] = h
#         return hidden

#     def set_data(self):
#         lod = self.lod
#         self.idx_in_seq_list, self.sorted_seqs = self.seq_to_batch(
#             lod, self.is_reverse)
#         batch_size = self.batch_size
#         frame_size = self.frame_size
#         feature_size = self.feature_size
#         #        input = np.ones((batch_size, feature_size), dtype='float32')

#         input = np.random.rand(batch_size, feature_size).astype('float32')
#         h0 = np.random.rand(len(self.idx_in_seq_list[0]),
#                             frame_size).astype('float32')
#         #        h0 = np.zeros((batch_size, frame_size), dtype='float32')
#         weightH = np.random.rand(frame_size, frame_size * 3).astype('float32')
#         #weightH = np.zeros((frame_size, frame_size * 3), dtype='float32')
#         #        weightX = np.ones((feature_size, frame_size * 3), dtype='float32')
#         weightX = np.random.rand(feature_size, frame_size * 3).astype('float32')
#         biasH = np.random.rand(1, frame_size * 3).astype('float32')
#         #biasH = np.zeros((1, frame_size * 3), dtype='float32')
#         bias = np.add(biasH, biasH)

#         self.inputs = {
#             'X': (input, lod),
#             'H0': h0,
#             'WeightX': weightX,
#             'WeightH': weightH,
#             'Bias': bias / 2
#         }

#         self.outputs = {
#             'Hidden': np.zeros(
#                 (batch_size, frame_size), dtype='float32')
#         }

#     def set_confs(self):
#         self.is_reverse = False
#         self.attrs = {
#             'activation': 'tanh',
#             'gate_activation': 'sigmoid',
#             'is_reverse': self.is_reverse,
#             'use_mkldnn': True
#         }

#     def setUp(self):
#         self.op_type = "fusion_gru"
#         self.set_confs()
#         self.set_data()
#         self.gru()

#     def test_check_output(self):
#         self.check_output(check_dygraph=(self.attrs['use_mkldnn'] == False))

# class TestGRUOpReverse(TestGRUOp):
#     def set_confs(self):
#         self.is_reverse = True
#         self.attrs = {
#             'activation': 'tanh',
#             'gate_activation': 'sigmoid',
#             'is_reverse': self.is_reverse,
#             'use_mkldnn': True
#         }

if __name__ == "__main__":
    unittest.main()
