#   copyright (c) 2019 paddlepaddle authors. all rights reserved.
#
# licensed under the apache license, version 2.0 (the "license");
# you may not use this file except in compliance with the license.
# you may obtain a copy of the license at
#
#     http://www.apache.org/licenses/license-2.0
#
# unless required by applicable law or agreed to in writing, software
# distributed under the license is distributed on an "as is" basis,
# without warranties or conditions of any kind, either express or implied.
# see the license for the specific language governing permissions and
# limitations under the license.

import unittest
import os
import sys
import argparse
import logging
import struct
import six
import numpy as np
import time
import paddle
import paddle.fluid as fluid
from paddle.fluid.framework import IrGraph
from paddle.fluid.contrib.slim.quantization import Qat2Int8MkldnnPass
from paddle.fluid import core

logging.basicConfig(format='%(asctime)s-%(levelname)s: %(message)s')
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size.')
    parser.add_argument(
        '--skip_batch_num',
        type=int,
        default=0,
        help='Number of the first minibatches to skip in performance statistics.'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='If used, the graph of QAT model is drawn.')
    parser.add_argument(
        '--qat_model', type=str, default='', help='A path to a QAT model.')
    parser.add_argument(
        '--save_model',
        action='store_true',
        help='If used, the QAT model will be saved after all transformations')
    parser.add_argument('--infer_data', type=str, default='', help='Data file.')
    parser.add_argument(
        '--labels', type=str, default='', help='File with labels.')
    parser.add_argument(
        '--batch_num',
        type=int,
        default=1,
        help='Number of batches to process. 0 or less means all.')
    parser.add_argument(
        '--acc_diff_threshold',
        type=float,
        default=0.01,
        help='Accepted accuracy difference threshold.')

    test_args, args = parser.parse_known_args(namespace=unittest)

    return test_args, sys.argv[:1] + args


class QatInt8NLPComparisonTest(unittest.TestCase):
    """
    Test for accuracy comparison of QAT FP32 and INT8 NLP inference.
    """

    #  def __init__(self, debug):
    #  self._debug = debug

    def _reader_creator(self, data_file=None, labels_file=None):
        assert data_file, "The dataset file is missing."
        assert labels_file, "The labels file is missing."

        def reader():
            with open(data_file, 'r') as df:
                with open(labels_file, 'r') as lf:
                    data_lines = df.readlines()
                    labels_lines = lf.readlines()
                    assert len(data_lines) == len(
                        labels_lines
                    ), "The number of labels does not match the length of the dataset."

                    for i in range(len(data_lines)):
                        print("line: {}".format(data_lines[i]))
                        data_fields = data_lines[i].split(';')
                        print("fields: {}".format(data_fields))
                        assert len(
                            data_fields
                        ) >= 2, "The number of data fields in the dataset is less than 2"
                        buffers = []
                        shape = []
                        for i in range(2):
                            data = data_fields[i].split(':')
                            print("data: {}".format(data))
                            assert len(
                                data
                            ) >= 2, "Size of data in the dataset is less than 2"
                            # Shape is stored under index 0, while data under 1
                            shape = data[0].split()
                            print("shape: {}".format(shape))
                            shape_np = np.array(shape).astype("int")
                            buffer_i = data[1].split()
                            print("buffer_i: {}".format(buffer_i))
                            buffer_np = np.array(buffer_i).astype("int64")
                            buffer_np.shape = tuple(shape_np)
                            print("buffer_np: {}".format(buffer_np))
                            buffers.append(buffer_np)
                        label = labels_lines[i].strip()
                        print("all: {}, {}, {}".format(buffers[0], buffers[1],
                                                       int(label)))
                        yield buffers[0], buffers[1], int(label)

        return reader

    def _get_batch_accuracy(self, batch_output=None, labels=None):
        total = len(batch_output)
        assert total > 0, "The batch output is empty."
        correct = 0
        for n, output in enumerate(batch_output):
            max_idx = output.index(max(output))
            if max_idx == labels[n]:
                correct += 1
        return float(correct) / float(total)

    def _prepare_for_fp32_mkldnn(self, graph):
        ops = graph.all_op_nodes()
        for op_node in ops:
            name = op_node.name()
            if name in ['depthwise_conv2d']:
                input_var_node = graph._find_node_by_name(
                    op_node.inputs, op_node.input("Input")[0])
                weight_var_node = graph._find_node_by_name(
                    op_node.inputs, op_node.input("Filter")[0])
                output_var_node = graph._find_node_by_name(
                    graph.all_var_nodes(), op_node.output("Output")[0])
                attrs = {
                    name: op_node.op().attr(name)
                    for name in op_node.op().attr_names()
                }

                conv_op_node = graph.create_op_node(
                    op_type='conv2d',
                    attrs=attrs,
                    inputs={
                        'Input': input_var_node,
                        'Filter': weight_var_node
                    },
                    outputs={'Output': output_var_node})

                graph.link_to(input_var_node, conv_op_node)
                graph.link_to(weight_var_node, conv_op_node)
                graph.link_to(conv_op_node, output_var_node)
                graph.safe_remove_nodes(op_node)

        return graph

    def _predict(self,
                 test_reader=None,
                 model_path=None,
                 batch_size=1,
                 batch_num=1,
                 skip_batch_num=0,
                 transform_to_int8=False):
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        inference_scope = fluid.executor.global_scope()
        with fluid.scope_guard(inference_scope):
            if os.path.exists(os.path.join(model_path, '__model__')):
                [inference_program, feed_target_names,
                 fetch_targets] = fluid.io.load_inference_model(model_path, exe)
            else:
                [inference_program, feed_target_names,
                 fetch_targets] = fluid.io.load_inference_model(
                     model_path, exe, 'model', 'params')

            graph = IrGraph(core.Graph(inference_program.desc), for_test=True)
            if (self._debug):
                graph.draw('.', 'qat_orig', graph.all_op_nodes())
            if (transform_to_int8):
                transform_to_mkldnn_int8_pass = Qat2Int8MkldnnPass(
                    {'fc', 'reshape2', 'transpose2'},
                    _scope=inference_scope,
                    _place=place,
                    _core=core,
                    _debug=self._debug)
                graph = transform_to_mkldnn_int8_pass.apply(graph)
            else:
                graph = self._prepare_for_fp32_mkldnn(graph)

            inference_program = graph.to_program()

            infer_accs = []
            fpses = []
            batch_times = []
            total_samples = 0
            iters = 0
            infer_start_time = time.time()
            for data in test_reader():
                if batch_num > 0 and iters >= batch_num:
                    break
                if iters == skip_batch_num:
                    total_samples = 0
                    infer_start_time = time.time()
                input1 = np.array([x[0] for x in data]).astype('int64')
                input2 = np.array([x[1] for x in data]).astype('int64')
                labels = np.array([x[2] for x in data]).astype('int64')

                start = time.time()
                out = exe.run(inference_program,
                              feed={
                                  feed_target_names[0]: input1,
                                  feed_target_names[1]: input2
                              },
                              fetch_list=fetch_targets)
                batch_time = (time.time() - start) * 1000  # in miliseconds
                batch_acc = self._get_batch_accuracy(out, labels)
                infer_accs.append(batch_acc)
                samples = len(data)
                total_samples += samples
                batch_times.append(batch_time)
                fps = samples / batch_time * 1000
                fpses.append(fps)
                iters += 1
                appx = ' (warm-up)' if iters <= skip_batch_num else ''
                _logger.info(
                    'batch {0}{5}, acc: {1:.4f}, latency: {3:.4f} ms, fps: {4:.2f}'
                    .format(iters, batch_acc, batch_time / batch_size, fps,
                            appx))

            # Postprocess benchmark data
            infer_total_time = time.time() - infer_start_time
            batch_latencies = batch_times[skip_batch_num:]
            batch_latency_avg = np.average(batch_latencies)
            latency_avg = batch_latency_avg / batch_size
            fpses = fpses[skip_batch_num:]
            fps_avg = np.average(fpses)
            acc_avg = np.mean(infer_accs)
            _logger.info('Total inference run time: {:.2f} s'.format(
                infer_total_time))

            if test_case_args.save_model:
                with fluid.scope_guard(inference_scope):
                    fluid.io.save_inference_model(
                        'transformed_qat_int8_model', feed_target_names,
                        fetch_targets, exe, inference_program)

            return acc_avg, fps_avg, latency_avg

    def _summarize_performance(self, fp32_fps, fp32_lat, int8_fps, int8_lat):
        _logger.info('--- Performance summary ---')
        _logger.info('FP32: avg fps: {0:.2f}, avg latency: {1:.4f} ms'.format(
            fp32_fps, fp32_lat))
        _logger.info('INT8: avg fps: {0:.2f}, avg latency: {1:.4f} ms'.format(
            int8_fps, int8_lat))

    def _compare_accuracy(self, fp32_acc, int8_acc, threshold):
        _logger.info('--- Accuracy summary ---')
        _logger.info(
            'Accepted accuracy drop threshold: {0}. (condition: (FP32_acc - IN8_acc) <= threshold)'
            .format(threshold))
        _logger.info('FP32: avg accuracy: {0:.4f}'.format(fp32_acc))
        _logger.info('INT8: avg accuracy: {0:.4f}'.format(int8_acc))
        assert fp32_acc > 0.0
        assert int8_acc > 0.0
        assert fp32_acc - int8_acc <= threshold

    def test_graph_transformation(self):
        if not fluid.core.is_compiled_with_mkldnn():
            return

        qat_model_path = test_case_args.qat_model
        data_path = test_case_args.infer_data
        labels_path = test_case_args.labels
        batch_size = test_case_args.batch_size
        batch_num = test_case_args.batch_num
        skip_batch_num = test_case_args.skip_batch_num
        acc_diff_threshold = test_case_args.acc_diff_threshold
        self._debug = test_case_args.debug

        _logger.info('QAT FP32 & INT8 prediction run.')
        _logger.info('QAT model: {0}'.format(qat_model_path))
        _logger.info('Dataset: {0}'.format(data_path))
        _logger.info('Labels: {0}'.format(labels_path))
        _logger.info('Batch size: {0}'.format(batch_size))
        _logger.info('Batch number: {0}'.format(batch_num))
        _logger.info('Accuracy drop threshold: {0}.'.format(acc_diff_threshold))

        #  _logger.info('--- QAT FP32 prediction start ---')
        #  val_reader = paddle.batch(
        #  self._reader_creator(data_path, labels_path), batch_size=batch_size)
        #  fp32_acc, fp32_fps, fp32_lat = self._predict(
        #  val_reader,
        #  qat_model_path,
        #  batch_size,
        #  batch_num,
        #  skip_batch_num,
        #  transform_to_int8=False)
        _logger.info('--- QAT INT8 prediction start ---')
        val_reader = paddle.batch(
            self._reader_creator(data_path, labels_path), batch_size=batch_size)
        int8_acc, int8_fps, int8_lat = self._predict(
            val_reader,
            qat_model_path,
            batch_size,
            batch_num,
            skip_batch_num,
            transform_to_int8=True)

        self._summarize_performance(fp32_fps, fp32_lat, int8_fps, int8_lat)
        self._compare_accuracy(fp32_acc, int8_acc, acc_diff_threshold)


if __name__ == '__main__':
    global test_case_args
    test_case_args, remaining_args = parse_args()
    unittest.main(argv=remaining_args)
