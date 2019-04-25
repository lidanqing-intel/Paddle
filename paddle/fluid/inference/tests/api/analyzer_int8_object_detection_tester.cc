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

#include <fstream>
#include <iostream>
#include "paddle/fluid/inference/api/paddle_analysis_config.h"
#include "paddle/fluid/inference/tests/api/tester_helper.h"

namespace paddle {
namespace inference {
namespace analysis {

void SetConfig(AnalysisConfig *cfg) {
  cfg->SetModel(FLAGS_infer_model);
  cfg->DisableGpu();
  cfg->SwitchIrOptim();
  cfg->SwitchSpecifyInputNames();
  cfg->SetCpuMathLibraryNumThreads(FLAGS_paddle_num_threads);
  cfg->EnableMKLDNN();
}

std::vector<int> LoadLod(std::ifstream &file, size_t offset, int total_images) {
  std::vector<int> lod;
  file.clear();
  file.seekg(offset);
  istream_iterator<int> fileStream(file);
  std::copy_n(fileStream, total_images, std::back_inserter(lod));
  if (file.eof()) LOG(ERROR) << name_ << ": reached end of stream";
  if (file.fail()) throw std::runtime_error(name_ + ": failed reading file.");
  return lod;
}

template <typename T>
class TensorReader {
 public:
  TensorReader(std::ifstream &file, size_t beginning_offset, std::string name)
      : file_(file), position(beginning_offset), name_(name) {}

  PaddleTensor NextBatch(std::vector<int> shape, vector<int> lod) {
    numel = std::accumulate(shape.begin(), shape.end(), size_t{1},
                            std::multiplies<size_t>());
    PaddleTensor tensor;
    tensor.name = name_;
    tensor.shape = shape;
    tensor.dtype = GetPaddleDType<T>();
    tensor.data.Resize(numel * sizeof(T));
    if (lod.empty() == False) {
      tensor.lod.clear();
      tensor.lod.push_back(lod);
    }
    file_.seekg(position);
    file_.read(static_cast<char *>(tensor.data.data()), numel * sizeof(float));
    position = file_.tellg();
    if (file_.eof()) LOG(ERROR) << name_ << ": reached end of stream";
    if (file_.fail())
      throw std::runtime_error(name_ + ": failed reading file.");
    return tensor;
  }

 protected:
  std::ifstream &file_;
  size_t position;
  std::string name_;
};

void SetInput(std::vector<std::vector<PaddleTensor>> *inputs,
              int32_t batch_size = FLAGS_batch_size, int process_images = 0) {
  std::ifstream file(FLAGS_infer_data, std::ios::binary);
  if (!file) {
    FAIL() << "Couldn't open file: " << FLAGS_infer_data;
  }

  int64_t total_images{0};
  file.read(reinterpret_cast<char *>(&total_images), sizeof(int64_t));
  LOG(INFO) << "Total images in file: " << total_images;

  auto image_beginning_offset = static_cast<size_t>(file.tellg());
  auto lod_offset_in_file =
      image_beginning_offset + sizeof(float) * total_images * 3 * 224 * 224;
  std::vector<int> lod_full_vector =
      LoadLod(file, lod_offset_in_file, total_images);
  vector<int>::const_iterator lod_first = lod_full_vector.begin();
  vector<int>::const_iterator lod_end = lod_full_vector.end();
  int sum_objects_num = std::accumulate(lod_first, lod_end, 0);
  auto labels_beginning_offset =
      lod_offset_in_file + sizeof(int) * total_images;
  auto bbox_beginning_offset =
      labels_beginning_offset + sizeof(int64_t) * sum_objects_num;
  auto difficult_beginning_offset =
      bbox_beginning_offset + sizeof(float) * sum_objects_num * 4;
  TensorReader<float> image_reader(file, image_beginning_offset, "image");
  TensorReader<int64_t> label_reader(file, labels_beginning_offset, "gt_label");
  TensorReader<float> bbox_reader(file, bbox_beginning_offset, "gt_bbox");
  TensorReader<int64_t> difficult_reader(file, difficult_beginning_offset,
                                         "gt_difficult");
  if (process_images == 0) process_images = total_images;
  auto iterations_max = process_images / batch_size;
  for (auto i = 0; i < iterations_max; i++) {
    auto images_tensor = image_reader.NextBatch({batch_size, 3, 300, 300}, {});
    std::vector<int> batch_lod =
        (lod_first + i * batch_size, lod_first + batch_size * (i + 1));
    batch_num_objects = std::accumulate(batch_lod.begin(), batch_lod.end(), 0);
    batch_lod.insert(batch_lod.begin(), 0);
    for (auto it = batch_lod.begin() + 1; it != batch_lod.end(); it++) {
      *it = *it + *(it - 1);
    }
    auto labels_tensor =
        label_reader.NextBatch({batch_num_objects, 1}, batch_lod);
    auto bbox_tensor = bbox_reader.NextBatch({batch_num_objects, 4}, batch_lod);
    auto difficult_tensor =
        difficult_reader.NextBatch({batch_num_objects, 1}, batch_lod);
    inputs->emplace_back(std::vector<PaddleTensor>{
        std::move(images_tensor), std::move(labels_tensor),
        std::move(bbox_tensor), std::move(difficult_tensor)});
  }
}

std::shared_ptr<std::vector<PaddleTensor>> GetWarmupData(
    std::ifstream &file, int64_t num_images = FLAGS_warmup_batch_size) {
  SetInput(std::vector<std::vector<PaddleTensor>> * inputs,
           FLAGS_warmup_batch_size, FLAGS_warmup_batch_size);

  auto warmup_data = std::make_shared<std::vector<PaddleTensor>>(inputs);

  return warmup_data;
}

TEST(Analyzer_int8_resnet50, quantization) {
  AnalysisConfig cfg;
  SetConfig(&cfg);

  AnalysisConfig q_cfg;
  SetConfig(&q_cfg);

  // read data from file and prepare batches with test data
  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);

  // prepare warmup batch from input data read earlier
  // warmup batch size can be different than batch size
  std::shared_ptr<std::vector<PaddleTensor>> warmup_data =
      GetWarmupData(input_slots_all);

  // configure quantizer
  q_cfg.EnableMkldnnQuantizer();
  q_cfg.mkldnn_quantizer_config()->SetWarmupData(warmup_data);
  q_cfg.mkldnn_quantizer_config()->SetWarmupBatchSize(FLAGS_warmup_batch_size);

  CompareQuantizedAndAnalysis(&cfg, &q_cfg, input_slots_all);
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
