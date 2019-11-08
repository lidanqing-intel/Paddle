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

template <typename T>
static void Split2DType(const std::string &str, char sep, std::vector<T> *v) {
  std::vector<std::string> pieces;
  split(str, sep, &pieces);
  std::transform(pieces.begin(), pieces.end(), std::back_inserter(*v),
                 [](const std::string &v) {
                   return convert<T>(v, [](const std::string &item) {
                     return std::stoll(item);
                   });
                 });
}

// Parse tensor from string
template <typename T>
void ParseTensor(const std::string &field, paddle::PaddleTensor *tensor) {
  std::vector<std::string> data;

  split(field, ':', &data);
  if (data.size() < 2) {
    LOG(ERROR) << "size of each tensor string data should be no shorter than 2 "
                  "shape:data !";
    return;
  }

  std::vector<int> shape;
  Split2DType<int>(data[0], ' ', &shape);

  std::string mat_str = data[1];

  std::vector<T> mat;
  Split2DType<T>(mat_str, ' ', &mat);

  tensor->shape = shape;
  auto size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()) *
      sizeof(T);
  tensor->data.Resize(size);
  std::copy(mat.begin(), mat.end(), static_cast<T *>(tensor->data.data()));
  tensor->dtype = GetPaddleDType<T>();
}

// data: src_ids ;  pos_ids ; sent_ids ; input_mask
// Parse input tensors from string
bool ParseLine(const std::string &line,
               std::vector<paddle::PaddleTensor> *tensors) {
  std::vector<std::string> fields;
  split(line, ';', &fields);

  if (fields.size() < 4) {
    LOG(ERROR) << "fields.size() should be no shorter than 7";
    return false;
  }
  tensors->clear();
  tensors->reserve(4);

  int i = 0;
  // src_ids
  paddle::PaddleTensor src_ids;
  ParseTensor<int64_t>(fields[i++], &src_ids);
  src_ids.name = "placeholder_0";
  tensors->push_back(src_ids);

  // pos_ids
  paddle::PaddleTensor pos_ids;
  ParseTensor<int64_t>(fields[i++], &pos_ids);
  pos_ids.name = "placeholder_1";
  tensors->push_back(pos_ids);

  // sent_ids
  paddle::PaddleTensor sent_ids;
  ParseTensor<int64_t>(fields[i++], &sent_ids);
  sent_ids.name = "placeholder_2";
  tensors->push_back(sent_ids);

  // input_mask
  paddle::PaddleTensor input_mask;
  ParseTensor<float>(fields[i++], &input_mask);
  input_mask.name = "placeholder_3";
  tensors->push_back(input_mask);

  return true;
}

bool LoadInputData(std::vector<std::vector<paddle::PaddleTensor>> *inputs) {
  if (FLAGS_infer_data.empty()) {
    LOG(ERROR) << "please set input data path";
    return false;
  }

  std::ifstream fin(FLAGS_infer_data);
  std::string line;

  int linenum = 0;
  while (std::getline(fin, line)) {
    std::vector<paddle::PaddleTensor> feed_data;
    if (!ParseLine(line, &feed_data)) {
      LOG(ERROR) << "Parse line[" << linenum << "] error!";
    } else {
      inputs->push_back(std::move(feed_data));
    }
    linenum++;
  }

  LOG(INFO) << "Load " << linenum << " samples from " << FLAGS_infer_data;
  return true;
}

TEST(Analyzer_int8_ernie_comparison_tester, quantization) {
  AnalysisConfig cfg;
  SetConfig(&cfg);

  AnalysisConfig q_cfg;
  SetConfig(&q_cfg);

  // read data from file and prepare batches with test data
  std::vector<std::vector<PaddleTensor>> input_slots_all;
  auto success = LoadInputData(&input_slots_all);
  std::cout << "size of input_slots_all: " << input_slots_all.size()
            << std::endl;
  std::cout << "size of input_slots_all[0]: " << input_slots_all[0].size()
            << std::endl;
  if (success == false) {
    LOG(ERROR) << "input data is wrong";
  }

  // prepare warmup batch from input data read earlier
  // warmup batch size can be different than batch size
  // std::shared_ptr<std::vector<PaddleTensor>> warmup_data =
  //     GetWarmupData(input_slots_all);

  // configure quantizer
  q_cfg.EnableMkldnnQuantizer();
  // q_cfg.mkldnn_quantizer_config()->SetWarmupData(warmup_data);
  // q_cfg.mkldnn_quantizer_config()->SetWarmupBatchSize(FLAGS_warmup_batch_size);

  CompareQuantizedAndAnalysis(&cfg, &q_cfg, input_slots_all);
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
