# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import gc

import tarfile
from PIL import Image, ImageEnhance, TarIO, ContainerIO
import os
from timeit import default_timer as timer
from datetime import timedelta




if __name__ == '__main__':

    fp = []
    tar = tarfile.open('utf8_full_imagenet_val.tar.gz', 'r:gz')
    print(len(tar.getmembers()))
    print(len(tar.members))
    
    for idx in range(0, 5001):
        tar_info = tar.members[idx]
        start = timer()

        fp.append(tar.extractfile(tar_info))
        if idx == 0  or idx == 1:
            idx = idx + 1
            tar.names = []
            continue

        tar_info = []
        # Test 1:
        #print(fp.read())
        # Test 2:
       img = Image.open(fp[idx])
        #fp = []
        end = timer()
        print(idx)
        print(timedelta(seconds=end - start))

    tar.close()
