#   copyright (c) 2019 paddlepaddle authors. all rights reserved.
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
import hashlib
import unittest
import os
import numpy as np
import time
import sys
import random
import functools
import contextlib
from PIL import Image, ImageEnhance
import math
from paddle.dataset.common import download, md5file
import tarfile

from timeit import default_timer as timer
from datetime import timedelta

random.seed(0)
np.random.seed(0)

DATA_DIM = 224
SIZE_FLOAT32 = 4
SIZE_INT64 = 8
FULL_SIZE_BYTES = 30106000008
FULL_IMAGES = 50000
TARGET_HASH = '8dc592db6dcc8d521e4d5ba9da5ca7d2'
FOLDER_NAME = "ILSVRC2012/"

img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))


def resize_short(img, target_size):
    percent = float(target_size) / min(img.size[0], img.size[1])
    resized_width = int(round(img.size[0] * percent))
    resized_height = int(round(img.size[1] * percent))
    img = img.resize((resized_width, resized_height), Image.LANCZOS)
    return img


def crop_image(img, target_size, center):
    width, height = img.size
    size = target_size
    if center == True:
        w_start = (width - size) / 2
        h_start = (height - size) / 2
    else:
        w_start = np.random.randint(0, width - size + 1)
        h_start = np.random.randint(0, height - size + 1)
    w_end = w_start + size
    h_end = h_start + size
    img = img.crop((w_start, h_start, w_end, h_end))
    return img


def process_image(img):
    img = resize_short(img, target_size=256)
    img = crop_image(img, target_size=DATA_DIM, center=True)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = np.array(img).astype('float32').transpose((2, 0, 1)) / 255
    img -= img_mean
    img /= img_std
    return img


def download_concat(cache_folder, zip_path):
    data_urls = []
    data_md5s = []
    data_urls.append(
        'https://paddle-inference-dist.bj.bcebos.com/int8/ILSVRC2012_img_val.tar.gz.partaa'
    )
    data_md5s.append('60f6525b0e1d127f345641d75d41f0a8')
    data_urls.append(
        'https://paddle-inference-dist.bj.bcebos.com/int8/ILSVRC2012_img_val.tar.gz.partab'
    )
    data_md5s.append('1e9f15f64e015e58d6f9ec3210ed18b5')
    file_names = []
    print("Downloading full ImageNet Validation dataset ...")
    for i in range(0, len(data_urls)):
        download(data_urls[i], cache_folder, data_md5s[i])
        file_name = os.path.join(cache_folder, data_urls[i].split('/')[-1])
        file_names.append(file_name)
        print("Downloaded part {0}\n".format(file_name))
    if not os.path.exists(zip_path):
        with open(zip_path, "w+") as outfile:
            for fname in file_names:
                with open(fname) as infile:
                    outfile.write(infile.read())


def print_processbar(done, total):
    done_filled = done * '='
    empty_filled = (total - done) * ' '
    percentage_done = done * 100 / total
    sys.stdout.write("\r[%s%s]%d%%" %
                     (done_filled, empty_filled, percentage_done))
    sys.stdout.flush()


def check_integrity(filename, target_hash):
    print('\nThe binary file exists. Checking file integrity...\n')
    md = hashlib.md5()
    count = 0
    total_parts = 50
    chunk_size = 8192
    onepart = FULL_SIZE_BYTES / chunk_size / total_parts
    with open(filename) as ifs:
        while True:
            buf = ifs.read(8192)
            if count % onepart == 0:
                done = count / onepart
                print_processbar(done, total_parts)
            count = count + 1
            if not buf:
                break
            md.update(buf)
    hash1 = md.hexdigest()
    if hash1 == target_hash:
        return True
    else:
        return False


def convert(tar_file, output_file):
    print('Converting 50000 images to binary file ...\n')
    f1 = tarfile.open(name=tar_file, mode='r:gz')
    val_info = f1.getmembers()[-1]
    val_list = f1.extractfile(val_info).read()
    lines = val_list.split('\n')
    num_images = len(lines)
    f1.close()

    f1 = tarfile.open(name=tar_file, mode='r:gz')

    f1.next()
    f1.next()

    with open(output_file, "w+b") as ofs:
        #save num_images(int64_t) to file
        ofs.seek(0)
        num = np.array(int(num_images)).astype('int64')
        ofs.write(num.tobytes())
        per_parts = 500
        full_parts = FULL_IMAGES / per_parts
        print_processbar(0, full_parts)

        fp_info = f1.next()

        idx = 0

        start = timer()
        for tar_info in f1:
            start = timer()
            img_name = tar_info.name

            if idx == 0 or idx == 1:
                idx = idx + 1
                f1.members = []
                f1.names = []
                continue

            fp = f1.extractfile(tar_info)
            #tar_info = None

            print(idx)

            img = Image.open(fp)
            #fp = None
            end = timer()
            print(timedelta(seconds=end - start))

            #            img = process_image(img)
            #            np_img = np.array(img)
            #            ofs.seek(SIZE_INT64 + SIZE_FLOAT32 * DATA_DIM * DATA_DIM * 3 * idx)
            #            ofs.write(np_img.astype('float32').tobytes())
            #            ofs.flush()
            #
            #            remove_len = (len(FOLDER_NAME))
            #
            #            img_name_prim = img_name[remove_len:]
            #
            #            matching = [s for s in lines if img_name_prim in s]
            #            _, label = matching[0].split()
            #
            #            #save label(int64_t) to ile
            #            label_int = (int)(label)
            #            np_label = np.array(label_int)
            #            ofs.seek(SIZE_INT64 + SIZE_FLOAT32 * DATA_DIM * DATA_DIM * 3 *
            #                     num_images + idx * SIZE_INT64)
            #            ofs.write(np_label.astype('int64').tobytes())
            #            ofs.flush()
            #
            #            if (idx - 1) % per_parts == 0:
            #                done = (idx - 1) / per_parts
            #                print_processbar(done, full_parts)
            #                end = timer()
            #                print(idx)
            #            i    print(timedelta(seconds=end - start))

            f1.members = []
            f1.names = []

            idx = idx + 1
    f1.close()
    print("Conversion finished.")


def run_convert():
    print('Start to download and convert 50000 images to binary file...')
    cache_folder = os.path.expanduser('~/.cache/paddle/dataset/int8/download')
    zip_path = os.path.join(cache_folder, 'full_imagenet_val.tar.gz.partaa')
    output_file = os.path.join(cache_folder, 'int8_full_val.bin')
    retry = 0
    try_limit = 3

    while not (os.path.exists(output_file) and
               os.path.getsize(output_file) == FULL_SIZE_BYTES and
               check_integrity(output_file, TARGET_HASH)):
        if os.path.exists(output_file):
            sys.stderr.write(
                "\n\nThe existing binary file is broken. Start to generate new one...\n\n".
                format(output_file))
            os.remove(output_file)
        if retry < try_limit:
            retry = retry + 1
        else:
            raise RuntimeError(
                "Can not convert the dataset to binary file with try limit {0}".
                format(try_limit))
        #download_concat(cache_folder, zip_path)
        convert(zip_path, output_file)
    print("\nSuccess! The binary file can be found at {0}".format(output_file))


if __name__ == '__main__':
    run_convert()
