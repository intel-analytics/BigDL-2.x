#
# Copyright 2018 Analytics Zoo Authors.
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
#
from zoo.serving.client import InputQueue, OutputQueue
import cv2
import os
import json
import yaml
import time
import subprocess
import sys
from multiprocessing import Process
from optparse import OptionParser


# To enqueue all image in specified directory and validate them using label txt
# using 8 threads to accelerate enqueueing image
if __name__ == "__main__":

    def inqueue_part(in_api: InputQueue, base_path, path_list):
        print("this thread got ", len(path_list), " images")
        for p in path_list:
            if not p.endswith("jpeg") and not p.endswith("JPEG"):
                continue
            img = cv2.imread(os.path.join(base_path, p))
            img = cv2.resize(img, (224, 224))
            input_api.enqueue_image(p, img)
    # params are set here, only need to set model and image path

    parser = OptionParser()
    parser.add_option("--img_path", type=str, dest="img_path",
                      help="The path of images you want to validate")

    parser.add_option("--img_num", type=int, dest="img_num",
                      help="The total number of images you validate")
    (options, args) = parser.parse_args(sys.argv)

    val_img_path = options.img_path
    total_img_num = options.img_num

    val_txt = os.path.join(val_img_path, "val.txt")

    path_list = os.listdir(val_img_path)

    input_api = InputQueue()

    # push image in queue
    top1_dict = {}
    top5_dict = {}
    # for p in path_list:
    #     if not p.endswith(format_wanted):
    #         continue
    #     img = cv2.imread(os.path.join(val_img_path, p))
    #     img = cv2.resize(img, (224, 224))
    #     redis_queue.enqueue_image(p, img)
    piece_len = int(len(path_list) / 8)
    s = b = 0
    procs = []
    for i in range(8):
        a = s
        s += piece_len
        if i == 7:
            b = len(path_list)
        else:
            b = s
        proc = Process(target=inqueue_part,
                       args=(input_api, val_img_path, path_list[a:b]))
        procs.append(proc)
        proc.start()
    for proc in procs:
        proc.join()

    # after image all pushed, start check from queue
    # get result from queue, wait until all written
    res_list = None
    while not res_list or len(res_list) < total_img_num:
        time.sleep(5)
        res_list = input_api.db.keys('result:*')
        print("Current records in Redis:", len(res_list))

    # prepare for validation, store result to dict
    output_api = OutputQueue()
    res_dict = output_api.dequeue()

    for uri in res_dict.keys():

        tmp_list = json.loads(res_dict[uri])
        top1_s, top5_s = set(), set()
        top1_s.add(tmp_list[0][0])
        for i in range(len(tmp_list)):
            top5_s.add(tmp_list[i][0])
        top5_dict[uri] = top5_s
        top1_dict[uri] = top1_s

    total, top1, top5 = 0, 0, 0

    # open label txt file and count the validation result
    with open(val_txt) as f:
        for line in f:
            line = line.strip().split(' ')
            img_uri, img_cls = line[0], int(line[1])
            if img_cls in top1_dict[img_uri]:
                top1 += 1
            if img_cls in top5_dict[img_uri]:
                top5 += 1
            total += 1
    print("top 1 accuracy is ", float(top1) / total)
    print("top 5 accuracy is ", float(top5) / total)

    # shutdown serving and re-initialize all
    # serving_process.terminate()
