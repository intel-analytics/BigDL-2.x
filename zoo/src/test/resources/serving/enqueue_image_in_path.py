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
    parser.add_option("--host", type=int, dest="host",
                          help="The total number of images you validate")
    parser.add_option("--port", type=int, dest="port",
                      help="The total number of images you validate")


    (options, args) = parser.parse_args(sys.argv)

    val_img_path = options.img_path
    total_img_num = options.img_num

    val_txt = os.path.join(val_img_path, "val.txt")

    path_list = os.listdir(val_img_path)

    input_api = InputQueue(options.host, options.port)

    output_api = OutputQueue(options.host, options.port)
    output_api.dequeue()

    # push image in queue
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
    print("image enqueue ended, total ", len(path_list))
