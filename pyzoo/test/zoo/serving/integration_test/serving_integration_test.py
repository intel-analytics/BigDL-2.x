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
from zoo.serving.client.utils.helpers import RedisQueue
import cv2
import os
import json
import yaml
import time
import subprocess
from multiprocessing import Process

if __name__ == "__main__":

    def inqueue_part(cli, base_path, path_list):
        print("this thread got ", len(path_list), " images")
        for p in path_list:
            if not p.endswith(format_wanted):
                continue
            img = cv2.imread(os.path.join(base_path, p))
            img = cv2.resize(img, (224, 224))
            cli.enqueue_image(p, img)
    # params are set here, only need to set model and image path
    if True:
        format_wanted = "jpeg"
        model_parent_path = "/home/litchy/pro/models"
        val_img_path = "/home/litchy/val_img"
        total_img_num = 11
    else:
        format_wanted = "JPEG"
        model_parent_path = "/root/sjm/models"
        val_img_path = "/root/sjm/val"
        total_img_num = 50000

    config_path = "./integration_test/config.yaml"
    tmp_cfg_path = "./integration_test/tmp_config.yaml"
    script_path = "./integration_test/zoo-cluster-serving.sh"
    val_txt = os.path.join(val_img_path, "val.txt")

    path_list = os.listdir(val_img_path)

    model_path_list = os.listdir(model_parent_path)

    for model_path in model_path_list:
        if not os.path.isdir(os.path.join(model_parent_path, model_path)):
            print("model path should be a directory, but " +
                  model_path + " is a file, skipped")
            continue

        # parse and modify yaml
        with open(config_path, 'r') as cfg_file:
            cfg = yaml.load(cfg_file)
            cfg['model']['path'] = os.path.join(model_parent_path, model_path)
        with open(tmp_cfg_path, 'w') as tmp_cfg_file:
            yaml.dump(cfg, tmp_cfg_file, default_flow_style=False)

        redis_queue = RedisQueue(tmp_cfg_path)
        redis_queue.db.flushall()
        # start serving
        serving_process = subprocess.Popen([script_path], shell=True)
        time.sleep(20)

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
                           args=(redis_queue, val_img_path, path_list[a:b]))
            procs.append(proc)
            proc.start()
        for proc in procs:
            proc.join()

        # after image all pushed, start check from queue
        # get result from queue, wait until all written
        res_list = None
        while not res_list or len(res_list) < total_img_num:
            time.sleep(20)
            res_list = redis_queue.db.keys('result:*')
            print("Current records in Redis:", len(res_list))

        # prepare for validation, store result to dict
        for res in res_list:
            res_dict = (redis_queue.get_results(res.decode('utf-8')))
            res_id = res_dict[b'_1'].decode('utf-8')
            res_value = res_dict[b'_2'].decode('utf-8')

            tmp_dict = json.loads(res_value)
            top1_s, top5_s = set(), set()
            top1_s.add(int(next(iter(tmp_dict))))
            for x in tmp_dict.keys():
                top5_s.add(int(x))
            top5_dict[res_id] = top5_s
            top1_dict[res_id] = top1_s

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
        serving_process.kill()
        redis_queue.db.flushall()



