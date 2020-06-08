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

import argparse
import cv2

from zoo.common.nncontext import init_nncontext
from zoo.models.image.objectdetection import *


sc = init_nncontext("Object Detection Example", redirect_spark_log=False)

import pandas as pd
import numpy as np

def load_mat():
    Ymats=dict()
    with open('/home/ding/proj/deepglo/datasets/ymat_asiainfo.npy', 'rb') as f:
        for kpi_name in ['kpi0','kpi1','kpi2','kpi3','kpi4','kpi5','kpi6','kpi7']:
    	    Ymats[kpi_name] = np.load(f)
    return Ymats

# config = {
#     'epochs': 1,
#     "lr": 0.001,
#     "lstm_1_units": 16,
#     "dropout_1": 0.2,
#     "lstm_2_units": 10,
#     "dropout_2": 0.2,
#     "batch_size": 32,
# }
# from zoo.automl.model.VanillaLSTM import *
# from zoo.automl.feature.time_sequence import TimeSequenceFeatureTransformer
# train_data = pd.DataFrame(data=np.random.randn(64, 4))
# val_data = pd.DataFrame(data=np.random.randn(16, 4))
# test_data = pd.DataFrame(data=np.random.randn(16, 4))
#
# future_seq_len = 1
# past_seq_len = 6
# tsft = TimeSequenceFeatureTransformer()
# x_train, y_train = tsft._roll_train(train_data,past_seq_len=past_seq_len,future_seq_len=future_seq_len)
# model = VanillaLSTM(check_optional_config=False, future_seq_len=future_seq_len)
# model.fit_eval(x_train,y_train,**config)


from zoo.automl.model.DTCNMF.DTCNMF_pytorch import DTCNMFPytorch
config = {
    'y_iters': 1,
    "init_epochs": 1,
    "max_FX_epoch": 1,
    "max_TCN_epoch": 1
}
model = DTCNMFPytorch()
Ymats = load_mat()
Ymat = Ymats["kpi2"]
t = model.fit_eval(Ymat, **config)




parser = argparse.ArgumentParser()
parser.add_argument('model_path', help="Path where the model is stored")
parser.add_argument('img_path', help="Path where the images are stored")
parser.add_argument('output_path', help="Path to store the detection results")
parser.add_argument("--partition_num", type=int, default=1, help="The number of partitions")


def predict(model_path, img_path, output_path, partition_num):
    model = ObjectDetector.load_model(model_path)
    image_set = ImageSet.read(img_path, sc, image_codec=1, min_partitions=partition_num)
    output = model.predict_image_set(image_set)

    config = model.get_config()
    visualizer = Visualizer(config.label_map(), encoding="jpg")
    visualized = visualizer(output).get_image(to_chw=False).collect()
    for img_id in range(len(visualized)):
        cv2.imwrite(output_path + '/' + str(img_id) + '.jpg', visualized[img_id])


if __name__ == "__main__":
    args = parser.parse_args()
    predict(args.model_path, args.img_path, args.output_path, args.partition_num)

print("finished...")
sc.stop()
