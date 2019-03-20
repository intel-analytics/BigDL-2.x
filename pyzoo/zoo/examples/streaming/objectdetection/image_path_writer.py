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
from time import sleep
from os import listdir
from os.path import isfile, join


def package_path_to_text(streaming_path, file_path, batch=10, delay=3):
    files = [join(file_path, f) + '\n' for f in listdir(file_path) if isfile(join(file_path, f))]
    index = 0
    curr = 0
    while curr < len(files):
        last = min(curr + batch, len(files))
        with open(join(streaming_path, str(index) + ".txt"), "w") as text_file:
            print("Writing to " + text_file.name)
            text_file.writelines(files[curr:last])
        index += 1
        curr = last
        sleep(delay)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', help="Path where the images are stored")
    parser.add_argument('--streaming_path', help="Path for streaming text",
                        default="/tmp/zoo/streaming")

    args = parser.parse_args()
    package_path_to_text(args.streaming_path, args.img_path)
