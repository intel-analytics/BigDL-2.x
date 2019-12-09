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

from zoo.common import convert_to_safe_path
import argparse
import uuid
import sys
from time import sleep
from os import listdir, mkdir, rmdir, access, R_OK, W_OK
from os.path import isfile, join
import shutil


def package_path_to_text(streaming_path, file_path, batch=10, delay=3):
    """
    Package {batch} image paths into text files, such that spark
    Streaming can read these paths
    :param streaming_path:
    :param file_path:
    :param batch:
    :param delay:
    :return:
    """
    files = []

    # Convert to abs
    file_path = convert_to_safe_path(file_path, False)
    streaming_path = convert_to_safe_path(streaming_path, False)

    # check access
    if not access(file_path, R_OK):
        sys.stdout.write('Not allowed!\n')
        sys.exit()

    if not access(streaming_path, W_OK):
        sys.stdout.write('Not allowed!\n')
        sys.exit()

    for f in listdir(file_path):
        if isfile(join(file_path, f)):
            files.append(join(file_path, f) + '\n')
    index = 0
    curr = 0
    tmp_dir = join("/tmp", str(uuid.uuid4()))
    mkdir(tmp_dir)
    print("Tmp dir at " + tmp_dir)
    while curr < len(files):
        last = min(curr + batch, len(files))
        # Because spark textFileStream requires create and move
        # Write to tmp location
        batch_file_name = join(streaming_path, str(index))
        with open(join(tmp_dir, str(index) + ".txt"), "w") as text_file:
            text_file.writelines(files[curr:last])
        # Move to streaming location
        shutil.move(text_file.name,
                    batch_file_name)
        print("Writing to " + batch_file_name)
        index += 1
        curr = last
        sleep(delay)
    rmdir(tmp_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', help="Path where the images are stored")
    parser.add_argument('--streaming_path', help="Path for streaming text",
                        default="/tmp/zoo/streaming")
    args = parser.parse_args()
    package_path_to_text(args.streaming_path, args.img_path)
