import sys
from os import listdir
from shutil import copyfile
import os
import random

# usage: python sample_training_data.py ${train_data_dir} ${sampled_output_dir} ${sample_ratio} 

path = sys.argv[1]
ratio = float(sys.argv[3])
output_path = sys.argv[2]
files = listdir(path)

sampled_files = []
for f in files:
    if random.uniform(0, 1) < ratio:
        sampled_files.append(f)
        
if not os.path.exists(output_path):
    os.mkdir(output_path)

for f in sampled_files:
    copyfile(os.path.join(path, f), os.path.join(output_path, f))

