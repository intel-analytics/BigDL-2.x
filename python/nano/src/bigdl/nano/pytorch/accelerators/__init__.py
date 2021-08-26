#
# Copyright 2016 The BigDL Authors.
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


import torch
_torch_save = torch.save

# To replace torch.save in ipex, you need to import and exec their __init__.py first.
from intel_pytorch_extension.ops.save import *
import intel_pytorch_extension as ipex
# And then you can replace torch.save with your customized function.
# Note that you need to temporarily store original torch.save,
# because it will be modified in ipex.ops.save.
torch.save = _torch_save

import copy
import pickle

from typing import Iterable, Dict

ITERABLE_TYPE = Iterable

DEFAULT_PROTOCOL = 2

torch_save = torch.save

def to_cpu(obj):
    # Recursively move the tensor in the output to the cpu inplace.
    if torch.is_tensor(obj):
        if obj.device.type == ipex.DEVICE:
            obj = obj.cpu()
        return

    iter_keys = obj.keys() if isinstance(obj, Dict) else range(len(obj))
    for k in iter_keys:
        if isinstance(obj, ITERABLE_TYPE):
            to_cpu(obj[k])

def nano_save(obj, f, pickle_module=pickle, pickle_protocol=DEFAULT_PROTOCOL,
              _use_new_zipfile_serialization=False):
    # Extend original `save` defined in ipex.ops.save
    # to support converting a list of xpu tensor to cpu in torch.save
    if isinstance(obj, ITERABLE_TYPE):
        obj_copy = copy.deepcopy(obj)
        to_cpu(obj_copy)
    elif isinstance(obj, torch.nn.Module):
        obj_copy = copy.deepcopy(obj).to('cpu')
    else:
        obj_copy = obj
    return torch_save(obj_copy, f, pickle_module, pickle_protocol, _use_new_zipfile_serialization)


torch.save = nano_save
