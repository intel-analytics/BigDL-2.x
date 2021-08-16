import torch
_torch_save = torch.save

# To replace torch.save in ipex, you need to import and exec their __init__.py first.
from intel_pytorch_extension.ops.save import *

# And then you can replace torch.save with your customized function.
# Note that you need to temporarily store original torch.save,
# because it will be modified in ipex.ops.save.
torch.save = _torch_save
from .nano_specified_saver import nano_save
