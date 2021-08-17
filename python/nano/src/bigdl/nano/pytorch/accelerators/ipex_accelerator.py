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


from typing import Union, Dict, Any, Optional
import logging
import subprocess
import intel_pytorch_extension as ipex

import torch
from torch.optim import Optimizer

import pytorch_lightning as pl
from pytorch_lightning.plugins import SingleDevicePlugin
from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.plugins.training_type import TrainingTypePlugin
from pytorch_lightning.plugins.precision import PrecisionPlugin, MixedPrecisionPlugin
from pytorch_lightning.utilities.exceptions import MisconfigurationException


_STEP_OUTPUT_TYPE = Union[torch.Tensor, Dict[str, Any]]


def check_avx512():
    cmd = "lscpu | grep avx512"
    try:
        subprocess.check_output(cmd, shell=True)
        return True
    except subprocess.CalledProcessError:
        logging.warning("avx512 disabled, fall back to non-ipex mode.")
        return False


class IPEXAccelerator(Accelerator):
    """ Accelerator for XPU devices. """

    def __init__(
        self,
        precision_plugin: PrecisionPlugin = PrecisionPlugin(),
        training_type_plugin: TrainingTypePlugin = SingleDevicePlugin(
            torch.device(ipex.DEVICE)),
        enable_bf16=False,
    ) -> None:
        """

        Args:
            precision_plugin: the plugin to handle precision-specific parts
            training_type_plugin: the plugin to handle different training routines
        """
        if enable_bf16:
            # Automatically mix precision
            ipex.enable_auto_mixed_precision(mixed_dtype=torch.bfloat16)

        self.device = ipex.DEVICE
        if not check_avx512():
            logging.warning("Enable ipex in a cpu instruction set"
                            " without avx512 may cause some random error.")
            self.device = 'cpu'
            training_type_plugin = SingleDevicePlugin(
                torch.device(self.device))

        super().__init__(precision_plugin=precision_plugin,
                         training_type_plugin=training_type_plugin)

    def setup(self, trainer: 'pl.Trainer', model: 'pl.LightningModule') -> None:
        """
        Raises:
            MisconfigurationException:
                If AMP is used with XPU.

        TODO: Supprot mix precision
        """
        if isinstance(self.precision_plugin, MixedPrecisionPlugin):
            raise MisconfigurationException(
                "amp + xpu is not supported. Please use a GPU option")

        return super().setup(trainer, model)

    def training_step_end(self, output: _STEP_OUTPUT_TYPE) -> _STEP_OUTPUT_TYPE:
        """
        For ipex xpu tensor do not support `tensor.storage()` right now,
        which is a required operation by pytorch_lightning,
        so just move output to cpu to store it, and move it back when doing backward.
        """
        if isinstance(output, torch.Tensor):
            output = output.cpu()
        elif isinstance(output, Dict):
            output = {key: xpu_tensor.cpu()
                      for key, xpu_tensor in output.items()}

        return super().training_step_end(output)

    def test_step_end(self, output: Optional[_STEP_OUTPUT_TYPE]) -> \
            Optional[_STEP_OUTPUT_TYPE]:
        """A hook to do something at the end of the test step

        Args:
            output: the output of the test step
        """
        if isinstance(output, torch.Tensor):
            output = output.cpu()
        elif isinstance(output, Dict):
            output = {key: xpu_tensor.cpu()
                      for key, xpu_tensor in output.items()}

        return super().test_step_end(output)

    def validation_step_end(self, output: Optional[_STEP_OUTPUT_TYPE]) -> \
            Optional[_STEP_OUTPUT_TYPE]:
        """A hook to do something at the end of the validation step

        Args:
            output: the output of the validation step
        """
        if isinstance(output, torch.Tensor):
            output = output.cpu()
        elif isinstance(output, Dict):
            output = {key: xpu_tensor.cpu()
                      for key, xpu_tensor in output.items()}

        return super().validation_step_end(output)

    def backward(self,  # type: ignore
                 closure_loss: torch.Tensor,
                 *args,
                 **kwargs) -> torch.Tensor:
        """
        Moving back loss to xpu device
        """
        closure_loss = closure_loss.to(self.device)
        return super().backward(
            closure_loss,
            *args,
            **kwargs,
        )
