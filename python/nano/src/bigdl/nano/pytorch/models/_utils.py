from torch import nn


def set_parameter_requires_grad(model: nn.Module, exception: str):
    """
    Freeze model parameters, except for the "exception" layer

    :param model: (nn.Module) Target model.
    :param exception: (str) The name of the layer that does not need to be frozen.
    :return:
    """

    for name, param in model.named_parameters():
        if exception not in name:
            param.requires_grad = False


class BackboneModule(nn.Module):
    """
    A model with the top layer removed, and with a new method `get_output_size` to retrieve the
    output channel size of the model.

    :param model_without_top: (nn.Module) A Model with the top layer removed.
    :param output_size: (int) Output channel size of the model.
    """

    def __init__(self, model_without_top: nn.Module, output_size: int):
        super(BackboneModule, self).__init__()
        self.net = model_without_top
        self.output_size = output_size

    def forward(self, x):
        return self.net(x)

    def get_output_size(self):
        """
        Returns the output channel size of the model.

        :return: (int) output channel size of the model.
        """

        return self.output_size


class SqueezeNetWithoutTopLayer(BackboneModule):
    """
    SqueezeNet model with the top convolution layer removed, and with a new method
    `get_output_size` to retrieve the output channel size of the model.
    """
    def forward(self, x):
        x = self.net.features(x)
        x = self.net.classifier(x)
        return x
