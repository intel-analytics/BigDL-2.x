import torch
from torch.overrides import has_torch_function_variadic, handle_torch_function
from typing import Callable, Optional

Tensor = torch.Tensor
torch_nn_functional = torch.nn.functional
_Reduction = torch.nn._reduction


def replace_torch_function(function_name: str, replace_func: Callable):
    setattr(torch.nn.functional, function_name, replace_func)


def workaround_cross_entropy(
    input: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    size_average: Optional[bool] = None,
    ignore_index: int = -100,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> Tensor:
    if has_torch_function_variadic(input, target):
        return handle_torch_function(
            workaround_cross_entropy,
            (input, target),
            input,
            target,
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
        )
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)

    # `log_softmax(input, 1).contiguous()` makes multidimensional tensors continuous in memory
    # May fix https://github.com/intel/intel-extension-for-pytorch/issues/175#issue-972312586
    return torch_nn_functional.nll_loss(torch_nn_functional.log_softmax(input, 1).contiguous(),
                                        target, weight, None, ignore_index, None, reduction)


# Usage: append your target method and your own implements to  `replacement_dict`

# Apply ops replacements here
replacement_dict = {
    "cross_entropy": workaround_cross_entropy
}


def apply_torch_functional_replacement():
    for k, v in replacement_dict.items():
        replace_torch_function(k, v)
