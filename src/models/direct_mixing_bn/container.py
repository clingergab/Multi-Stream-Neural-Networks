"""
Direct Mixing container modules for N-stream processing.

Provides DMSequential and DMReLU for managing N-stream operations.
"""

import operator
from collections import OrderedDict
from collections.abc import Iterator
from itertools import islice
from typing import Any, Optional, Union, overload
from typing_extensions import Self

import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn import Module
from torch._jit_internal import _copy_to_script_wrapper


__all__ = [
    "DMSequential",
    "DMReLU",
]


class DMReLU(nn.Module):
    """
    Direct Mixing ReLU activation function for N streams.

    Applies ReLU activation separately to all stream and integrated pathways.

    Args:
        inplace: If set to True, will do this operation in-place. Default: False

    Shape:
        - Stream Inputs: List of tensors of any shape
        - Integrated Input: Any tensor shape (or None)
        - Stream Outputs: List of tensors with same shape as stream inputs
        - Integrated Output: Same shape as integrated input (or None)

    Examples::
        >>> m = DMReLU()
        >>> stream_inputs = [torch.randn(2, 64, 28, 28) for _ in range(3)]
        >>> integrated_input = torch.randn(2, 64, 28, 28)
        >>> stream_outputs, integrated_output = m(stream_inputs, integrated_input)
    """

    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, stream_inputs: list[Tensor], integrated_input: Optional[Tensor] = None) -> tuple[list[Tensor], Tensor]:
        """Apply ReLU to all N pathways."""
        stream_outputs = [F.relu(stream_input, inplace=self.inplace) for stream_input in stream_inputs]

        if integrated_input is not None:
            integrated_out = F.relu(integrated_input, inplace=self.inplace)
        else:
            integrated_out = None

        return stream_outputs, integrated_out

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


class DMSequential(Module):
    r"""A direct mixing sequential container for N-stream processing.

    Modules will be added to it in the order they are passed in the
    constructor. Alternatively, an ``OrderedDict`` of modules can be
    passed in. The ``forward()`` method of ``DMSequential`` accepts
    stream_inputs (list[Tensor]) and integrated_input (Optional[Tensor])
    and forwards them through the contained modules.

    DMSequential is an extension of PyTorch's Sequential for N-stream processing.

    All modules within DMSequential must support N-stream input/output:
    - Each module takes (stream_inputs: list[Tensor], integrated_input: Optional[Tensor])
    - Each module returns (stream_outputs: list[Tensor], integrated_output: Optional[Tensor])

    This design maintains the clean N-stream architecture where DMSequential
    extends nn.Sequential behavior to direct mixing processing.
    """

    _modules: dict[str, Module]  # type: ignore[assignment]

    @overload
    def __init__(self, *args: Module) -> None:
        ...

    @overload
    def __init__(self, arg: "OrderedDict[str, Module]") -> None:
        ...

    def __init__(self, *args):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx) -> Module:
        """Get the idx-th item of the iterator."""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError(f"index {idx} is out of range")
        idx %= size
        return next(islice(iterator, idx, None))

    @_copy_to_script_wrapper
    def __getitem__(self, idx: Union[slice, int]) -> Union["DMSequential", Module]:
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx: int, module: Module) -> None:
        key: str = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx: Union[slice, int]) -> None:
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)
        # To preserve numbering
        str_indices = [str(i) for i in range(len(self._modules))]
        self._modules = OrderedDict(list(zip(str_indices, self._modules.values())))

    @_copy_to_script_wrapper
    def __len__(self) -> int:
        return len(self._modules)

    def __add__(self, other) -> "DMSequential":
        if isinstance(other, DMSequential):
            ret = DMSequential()
            for layer in self:
                ret.append(layer)
            for layer in other:
                ret.append(layer)
            return ret
        else:
            raise ValueError(
                "add operator supports only objects "
                f"of DMSequential class, but {str(type(other))} is given."
            )

    def pop(self, key: Union[int, slice]) -> Module:
        v = self[key]
        del self[key]
        return v

    def __iadd__(self, other) -> Self:
        if isinstance(other, DMSequential):
            offset = len(self)
            for i, module in enumerate(other):
                self.add_module(str(i + offset), module)
            return self
        else:
            raise ValueError(
                "add operator supports only objects "
                f"of DMSequential class, but {str(type(other))} is given."
            )

    def __mul__(self, other: int) -> "DMSequential":
        if not isinstance(other, int):
            raise TypeError(
                f"unsupported operand type(s) for *: {type(self)} and {type(other)}"
            )
        elif other <= 0:
            raise ValueError(
                f"Non-positive multiplication factor {other} for {type(self)}"
            )
        else:
            combined = DMSequential()
            offset = 0
            for _ in range(other):
                for module in self:
                    combined.add_module(str(offset), module)
                    offset += 1
            return combined

    def __rmul__(self, other: int) -> "DMSequential":
        return self.__mul__(other)

    def __imul__(self, other: int) -> Self:
        if not isinstance(other, int):
            raise TypeError(
                f"unsupported operand type(s) for *: {type(self)} and {type(other)}"
            )
        elif other <= 0:
            raise ValueError(
                f"Non-positive multiplication factor {other} for {type(self)}"
            )
        else:
            len_original = len(self)
            offset = len(self)
            for _ in range(other - 1):
                for i in range(len_original):
                    self.add_module(str(i + offset), self._modules[str(i)])
                offset += len_original
            return self

    @_copy_to_script_wrapper
    def __dir__(self):
        keys = super().__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    @_copy_to_script_wrapper
    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def forward(self, stream_inputs: list[Tensor], integrated_input: Optional[Tensor] = None) -> tuple[list[Tensor], Tensor]:
        """
        Forward pass through the sequential container.

        Each module must accept (stream_inputs: list[Tensor], integrated_input: Optional[Tensor])
        and return (stream_outputs: list[Tensor], integrated_output: Optional[Tensor]).

        This ensures compatibility with DM modules like DMConv2d,
        DMBatchNorm2d, DMReLU, etc.
        """
        for module in self:
            stream_inputs, integrated_input = module(stream_inputs, integrated_input)
        return stream_inputs, integrated_input

    def append(self, module: Module) -> "DMSequential":
        r"""Append a given module to the end.

        Args:
            module (nn.Module): module to append
        """
        self.add_module(str(len(self)), module)
        return self

    def insert(self, index: int, module: Module) -> "DMSequential":
        if not isinstance(module, Module):
            raise AssertionError(f"module should be of type: {Module}")
        n = len(self._modules)
        if not (-n <= index <= n):
            raise IndexError(f"Index out of range: {index}")
        if index < 0:
            index += n
        for i in range(n, index, -1):
            self._modules[str(i)] = self._modules[str(i - 1)]
        self._modules[str(index)] = module
        return self

    def extend(self, sequential) -> "DMSequential":
        for layer in sequential:
            self.append(layer)
        return self
