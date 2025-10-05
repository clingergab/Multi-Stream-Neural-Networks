"""
Multi-Channel container modules - mirrors PyTorch's container.py but for dual-stream processing.
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
    "MCSequential",
    "MCReLU",
]


class MCReLU(nn.Module):
    """
    Multi-Channel ReLU activation function.
    
    Applies ReLU activation separately to color and stream2 pathways,
    following the dual-stream pattern for compatibility with MCSequential.
    
    Args:
        inplace: If set to True, will do this operation in-place. Default: False
        
    Shape:
        - Stream1 Input: Any tensor shape
        - Stream2 Input: Any tensor shape  
        - Stream1 Output: Same shape as stream1 input
        - Stream2 Output: Same shape as stream2 input
        
    Examples::
        >>> m = MCReLU()
        >>> stream1_input = torch.randn(2, 3, 4, 4)
        >>> stream2_input = torch.randn(2, 1, 4, 4)
        >>> stream1_output, stream2_output = m(stream1_input, stream2_input)
    """
    
    __constants__ = ['inplace']
    inplace: bool
    
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
    
    def forward(self, stream1_input: Tensor, stream2_input: Tensor) -> tuple[Tensor, Tensor]:
        """Apply ReLU to both pathways."""
        return F.relu(stream1_input, inplace=self.inplace), F.relu(stream2_input, inplace=self.inplace)
    
    def forward_stream1(self, stream1_input: Tensor) -> Tensor:
        """Apply ReLU to stream1 pathway only."""
        return F.relu(stream1_input, inplace=self.inplace)
    
    def forward_stream2(self, stream2_input: Tensor) -> Tensor:
        """Apply ReLU to stream2 pathway only."""
        return F.relu(stream2_input, inplace=self.inplace)
    
    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


class MCSequential(Module):
    r"""A multi-channel sequential container.

    Modules will be added to it in the order they are passed in the
    constructor. Alternatively, an ``OrderedDict`` of modules can be
    passed in. The ``forward()`` method of ``MCSequential`` accepts two
    inputs (stream1 and stream2) and forwards them through the contained modules.

    MCSequential is an extension of PyTorch's Sequential for dual-stream multi-channel processing.
    
    All modules within MCSequential must support dual-stream input/output:
    - Each module takes two inputs (stream1_input, stream2_input)
    - Each module returns two outputs (stream1_output, stream2_output)
    - Each module must have forward_stream1() and forward_stream2() methods
      
    This design maintains the clean dual-stream architecture where MCSequential
    extends nn.Sequential behavior to multi-channel processing.
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
    def __getitem__(self, idx: Union[slice, int]) -> Union["MCSequential", Module]:
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

    def __add__(self, other) -> "MCSequential":
        if isinstance(other, MCSequential):
            ret = MCSequential()
            for layer in self:
                ret.append(layer)
            for layer in other:
                ret.append(layer)
            return ret
        else:
            raise ValueError(
                "add operator supports only objects "
                f"of MCSequential class, but {str(type(other))} is given."
            )

    def pop(self, key: Union[int, slice]) -> Module:
        v = self[key]
        del self[key]
        return v

    def __iadd__(self, other) -> Self:
        if isinstance(other, MCSequential):
            offset = len(self)
            for i, module in enumerate(other):
                self.add_module(str(i + offset), module)
            return self
        else:
            raise ValueError(
                "add operator supports only objects "
                f"of MCSequential class, but {str(type(other))} is given."
            )

    def __mul__(self, other: int) -> "MCSequential":
        if not isinstance(other, int):
            raise TypeError(
                f"unsupported operand type(s) for *: {type(self)} and {type(other)}"
            )
        elif other <= 0:
            raise ValueError(
                f"Non-positive multiplication factor {other} for {type(self)}"
            )
        else:
            combined = MCSequential()
            offset = 0
            for _ in range(other):
                for module in self:
                    combined.add_module(str(offset), module)
                    offset += 1
            return combined

    def __rmul__(self, other: int) -> "MCSequential":
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

    def forward(self, stream1_input, stream2_input):
        """
        Forward pass through the sequential container.
        
        Each module must accept dual inputs (stream1_input, stream2_input) 
        and return dual outputs (stream1_output, stream2_output).
        
        This ensures compatibility with multi-channel modules like MCConv2d, 
        MCBatchNorm2d, MCReLU, etc.
        """
        for module in self:
            stream1_input, stream2_input = module(stream1_input, stream2_input)
        return stream1_input, stream2_input

    def forward_stream1(self, stream1_input):
        """Forward pass through stream1 pathway only."""
        for module in self:
            stream1_input = module.forward_stream1(stream1_input)
        return stream1_input
    
    def forward_stream2(self, stream2_input):
        """Forward pass through stream2 pathway only."""
        for module in self:
            stream2_input = module.forward_stream2(stream2_input)
        return stream2_input

    def append(self, module: Module) -> "MCSequential":
        r"""Append a given module to the end.

        Args:
            module (nn.Module): module to append
        """
        self.add_module(str(len(self)), module)
        return self

    def insert(self, index: int, module: Module) -> "MCSequential":
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

    def extend(self, sequential) -> "MCSequential":
        for layer in sequential:
            self.append(layer)
        return self
