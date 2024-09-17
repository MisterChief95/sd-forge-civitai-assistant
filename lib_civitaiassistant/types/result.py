from dataclasses import dataclass, astuple
from typing import Generic, TypeVar


T = TypeVar("T")


@dataclass  
class Result(Generic[T]):
    value: T = None
    error: Exception = None

    def __iter__(self):
        return iter(astuple(self))
    
    def __getitem__(self, keys):
        return iter(getattr(self, k) for k in keys)
