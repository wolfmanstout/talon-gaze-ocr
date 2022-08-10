from typing import Callable, Hashable, Sequence, Optional, Union, TypeVar

__author__: str
__license__: str
__version__: str

_StringType = Sequence[Hashable]
_S1 = TypeVar("_S1")
_S2 = TypeVar("_S2")

def jaro_similarity(
    s1: _S1, s2: _S2, *,
    processor: Optional[Callable[[Union[_S1, _S2]], _StringType]] = None,
    score_cutoff: Optional[float] = 0) -> float: ...

def jarowinkler_similarity(
    s1: _S1, s2: _S2, *,
    prefix_weight: float = 0.1,
    processor: Optional[Callable[[Union[_S1, _S2]], _StringType]] = None,
    score_cutoff: Optional[float] = 0) -> float: ...
