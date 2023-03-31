# Module with voting functions

from abc import ABC, ABCMeta, abstractmethod
from typing import Any, TypeVar 

class Comparable(metaclass=ABCMeta):
    @abstractmethod
    def __lt__(self, other: Any) -> bool: ...

RankingEntry = TypeVar('RankingEntry', bound=Comparable)

class RankingEnsemble(ABC):

    @abstractmethod
    @staticmethod
    def combine(rankings: list[list[RankingEntry]]) -> list[RankingEntry]: ...

class Borda(RankingEnsemble):
    def combine(rankings): 
        pass

class Copeland(RankingEnsemble):
    def combine(rankings):
        pass
