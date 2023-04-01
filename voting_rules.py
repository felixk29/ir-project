# Module with voting functions

from abc import ABC, ABCMeta, abstractmethod
from typing import Any, TypeVar

class Comparable(metaclass=ABCMeta):
    @abstractmethod
    def __lt__(self, other: Any) -> bool: ...

RankingItem = TypeVar('RankingItem', bound=Comparable)


class RankingEnsemble(ABC):

    @staticmethod
    @abstractmethod
    def combine(rankings: list[list[RankingItem]]) -> list[RankingItem]: ...

class Borda(RankingEnsemble):
    def combine(rankings): 
        item_scores = {}
        for ranking in rankings:
            modifier = len(ranking) - 1
            for item in ranking:
                current_score = item_scores.get(item, 0)
                current_score += modifier
                item_scores[item] = current_score
                modifier -= 1

        sorted_scores = sorted(item_scores.items(), key=lambda x:x[1], reverse=True)

        final_ranking_by_score = [item for item, _ in sorted_scores]

        scores_sorted_by_item_id = sorted(sorted_scores, key=lambda x:x[0], reverse=False)

        final_ranking_by_id = [score for _, score in scores_sorted_by_item_id]

        return final_ranking_by_score, final_ranking_by_id

class Copeland(RankingEnsemble):
    def combine(rankings):
        INF = 1e9

        ranking_positions = []

        items_set = set()
        for ranking in rankings:
            for item in ranking:
                items_set.add(item)
            
        items = list(items_set)

        item_number = len(items)

        for ranking in rankings:
            current_dict = {}
            for i in range(len(ranking)):
                item = ranking[i]
                current_dict[item] = i

            ranking_positions.append(current_dict)

        score_matrix = []
        for i in range(item_number):
            score_matrix.append([])
            for j in range(item_number):
                if i == j: 
                    score_matrix[i].append(0)
                else: 
                    balance = 0 

                    for positions in ranking_positions:
                        if positions.get(items[i], INF) < positions.get(items[j], INF):
                            balance += 1
                        else:
                            balance -= 1
                    
                    if balance > 0:
                        score_matrix[i].append(2)
                    elif balance == 0:
                        score_matrix[i].append(1)
                    else:
                        score_matrix[i].append(0)

        item_scores = {}
        for i in range(item_number):
            score = 0 
            for j in score_matrix[i]:
                score += j

            item_scores[items[i]] = score 

        sorted_scores = sorted(item_scores.items(), key=lambda x:x[1], reverse=True)

        final_ranking_by_score = [item for item, _ in sorted_scores]

        scores_sorted_by_item_id = sorted(sorted_scores, key=lambda x:x[0], reverse=False)

        final_ranking_by_id = [score for _, score in scores_sorted_by_item_id]

        return final_ranking_by_score, final_ranking_by_id


