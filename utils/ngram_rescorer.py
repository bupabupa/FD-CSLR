import json
import math
from collections import Counter
from pathlib import Path


BOS = -1
EOS = -2


class NGramRescorer:
    """Interpolated gloss n-gram scorer for CTC beam rescoring."""

    def __init__(self, sequences, order=3, vocabulary_size=1296):
        order = int(order)
        if order < 1:
            raise ValueError("order must be positive")
        if vocabulary_size < 1:
            raise ValueError("vocabulary_size must be positive")
        self.order = order
        self.vocabulary_size = int(vocabulary_size)
        self.counts = [Counter() for _ in range(order + 1)]
        self.documents = 0
        self.tokens = 0
        for sequence in sequences:
            values = tuple(int(value) for value in sequence)
            if any(value < 0 or value >= vocabulary_size for value in values):
                raise ValueError("sequence contains an out-of-range class id")
            padded = (BOS,) * (order - 1) + values + (EOS,)
            self.documents += 1
            self.tokens += len(values)
            for end in range(order - 1, len(padded)):
                for size in range(1, order + 1):
                    self.counts[size][padded[end - size + 1:end + 1]] += 1
        self.unigram_total = sum(self.counts[1].values())

    @classmethod
    def from_json(cls, path):
        with open(path, encoding="utf-8") as source:
            payload = json.load(source)
        if payload.get("source") != "D_train":
            raise ValueError("rescorer corpus must declare source=D_train")
        return cls(
            payload["sequences"],
            order=int(payload["order"]),
            vocabulary_size=int(payload["num_classes"]),
        )

    def _unigram_probability(self, token):
        count = self.counts[1].get((token,), 0)
        denominator = self.unigram_total + 0.1 * (
            self.vocabulary_size + 2
        )
        return (count + 0.1) / denominator

    def score(self, sequence):
        values = tuple(int(value) for value in sequence)
        if any(value < 0 or value >= self.vocabulary_size for value in values):
            raise ValueError("sequence contains an out-of-range class id")
        history = [BOS] * (self.order - 1)
        total = 0.0
        for token in values + (EOS,):
            probability = self._unigram_probability(token)
            for size in range(2, self.order + 1):
                context = tuple(history[-(size - 1):])
                context_count = self.counts[size - 1].get(context, 0)
                if context_count <= 0:
                    continue
                numerator = self.counts[size].get(
                    context + (token,), 0
                )
                probability = (
                    0.7 * numerator / context_count + 0.3 * probability
                )
            total += math.log(max(probability, 1e-12))
            history.append(token)
            if len(history) > self.order - 1:
                history.pop(0)
        return total


def rescorer_score_total(score, alpha):
    return float(alpha) * float(score)
