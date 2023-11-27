from abc import abstractmethod
from datasets import load_dataset
import random
from collections import Counter
import math
import samples

class Classifier:
    @abstractmethod
    def is_code_prompt(self, prompt: str) -> bool:
        pass


class RandomClassifier(Classifier):
    def is_code_prompt(self, prompt: str) -> bool:
        return bool(random.getrandbits(1))

class NgramClassifier:
    def __init__(self, ngram_size=2):
        self.ngram_size = ngram_size
        self.code_ngrams = Counter()
        self.language_ngrams = Counter()
        self.total_code_ngrams = 0
        self.total_language_ngrams = 0
        self.train(samples.code_samples, samples.language_samples)

    def _preprocess(self, text):
        return text.lower().strip()

    def _extract_ngrams(self, text):
        text = self._preprocess(text)
        ngrams = []
        for i in range(len(text) - self.ngram_size + 1):
            ngrams.append(text[i:i + self.ngram_size])
        return ngrams

    def train(self, code_samples, language_samples):
        for sample in code_samples:
            ngrams = self._extract_ngrams(sample)
            self.code_ngrams.update(ngrams)
            self.total_code_ngrams += len(ngrams)

        for sample in language_samples:
            ngrams = self._extract_ngrams(sample)
            self.language_ngrams.update(ngrams)
            self.total_language_ngrams += len(ngrams)

    def _calculate_ngram_probability(self, ngram, is_code):
        if is_code:
            return (self.code_ngrams[ngram] + 1) / (self.total_code_ngrams + 1)
        else:
            return (self.language_ngrams[ngram] + 1) / (self.total_language_ngrams + 1)

    def is_code_prompt(self, prompt):
        ngrams = self._extract_ngrams(prompt)
        code_prob = 0
        lang_prob = 0

        for ngram in ngrams:
            code_prob += math.log(self._calculate_ngram_probability(ngram, True))
            lang_prob += math.log(self._calculate_ngram_probability(ngram, False))
        
        return code_prob > lang_prob

# List of classifiers to use
CLASSIFIERS = [RandomClassifier, NgramClassifier]
