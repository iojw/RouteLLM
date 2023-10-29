from abc import abstractmethod
import random


class Classifier:
    @abstractmethod
    def is_code_prompt(self, prompt: str) -> bool:
        pass


class RandomClassifier(Classifier):
    def is_code_prompt(self, prompt: str) -> bool:
        return bool(random.getrandbits(1))


# List of classifiers to use
CLASSIFIERS = [RandomClassifier]
