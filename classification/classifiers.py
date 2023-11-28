from abc import abstractmethod
from datasets import load_dataset
import random

import os
import openai
from collections import Counter
import math
import samples

class Classifier:
    @abstractmethod
    def is_code_prompt(self, prompt: str) -> bool:
        pass


class RandomClassifier(Classifier):
    def __init__(self, model=None, api_base=None, api_key=None):
        self.model = "Random"

    def is_code_prompt(self, prompt: str) -> bool:
        return bool(random.getrandbits(1))

class NgramClassifier:
    def __init__(self, ngram_size=2):
        self.model = "Ngram"
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

class LLMClassifier(Classifier):
    def __init__(self, model=None, api_base=None, api_key=None):
        assert model is not None, "Please specify a model name"

        self.model = model
        self.api_base = "https://api.openai.com/v1" if api_base is None else api_base
        self.api_key = os.environ["OPENAI_API_KEY"] if api_key is None else api_key

    def is_code_prompt(self, prompt: str) -> bool:
        openai.api_key = self.api_key
        openai.api_base = self.api_base

        prompt_template = """
Please determine whether the following user prompt is related to code or not:\n\n"
{prompt}\n\n
====
If it's related to code, output "[[Y]]", if not, output "[[N]]". Please carefully follow this format.
"""
        convs = [
            {"role": "user", "content": prompt_template.format(prompt=prompt)},
        ]
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=convs,
            temperature=0,
            max_tokens=512,
        )
        output = response["choices"][0]["message"]["content"]
        if "[[Y]]" in output:
            return True
        elif "[[N]]" in output:
            return False
        else:
            raise ValueError("Invalid response.", output)

    def classify_prompt(self, prompt: str) -> bool:
        openai.api_key = self.api_key
        openai.api_base = self.api_base

        prompt_template = """
Determine whether the user query falls into one of the following categories:
1. Coding: Queries about coding, programming languages, libraries, and tools.
2. Math: Queries about math problem solving.
3. Others: Anything that does not fall into the above categories.
Your output should be wrapped by "[[" and "]]". For example, "[[3. Others]]".

[USER QUERY] {prompt!r}

[ANSWER]
"""
        convs = [
            {"role": "user", "content": prompt_template.format(prompt=prompt)},
        ]
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=convs,
            temperature=0,
            max_tokens=512,
        )
        output = response["choices"][0]["message"]["content"]

        # regex to extract the answer
        import re
        m = re.search(r"\[\[(.*)\]\]", output)
        if m is None:
            print("Invalid response.", output)
            return "format_error"
        output = m.group(1)
        if "Coding" in output:
            return "coding"
        elif "Math" in output:
            return "math"
        elif "Others" in output:
            return "others"
