from abc import abstractmethod
import random

import os
import openai

class Classifier:
    @abstractmethod
    def is_code_prompt(self, prompt: str) -> bool:
        pass


class RandomClassifier(Classifier):
    def __init__(self, model=None, api_base=None, api_key=None):
        self.model = "Random"

    def is_code_prompt(self, prompt: str) -> bool:
        return bool(random.getrandbits(1))


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
            {"role": "system", "content": """
Please determine whether the following user prompt is related to code or not. If it's related to code, output "[[Y]]", if not, output "[[N]]". Please carefully follow this format.
"""},
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
