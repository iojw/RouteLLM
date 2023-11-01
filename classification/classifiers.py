from abc import abstractmethod
import random

import openai

class Classifier:
    @abstractmethod
    def is_code_prompt(self, prompt: str) -> bool:
        pass


class RandomClassifier(Classifier):
    def is_code_prompt(self, prompt: str) -> bool:
        return bool(random.getrandbits(1))


class GPTClassifier(Classifier):
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model

    def is_code_prompt(self, prompt: str) -> bool:
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
            raise ValueError("Invalid response from GPT-3", output)

# List of classifiers to use
CLASSIFIERS = [RandomClassifier, GPTClassifier]
