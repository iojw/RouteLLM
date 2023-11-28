import json
import textwrap

from vllm import LLM, SamplingParams


with open('preprocess/gsm.json') as f:
    data = json.load(f)


formatted_data = []
for query in data[:10]:
    formatted_query = textwrap.dedent(f"""
        [INST] Determine whether the user query falls into one of the following categories:
        1. Programming: Queries about programming languages, libraries, and tools.
        2. Math: Queries about math, statistics, and data science.
        3. None: Anything that does not fall into the above categories.
        Your choice should be wrapped by “[[” and “]]”. For example, “[[3. None]]”.

        [USER QUERY] {query!r}

        [ANSWER]"""
    )
    formatted_query = formatted_query.lstrip()
    formatted_data.append(formatted_query)

# llm = LLM("meta-llama/Llama-2-7b-chat-hf")
llm = LLM("./output/")
sampling_params = SamplingParams(max_tokens=128, stop=["]]", "]]."])
outputs = llm.generate(formatted_data, sampling_params)
for query, output in zip(formatted_data, outputs):
    print(f"{query!r} {output.outputs[0].text!r}")
