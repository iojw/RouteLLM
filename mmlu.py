import pandas as pd


FILENAMES = {
    "codellama": "lm_eval/codellama_mmlu/pretrained__codellama__CodeLlama-34b-Instruct-hf,tensor_parallel_size__1,dtype__auto,gpu_memory_utilization__0.9_mmlu_miscellaneous.jsonl",
    "llama-chat": "lm_eval/llama_mmlu/pretrained__meta-llama__Llama-2-70b-chat-hf,tensor_parallel_size__4,dtype__auto,gpu_memory_utilization__0.9_mmlu_miscellaneous.jsonl",
    "metamath": "lm_eval/metamath_mmlu/pretrained__meta-math__MetaMath-70B-V1.0,tensor_parallel_size__4,dtype__auto,gpu_memory_utilization__0.9_mmlu_miscellaneous.jsonl",
}


mmlu_df = pd.read_csv("embed_v1.csv")
mmlu_df = mmlu_df[mmlu_df["source"] == "mmlu"]
mmlu_prompt = mmlu_df["prompt"]

results = []

for model_name, filename in FILENAMES.items():
    cnt = 0
    df = pd.read_json(filename)
    doc = df["doc"]
    for i in range(len(doc)):
        query = doc[i]["question"]
        for j in range(len(mmlu_df)):
            if query in mmlu_prompt[j]:
                prompt_id = mmlu_df["prompt_id"][j]
                acc = df["acc"][i]
                results.append((model_name, prompt_id, acc))
                break

df = pd.DataFrame(results, columns=["model", "prompt_id", "score"])
df.to_csv("mmlu.csv", index=False)
