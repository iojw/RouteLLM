import pandas as pd


FILENAMES = {
    "codellama": "lm_eval/codellama_gsm8k/pretrained__codellama__CodeLlama-34b-Instruct-hf,tensor_parallel_size__4,dtype__auto,gpu_memory_utilization__0.9_gsm8k.jsonl",
    "llama-chat": "lm_eval/llama_gsm8k/pretrained__meta-llama__Llama-2-70b-chat-hf,tensor_parallel_size__4,dtype__auto,gpu_memory_utilization__0.9_gsm8k.jsonl",
    "metamath": "lm_eval/metamath_gsm8k/pretrained__meta-math__MetaMath-70B-V1.0,tensor_parallel_size__4,dtype__auto,gpu_memory_utilization__0.9_gsm8k.jsonl",
}

gsm_df = pd.read_csv("embed_v1.csv")
gsm_df = gsm_df[gsm_df["source"] == "gsm8k"]
gsm_prompt = gsm_df["prompt"].reset_index(drop=True)

results = []

for model_name, filename in FILENAMES.items():
    cnt = 0
    df = pd.read_json(filename)
    doc = df["doc"]
    for i in range(len(doc)):
        query = doc[i]["question"]
        for j in range(len(gsm_df)):
            if query in gsm_prompt.iloc[j]:
                prompt_id = gsm_df["prompt_id"].iloc[j]
                acc = df["exact_match"][i]
                results.append((model_name, prompt_id, acc))
                break

df = pd.DataFrame(results, columns=["model", "prompt_id", "score"])
df.to_csv("gsm8k.csv", index=False)
