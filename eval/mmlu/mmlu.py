import pandas as pd

categories = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]

results = []

for category in categories:
    FILENAMES = {
        "CodeLlama-34b-Instruct-hf": f"codellama_mmlu/pretrained__codellama__CodeLlama-34b-Instruct-hf,tensor_parallel_size__1,dtype__auto,gpu_memory_utilization__0.9_mmlu_{category}.jsonl",
        "Llama-2-70b-chat-hf": f"llama_mmlu/pretrained__meta-llama__Llama-2-70b-chat-hf,tensor_parallel_size__4,dtype__auto,gpu_memory_utilization__0.9_mmlu_{category}.jsonl",
        "MetaMath-70B-V1.0": f"metamath_mmlu/pretrained__meta-math__MetaMath-70B-V1.0,tensor_parallel_size__4,dtype__auto,gpu_memory_utilization__0.9_mmlu_{category}.jsonl",
    }

    for model_name, filename in FILENAMES.items():
        df = pd.read_json(filename)
        doc = df["doc"]
        for i in range(len(doc)):
            prompt_id = f"mmlu-{category}-{df['doc_id'][i]}"
            acc = df["acc"][i]
            results.append((model_name, prompt_id, acc))

df = pd.DataFrame(results, columns=["model", "prompt_id", "score"])

print(len(df), "rows")
df.to_csv("mmlu.csv", index=False)
