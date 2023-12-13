import classifiers
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import argparse

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


def eval(args):
    results = []
    classifier = classifiers.EmbeddingClassifier(model_path="embedding_model_3.pt")
    print(f"Classifier: {classifier.model}")

    for category in categories:
        data = pd.read_json(f"../eval/mmlu/codellama_mmlu/pretrained__codellama__CodeLlama-34b-Instruct-hf,tensor_parallel_size__1,dtype__auto,gpu_memory_utilization__0.9_mmlu_{category}.jsonl")
        with ThreadPoolExecutor(args.parallel) as executor:
            future = []
            index_list = []
            for index, row in data.iterrows():
                future.append(executor.submit(classifier.classify_prompt, row["arguments"][0][0]))
                index_list.append(index)
            for idx, f in tqdm(enumerate(future)):
                doc_id = data.loc[index_list[idx], "doc_id"]
                prediction = f.result()
                prompt_id = f"mmlu-{category}-{doc_id}"
                results.append((prompt_id, prediction))

    df = pd.DataFrame(results, columns=["prompt_id", "prediction"])
    df.to_csv("mmlu.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", type=int, default=8)
    args = parser.parse_args()

    eval(args)
