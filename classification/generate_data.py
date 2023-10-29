from datasets import load_dataset, concatenate_datasets

PROMPTS_PER_DATASET = 100
SEED = 1

mmlu_dataset = (
    load_dataset("lukaemon/mmlu", "miscellaneous", split="test")
    .shuffle(seed=SEED)
    .select(range(PROMPTS_PER_DATASET))
)

mmlu_dataset = mmlu_dataset.map(
    # Based on the original eval: https://github.com/hendrycks/test/blob/master/evaluate.py
    lambda row: {
        "prompt": "The following are multiple choice questions (with answers) about miscellaneous.\n\n"
        f"{row['input']}\n"
        + "\n".join(f"{choice}. {row[choice]}" for choice in ["A", "B", "C", "D"])
        + "\nAnswer:",
        "is_code_prompt": False,
    },
    remove_columns=["input", "A", "B", "C", "D", "target"],
)

human_eval_dataset = (
    load_dataset("openai_humaneval", split="test")
    .shuffle(seed=SEED)
    .select(range(PROMPTS_PER_DATASET))
)

human_eval_dataset = human_eval_dataset.map(
    lambda row: {"prompt": row["prompt"], "is_code_prompt": True},
    remove_columns=["task_id", "prompt", "canonical_solution", "test", "entry_point"],
)

concatenate_datasets([mmlu_dataset, human_eval_dataset]).shuffle(seed=SEED).to_csv(
    "data.csv"
)
