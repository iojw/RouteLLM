from datasets import load_dataset, concatenate_datasets


SEED = 1
PROMPTS_PER_DATASET = 100


def gen_mmlu():
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
            "source": "mmlu",
        },
        remove_columns=["input", "A", "B", "C", "D", "target"],
    )
    return mmlu_dataset


def gen_human_eval():
    human_eval_dataset = (
        load_dataset("openai_humaneval", split="test")
        .shuffle(seed=SEED)
        .select(range(PROMPTS_PER_DATASET))
    )
    human_eval_dataset = human_eval_dataset.map(
        lambda row: {"prompt": row["prompt"], "source": "humaneval"},
        remove_columns=["task_id", "prompt", "canonical_solution", "test", "entry_point"],
    )
    return human_eval_dataset


def gen_gsk8k():
    gsm8k_dataset = (
        load_dataset("gsm8k", "main", split="test")
        .shuffle(seed=SEED)
        .select(range(PROMPTS_PER_DATASET))
    )
    gsm8k_dataset = gsm8k_dataset.map(
        lambda row: {"prompt": row["question"], "source": "gsm8k"},
        remove_columns=["answer", "question"],
    )
    return gsm8k_dataset


def gen_mtbench():
    mtbench_dataset = (
        load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
    )
    mtbench_dataset = mtbench_dataset.map(
        lambda row: {"prompt": row["prompt"][0], "source": "mtbench-"+row["category"]},
        remove_columns=["prompt", "prompt_id", "category", "reference"],
    )

    # only keep coding/math/stem categories
    mtbench_dataset = mtbench_dataset.filter(
        lambda row: row["source"] in ["mtbench-coding", "mtbench-math", "mtbench-writing"])
    return mtbench_dataset

def gen_data_v0():
    mmlu_dataset = gen_mmlu()
    human_eval_dataset = gen_human_eval()

    concatenate_datasets([mmlu_dataset, human_eval_dataset]).shuffle(seed=SEED).to_csv(
        "benchmark/data_v0.csv"
    )


def gen_data_v1():
    mmlu_dataset = gen_mmlu()
    human_eval_dataset = gen_human_eval()
    gsk8k_dataset = gen_gsk8k()

    concatenate_datasets([mmlu_dataset, human_eval_dataset, gsk8k_dataset]).shuffle(seed=SEED).to_csv(
        "benchmark/data_v1.csv"
    )


def gen_data_mtbench():
    mt_bench_dataset = gen_mtbench()

    concatenate_datasets([mt_bench_dataset]).shuffle(seed=SEED).to_csv(
        "benchmark/data_mtbench.csv"
    )


# gen_data_v0()
gen_data_v1()
gen_data_mtbench()
