import pandas as pd

benchmark_files = {
    "mmlu": ("mmlu/mmlu.csv", "classification/mmlu.csv", "score", 14042),
    "gsm8k": ("gsm8k/gsm8k.csv", "classification/gsm8k.csv", "score", 1319),
    "humaneval (pass@1)": (
        "humaneval/humaneval.csv",
        "classification/humaneval.csv",
        "pass@1",
        164,
    ),
    "humaneval (pass@10)": (
        "humaneval/humaneval.csv",
        "classification/humaneval.csv",
        "pass@10",
        164,
    ),
    "MTBench": ("mtbench/mtbench.csv", "classification/mtbench.csv", "score", 80),
}

combined_scores = {}

for benchmark, (
    filename,
    classification_filename,
    score_col,
    num_prompts,
) in benchmark_files.items():
    benchmark_df = pd.read_csv(filename)
    classification_df = pd.read_csv(classification_filename)

    merged_df = pd.merge(benchmark_df, classification_df, on="prompt_id")

    coding_df = merged_df[
        (merged_df["prediction"] == "Label.CODING")
        & (merged_df["model"] == "CodeLlama-34b-Instruct-hf")
    ]
    none_df = merged_df[
        (merged_df["prediction"] == "Label.NONE")
        & (merged_df["model"] == "Llama-2-70b-chat-hf")
    ]
    math_df = merged_df[
        (merged_df["prediction"] == "Label.MATH")
        & (merged_df["model"] == "MetaMath-70B-V1.0")
    ]

    combined_df = pd.concat([coding_df, none_df, math_df])

    assert (
        len(combined_df) == num_prompts
    ), f"Expected {num_prompts} prompts for {benchmark}, got {len(combined_df)}"

    total_score = combined_df[score_col].mean()
    combined_scores[benchmark] = total_score

for benchmark, score in combined_scores.items():
    print(f"{benchmark}: {score}")
