import pandas as pd

# Only forward coding / math to GPT-4 and forward everything else to Llama-70b

benchmark_files = {
    "MTBench": ("mtbench/mtbench.csv", "classification/mtbench.csv", "score", 80),
}

combined_scores = {}

categories = {}
cost = 0

print("MixLlama\n========")
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
        & (merged_df["model"] == "gpt-4")
    ]
    none_df = merged_df[
        (merged_df["prediction"] == "Label.NONE")
        & (merged_df["model"] == "Llama-2-70b-chat-hf")
    ]
    math_df = merged_df[
        (merged_df["prediction"] == "Label.MATH")
        & (merged_df["model"] == "gpt-4")
    ]


    cost += coding_df['tokens'].sum() / 1000 * 0.03
    cost += math_df['tokens'].sum() / 1000 * 0.03
    cost += none_df['tokens'].sum() / 1000 * 0.001
    cost_per_1000_tokens = cost / (coding_df['tokens'].sum() + math_df['tokens'].sum() + none_df['tokens'].sum()) * 1000

    # cost += coding_df['tokens'].sum() * 0.00003
    # cost += math_df['tokens'].sum() * 0.00003
    # cost += none_df['tokens'].sum() * 0.000001
    # cost_per_1000_tokens = cost / (coding_df['tokens'].sum() + math_df['tokens'].sum() + none_df['tokens'].sum()) * 1000

    combined_df = pd.concat([coding_df, none_df, math_df])

    print(combined_df)
    assert (
        len(combined_df) == num_prompts
    ), f"Expected {num_prompts} prompts for {benchmark}, got {len(combined_df)}"

    total_score = combined_df[score_col].mean()
    print(f"{benchmark}:\n{total_score}")
    print("cost per 1000 tokens:", cost_per_1000_tokens)

