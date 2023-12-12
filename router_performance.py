import matplotlib.pyplot as plt


PERFORMANCE1 = {
    "Random": 0.333,
    "N-gram": 0.63,
    "Embedding": 1.0,
    "Zero-shot\nLlama2-7B": 0.296,
    "Finetuned\nLlama2-7B": 0.946,
    "Zero-shot\nGPT-3.5": 0.99,
}

PERFORMANCE2 = {
    "Random": 0.333,
    "N-gram": 0.56,
    "Embedding": 0.966,
    "Zero-shot\nLlama2-7B": 0.5,
    "Finetuned\nLlama2-7B": 0.833,
    "Zero-shot\nGPT-3.5": 1.0,
}


def draw_performance(performance, title, color):
    plt.figure(figsize=(11, 6))
    plt.bar(performance.keys(), performance.values(), color=color)
    # plt.axhline(y=0.33, color="black", linestyle="--")
    for k, v in performance.items():
        plt.text(k, v + 0.01, f"{v:.2f}", ha="center", fontsize=20)
    plt.ylim(0, 1.1)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=20)
    plt.ylabel("F1 score", fontsize=20)
    plt.tight_layout()
    plt.savefig(f"{title}.png", dpi=600)


if __name__ == "__main__":
    draw_performance(PERFORMANCE1, "router_academic", color="#fdb515ff")
    draw_performance(PERFORMANCE2, "router_chat", color="#30a2ffff")
