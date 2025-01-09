import argparse
import json
import random
import time
from typing import List

from datasets import load_dataset
import pandarallel
import yaml

from routellm.controller import Controller
from routellm.routers.routers import ROUTER_CLS

router = None

def calculate_threshold(query):
    return router.calculate_threshold(query)


def measure_router_throughput(
    router_list: List[str], num_queries: int, dataset_name: str
) -> None:
    random.seed(0)

    battles_df = load_dataset(args.battles_dataset, split="train").to_pandas()
    # battles_df = battles_df[battles_df['prompt'].str.len() <= 4096]
    battles_df = battles_df.sample(n=num_queries, random_state=0)
    controller = Controller(
        routers=args.routers,
        config=yaml.safe_load(open(args.config, "r")) if args.config else None,
        # This is not needed since we only calculate the win rate
        strong_model=None,
        weak_model=None,
        progress_bar=False,
    )

    first_turn = battles_df["prompt"].apply(lambda x: json.loads(x)[0][:1000])
    # first_turn = first_turn.tolist()
    # first_turn_concat = " ".join(first_turn)[:10000]
    for router in args.routers:
        # warmup
        controller.batch_calculate_win_rate(
            first_turn[:15], router
        )
        # for _ in range(10):
        #     controller.route(first_turn_concat, router, 0)

        start_time = time.perf_counter()
        # for _ in range(10):
        #     controller.route(first_turn_concat, router, 0)
        # controller.batch_calculate_win_rate(
        #     first_turn, router
        # )
        for p in first_turn:
            controller.route(
                p, router, 0
            )
        end_time = time.perf_counter()

        throughput = num_queries / ((end_time - start_time) / 10)
        print(f"{router} throughput: {throughput:.2f} queries per second, {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Measure maximum throughput of routers."
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=100,
        help="Number of queries to run for each router",
    )
    parser.add_argument(
        "--battles_dataset", type=str, default="lmsys/lmsys-arena-human-preference-55k"
    )
    parser.add_argument(
        "--routers",
        nargs="+",
        type=str,
        default=["random"],
        choices=list(ROUTER_CLS.keys()),
    )
    parser.add_argument("--config", type=str, default=None)

    args = parser.parse_args()
    print(args)

    measure_router_throughput(args.routers, args.num_queries, args.battles_dataset)
