import os, requests, subprocess, json, shutil, time
from typing import List, Optional
from timeit import default_timer as timer

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import click

KNOWN_ALGORITHMS = [
    "basic-clustering",
    "best-clustering",
    "sensitivity-sampling",
    "group-sampling",
    "bico",
    "stream-km++",
    "ray-maker",
]

KNOWN_DATASETS = [
    "covertype",
    "covertypelowd",
    "census",
    "censuslowd",
    "tower",
    "enron",
    "caltech101",
    "caltech101lowd",
    "hardinstanceb1",
    "hardinstanceb15",
    "hardinstanceb2",
    "nytimes100d",
    "nytimes",
    "nytimespcalowd",
]

MT_PATH = "IRT/eval_k_means_coresets_main/mt/bin/mt.exe"


def generate_random_seed() -> int:
    # Calling mt.exe to generate a random seed quickly results 
    # in the same random seed. So pause for a second for each
    # generation.
    if not os.path.exists(MT_PATH):
        raise Exception(f"Random seed generator '{MT_PATH}' cannot be found. You can build it: make -C mt")
    time.sleep(1)
    p = subprocess.run([MT_PATH], stdout=subprocess.PIPE)
    return int(p.stdout)


def validate_algorithms(ctx, param, value):
    if value is None or value == "all":
        return KNOWN_ALGORITHMS
    ret_val = []
    for s in value.split(","):
        if s in ["basic", "basic-clustering"]:
            ret_val.append("basic-clustering")
        elif s in ["best", "best-clustering"]:
            ret_val.append("best-clustering")
        elif s in ["ss", "sensitivity", "sensitivity-sampling"]:
            ret_val.append("sensitivity-sampling")
        elif s in ["gs", "group", "group-sampling"]:
            ret_val.append("group-sampling")
        elif s in ["bico"]:
            ret_val.append("bico")
        elif s in ["skm", "skm++", "stream-km++"]:
            ret_val.append("stream-km++")
        elif s in ["ray", "ray-maker"]:
            ret_val.append("ray-maker")
        elif s in KNOWN_ALGORITHMS:
            ret_val.append(s)
        else:
            raise Exception(f"Known algorithm: {s}")
    return ret_val


def validate_datasets(ctx, param, value):
    if value is None or value == "all":
        return KNOWN_DATASETS
    ret_val = []
    for s in value.split(","):
        if s in ["en", "enron"]:
            ret_val.append("enron")
        elif s in ["cov", "covertype"]:
            ret_val.append("covertype")
        elif s in ["cen", "census"]:
            ret_val.append("census")
        elif s in ["t", "to", "tower"]:
            ret_val.append("tower")
        elif s in KNOWN_DATASETS:
            ret_val.append(s)
        else:
            raise Exception(f"Known dataset: {s}")
    return ret_val
        

@click.command(help="Generates experiment files.")
@click.option(
    "-s",
    "--iter-start",
    type=click.INT,
    required=True,
)
@click.option(
    "-e",
    "--iter-end",
    type=click.INT,
    default=-1,
    required=False,
)
@click.option(
    "-m",
    "--coreset-size",
    type=click.INT,
    required=True,
)
@click.option(
    "-k",
    "--number-clusters",
    type=click.INT,
    required=True,
)
@click.option(
    "-a",
    "--algorithms",
    type=click.STRING,
    required=False,
    default=None,
    callback=validate_algorithms
)
@click.option(
    "-d",
    "--datasets",
    type=click.STRING,
    required=False,
    default=None,
    callback=validate_datasets
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Recreate files."
)
def main(iter_start: int, iter_end: Optional[int], coreset_size: int, number_clusters: int, algorithms: List[str], datasets: List[str], force: bool) -> None:
    call_main(iter_start, iter_end, coreset_size, number_clusters, algorithms, datasets, force)

def call_main(iter_start: int, iter_end: Optional[int], coreset_size: int, number_clusters: int, algorithms: List[str], datasets: List[str], force: bool) -> None:
    if not os.path.exists(MT_PATH):
        print(f"Random seed generator '{MT_PATH}' cannot be found. You can build it: make -C mt")
        return
    
    if iter_end is None or iter_end < iter_start:
        iter_end = iter_start

    dataset_k_ranges = {
        "census": [10, 20, 30, 40, 50],
        "censuslowd": [10, 20, 30, 40, 50],
        "covertype": [10, 20, 30, 40, 50],
        "covertypelowd": [10, 20, 30, 40, 50],
        "enron": [10, 20, 30, 40, 50],
        "tower": [20, 40, 60, 80, 100],
        "caltech101": [10, 20, 30, 40, 50],
        "caltech101lowd": [10, 20, 30, 40, 50],
        "hardinstanceb1": [10, 20, 30, 40],
        "hardinstanceb15": [10, 20, 30, 40],
        "hardinstanceb2": [10, 20, 30, 40],
        "nytimes100d": [10, 20, 30, 40, 50],
        "nytimes": [10, 20, 30, 40, 50],
        "nytimespcalowd": [10, 20, 30, 40, 50],
    }

    ready_dir = Path("eval_k_means_coresets_main/data/queue/ready")
    if not ready_dir.exists():
        os.makedirs(ready_dir)
    for dataset in datasets:
        for algo in algorithms:
            k = number_clusters
            for i in range(iter_start, iter_end + 1):
                m = coreset_size
                random_seed = generate_random_seed()
                print(f"Random seed {random_seed}")
                exp_details = {
                    "iteration": i,
                    "algorithm": algo,
                    "dataset": dataset,
                    "k": k,
                    "m": m,
                    "randomSeed": random_seed,
                }
                file_path = ready_dir / f"{i:03}-{dataset}-{algo}-k{k}-m{m}.json"
                if force or not file_path.exists():
                    print(f"Writing {file_path}...")
                    with open(file_path, "w") as f:
                        json.dump(exp_details, f, indent=4)
                else:
                    print(f"File already exists {file_path}. Skipping...")

if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
