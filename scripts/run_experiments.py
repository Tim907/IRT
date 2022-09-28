from IRT.datasets import Basic_Dataset
from IRT.utils import run_experiments

NUM_RUNS = 21

dataset = Basic_Dataset(use_caching=False)

run_experiments(
    dataset=dataset,
    num_runs=NUM_RUNS,
    add=True
)