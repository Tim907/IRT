from IRT.datasets import Basic_Dataset
from IRT.utils import run_experiments

MIN_SIZE = 50
MAX_SIZE = 50
STEP_SIZE = 1
NUM_RUNS = 1

dataset = Basic_Dataset(n=10000, m=20, use_caching=False)

run_experiments(
    dataset=dataset,
    num_runs=NUM_RUNS,
    min_size=MIN_SIZE,
    max_size=MAX_SIZE,
    step_size=STEP_SIZE,
    add=True
)