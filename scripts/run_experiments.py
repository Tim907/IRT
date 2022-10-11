from IRT.datasets import BasicDataset, RealisticDataset
from IRT.utils import run_experiments

MIN_SIZE = 100
MAX_SIZE = 100
STEP_SIZE = 1
NUM_RUNS = 1

#dataset = RealisticDataset(use_caching=False)
dataset = BasicDataset(100000,200,use_caching=False)

run_experiments(
    dataset=dataset,
    num_runs=NUM_RUNS,
    min_size=MIN_SIZE,
    max_size=MAX_SIZE,
    step_size=STEP_SIZE,
    add=True
)