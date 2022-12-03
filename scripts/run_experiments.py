from IRT.datasets import BasicDataset, RealisticDataset
from IRT.utils import run_experiments

SIZES = (0, 200)
NUM_RUNS = 1
THREE_PL = True

#dataset = RealisticDataset(use_caching=False)
#dataset = BasicDataset(100000,200,use_caching=False)
dataset = BasicDataset(10000,100,use_caching=False)

run_experiments(
    dataset=dataset,
    num_runs=NUM_RUNS,
    sizes=SIZES,
    add=True,
    ThreePL=THREE_PL
)