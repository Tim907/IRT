import os
os.environ['NUMEXPR_MAX_THREADS'] = '8'

from IRT.datasets import BasicDataset, RealisticDataset, Realistic3PLDataset
from IRT.utils import run_experiments

#SIZES = (0, 500)
NUM_RUNS = 1
THREE_PL = False #True

dataset = RealisticDataset(use_caching=False)
#dataset = Realistic3PLDataset(use_caching=False)

#dataset = BasicDataset(100000,200,use_caching=False)
#dataset = BasicDataset(10000,100,use_caching=False)

# run_experiments(
    # dataset=dataset,
    # num_runs=NUM_RUNS,
    # sizes=SIZES,
    # add=True,
    # ThreePL=THREE_PL
# )

run_experiments(
    dataset=dataset,
    num_runs=NUM_RUNS,
    sizes=(0, 50),
    add=True,
    ThreePL=THREE_PL
)

# run_experiments(
    # dataset=dataset,
    # num_runs=NUM_RUNS,
    # sizes=(0, 1000),
    # add=True,
    # ThreePL=THREE_PL
# )

# run_experiments(
    # dataset=dataset,
    # num_runs=NUM_RUNS,
    # sizes=(0, 4000),
    # add=True,
    # ThreePL=THREE_PL
# )

# run_experiments(
    # dataset=dataset,
    # num_runs=NUM_RUNS,
    # sizes=(0, 6000),
    # add=True,
    # ThreePL=THREE_PL
# )

# run_experiments(
    # dataset=dataset,
    # num_runs=NUM_RUNS,
    # sizes=(0, 8000),
    # add=True,
    # ThreePL=THREE_PL
# )

# run_experiments(
    # dataset=dataset,
    # num_runs=NUM_RUNS,
    # sizes=(0, 10000),
    # add=True,
    # ThreePL=THREE_PL
# )

# run_experiments(
    # dataset=dataset,
    # num_runs=NUM_RUNS,
    # sizes=(0, 20000),
    # add=True,
    # ThreePL=THREE_PL
# )
