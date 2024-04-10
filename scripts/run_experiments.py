import os
os.environ['NUMEXPR_MAX_THREADS'] = '8'

from IRT.datasets import BasicDataset, RealisticDataset, Realistic3PLDataset
from IRT.utils import run_experiments

#SIZES = (0, 500)
NUM_RUNS = 1 # Sets how many repetitions of the same experiment 
THREE_PL = False #True  #Sets whether it is 2PL or 3PL experiment

dataset = RealisticDataset(use_caching=False) # Sets which input set of labels is used
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
    sizes=(0, 50), # Sets the coreset sizes for (items, examinees). "0" - no coreset computed
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

