import logging

from . import settings
from . import optimizer
from .datasets import Dataset
from .experiments import (
    L2SExperiment, UniformSamplingExperiment, SensitivitySamplingExperiment, L1LewisSamplingExperiment
)

logger = logging.getLogger(settings.LOGGER_NAME)


def run_experiments(dataset: Dataset, num_runs, sizes, add=False, ThreePL=False):
    # check if results directory exists
    if not settings.RESULTS_DIR.exists():
        settings.RESULTS_DIR.mkdir()
    
    #next three lines introduce the file for collecting the values of the goal function. It will be filled after each experiment
    cumulative_res_filename = dataset.get_name() + f"_goal_values.csv"
    f = open(settings.RESULTS_DIR / f"{cumulative_res_filename}", "wt")
    f.close()
    
    added_num_runs = num_runs
    for rrr in range(1, added_num_runs + 1):

        logger.info("Starting IRT experiment number " + f"{rrr}")
        
        # #next block is used if the "coreset" is obtained via uniform sampling
        # experiment = UniformSamplingExperiment(
           # dataset,
           # results_filename= f"{dataset.get_name()}_uni" + f"_{rrr}",
           # sizes=sizes,
           # num_runs=1,
        # )
        # experiment.run(parallel=False, add=add, ThreePL=ThreePL)
        
        #next block is used if our coreset is to be constructed
        # experiment = L2SExperiment(
            # dataset,
            # results_filename= f"{dataset.get_name()}_l2s" + f"_{rrr}",
            # sizes=sizes,
            # num_runs=1,
        # )
        # experiment.run(parallel=False, add=add, ThreePL=ThreePL)

        # experiment = SensitivitySamplingExperiment(
        #     dataset,
        #     results_filename= f"{dataset.get_name()}_l2s" + f"_{rrr}",
        #     sizes=sizes,
        #     num_runs=1,
        # )

        experiment = L1LewisSamplingExperiment(
            dataset,
            results_filename= f"{dataset.get_name()}_l2s" + f"_{rrr}",
            sizes=sizes,
            num_runs=1,
            fast_approx=True
        )

        # experiment = L1LewisSamplingExperiment(
        #     dataset,
        #     results_filename= f"{dataset.get_name()}_l2s" + f"_{rrr}",
        #     sizes=sizes,
        #     num_runs=1,
        #     fast_approx=True
        # )

        experiment.run(parallel=False, add=add, ThreePL=ThreePL)
    