import logging

from . import settings
from . import optimizer
from .datasets import Dataset
from .experiments import (
    L2SExperiment,
)

logger = logging.getLogger(settings.LOGGER_NAME)


def run_experiments(dataset: Dataset, num_runs, add=False):
    # check if results directory exists
    if not settings.RESULTS_DIR.exists():
        settings.RESULTS_DIR.mkdir()

    logger.info("Starting IRT experiment")
    experiment = L2SExperiment(
        dataset,
        results_filename= f"{dataset.get_name()}_l2s",
        min_size=1,
        max_size=5,
        step_size=1,
        num_runs=1,
    )
    experiment.run(parallel=False, add=add)