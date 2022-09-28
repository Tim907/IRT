import logging

from . import settings
from . import optimizer
from .datasets import Dataset
from .experiments import (
    IRTExperiment,
)

logger = logging.getLogger(settings.LOGGER_NAME)


def run_experiments(dataset: Dataset, num_runs, add=False):
    # check if results directory exists
    if not settings.RESULTS_DIR.exists():
        settings.RESULTS_DIR.mkdir()

    logger.info("Starting IRT experiment")
    experiment = IRTExperiment(
        dataset=dataset,
        results_filename=settings.RESULTS_DIR / f"{dataset.get_name()}_cauchy.csv",
        min_size=20,
        max_size=20,
        step_size=20,
        num_runs=num_runs,
    )
    experiment.run(parallel=False, add=add)