import abc
import logging
from time import perf_counter

import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import SGDClassifier

from . import optimizer, settings
from .datasets import Dataset

logger = logging.getLogger(settings.LOGGER_NAME)

_rng = np.random.default_rng()


class BaseExperiment(abc.ABC):
    def __init__(
        self,
        num_runs,
        min_size,
        max_size,
        step_size,
        dataset: Dataset,
        results_filename,
    ):
        self.num_runs = num_runs
        self.min_size = min_size
        self.max_size = max_size
        self.step_size = step_size
        self.dataset = dataset
        self.results_filename = results_filename
        self.optimizer = optimizer

    @abc.abstractmethod
    def get_reduced_matrix_and_weights(self, config):
        pass

    def get_config_grid(self):
        """
        Returns a list of configurations that are used to run the experiments.
        """
        grid = []
        for size in np.arange(
            start=self.min_size,
            stop=self.max_size + self.step_size,
            step=self.step_size,
        ):
            for run in range(1, self.num_runs + 1):
                grid.append({"run": run, "size": size})

        return grid


    def run(self, parallel=False, n_jobs=-3, add=False):
        beta_opt = self.dataset.get_beta_opt()
        #objective_function = optimizer.get_objective_function(self.dataset.get_X())
        #f_opt = objective_function(beta_opt)

        logger.info("Running experiments...")

        def job_function(cur_config):
            logger.info(f"Current experimental config: {cur_config}")

            start_time = perf_counter()

            reduced_matrix, weights = self.get_reduced_matrix_and_weights(cur_config)
            sampling_time = perf_counter() - start_time

            cur_beta_opt = self.optimizer.optimize(reduced_matrix, weights)
            total_time = perf_counter() - start_time

            cur_ratio = 1#objective_function(cur_beta_opt) / f_opt
            return {
                **cur_config,
                "ratio": cur_ratio,
                "sampling_time_s": sampling_time,
                "total_time_s": total_time,
            }

        if parallel:
            results = Parallel(n_jobs=n_jobs)(
                delayed(job_function)(cur_config)
                for cur_config in self.get_config_grid()
            )
        else:
            results = [
                job_function(cur_config) for cur_config in self.get_config_grid()
            ]

        logger.info(f"Writing results to {self.results_filename}")

        df = pd.DataFrame(results)
        if not os.path.isfile(self.results_filename):
            df.to_csv(self.results_filename, index=False)
        else:
            df.to_csv(self.results_filename, mode = None if add == False else "a", header = add == False, index=False)

        logger.info("Done.")


class IRTExperiment(BaseExperiment):

    def __init__(
        self,
        dataset: Dataset,
        results_filename,
        min_size,
        max_size,
        step_size,
        num_runs,
    ):
        super().__init__(
            num_runs=num_runs,
            min_size=min_size,
            max_size=max_size,
            step_size=step_size,
            dataset=dataset,
            results_filename=results_filename,
        )

    def get_reduced_matrix_and_weights(self, config):
        pass