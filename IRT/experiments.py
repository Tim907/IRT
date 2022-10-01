import abc
import logging
import math
from time import perf_counter

import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import SGDClassifier

from . import optimizer, settings, datasets
from .datasets import Dataset
from .l2s_sampling import l2s_sampling

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
    def get_reduced_matrix_and_weights(self, Z, config):
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


    def IRT(self, X, config=None):
        n = X.shape[1]
        m = X.shape[0]

        theta = np.zeros(X.shape[1]) + np.random.standard_normal(X.shape[1])
        Alpha = np.vstack((theta, -np.ones(X.shape[1]))).T
        Beta = np.vstack((np.ones(X.shape[0]) + np.random.standard_normal(X.shape[0]), np.random.standard_normal(X.shape[0]))).T

        sumCostOld = math.inf
        for iteration in range(500):
            sumCost = 0
            weights = None

            updated_param = np.zeros(m * 2).reshape(m, 2)
            for i in range(m):
                Z = datasets.make_Z(Alpha, X[i, :])
                if config is not None:
                    Z, weights = self.get_reduced_matrix_and_weights(Z, config)
                opt = optimizer.optimize(Z, w=weights)
                updated_param[i, ] = opt.x
                sumCost += opt.fun
            Beta = updated_param

            updated_param = np.zeros(n * 2).reshape(n, 2)
            for i in range(n):
                Z = datasets.make_Z(Beta, X[:, i])
                opt = optimizer.optimize(Z)
                updated_param[i, ] = opt.x
                sumCost += opt.fun
            # Alpha has fixed -1 in second column
            updated_param[:, 1] = -1
            Alpha = updated_param

            logger.info(f"Iteration {iteration+1} has total cost {sumCost}.")
            if sumCostOld - sumCost < 0.0001:
                logger.info(f"ended early because improvement of {sumCostOld - sumCost} is too low.")
                break
            sumCostOld = sumCost

        if config is None:
            result_filename = self.dataset.get_name()
        else:
            size = config["size"]
            result_filename = self.results_filename + f"_{size}"
        df = pd.DataFrame(Alpha)
        df.to_csv(settings.RESULTS_DIR / f"{result_filename}_Alpha.csv", header=False, index=False)
        df = pd.DataFrame(Beta)
        df.to_csv(settings.RESULTS_DIR / f"{result_filename}_Beta.csv", header=False, index=False)
        df = pd.DataFrame(X)
        df.to_csv(settings.RESULTS_DIR / f"{result_filename}_data.csv", header=False, index=False)


    def run(self, parallel=False, n_jobs=-3, add=False):
        X = self.dataset.get_X()
        logger.info("Computing IRT on full dataset...")
        self.IRT(X)

        logger.info("Running experiments...")

        def job_function(cur_config):
            logger.info(f"Current experimental config: {cur_config}")
            self.IRT(X, cur_config)

        for cur_config in self.get_config_grid():
            job_function(cur_config)

        logger.info("Done.")


class L2SExperiment(BaseExperiment):
    def __init__(
        self,
        dataset: Dataset,
        results_filename,
        min_size,
        max_size,
        step_size,
        num_runs
    ):
        super().__init__(
            num_runs=num_runs,
            min_size=min_size,
            max_size=max_size,
            step_size=step_size,
            dataset=dataset,
            results_filename=results_filename
        )

    def get_reduced_matrix_and_weights(self, Z, config):
        size = config["size"]

        reduced_matrix, weights = l2s_sampling(Z, size=size)

        return reduced_matrix, weights