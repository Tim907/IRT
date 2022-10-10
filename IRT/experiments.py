import abc
import logging
import math
from time import perf_counter

import os
import numpy as np
import pandas as pd
import scipy.stats
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
        n = X.shape[1] # number of students
        m = X.shape[0] # number of items
        logger.info(f"Running IRT with {n} students and {m} items.")
        if config is not None:
            size = config["size"]
            logger.info(f"Working with coreset of size {size}.")

        theta = np.zeros(X.shape[1])
        Alpha = np.vstack((theta, -np.ones(X.shape[1]))).T
        #Beta = np.vstack((np.ones(X.shape[0]) + np.random.standard_normal(X.shape[0]), np.random.standard_normal(X.shape[0]))).T
        #Beta = np.vstack((scipy.stats.norm.ppf((X == 1).mean(axis=1)) / (np.sqrt(np.absolute(1-(0.5)**2)) / 1.702), np.ones(X.shape[0]) * 0.15)).T
        Beta = np.vstack((scipy.stats.norm.ppf(((X+1)/2).mean(axis=1)) * 1.702 / np.sqrt(0.75), np.ones(X.shape[0]) * 0.851)).T
        
        Alpha_core = None# Alpha
        X_core = None # X

        
        t1_start = perf_counter()
        sumCostOld = math.inf
        for iteration in range(10):
            sumCost = 0
            weights = None
            coreset = None

            # updated_param = np.zeros(n * 2).reshape(n, 2)
            for i in range(n):
                Z = datasets.make_Z(Beta, X[:, i])
                opt = optimizer.optimize(Z, bnds=((-6, 6), (-1.0, -1.0)), theta_init = Alpha[i,:])
                Alpha[i, ] = opt.x
                sumCost += opt.fun
            # Alpha has fixed -1 in second column 
            # handled by bnds argument in optimizer
            # Alpha = updated_param
            # Alpha_core = Alpha

            if config is not None:
                coreset, weights = self.get_reduced_matrix_and_weights(Alpha, config)
                Alpha_core = Alpha[coreset]
                X_core = X[:, coreset]

            # updated_param = np.zeros(m * 2).reshape(m, 2)
            
        
            
            for i in range(m):
                if config is not None:
                    Z = datasets.make_Z(Alpha_core, X_core[i,:])
                else:
                    Z = datasets.make_Z(Alpha, X[i,:])
                opt = optimizer.optimize(Z, w=weights, bnds=((0, 5), (-6, 6)), theta_init = Beta[i,:])
                Beta[i, ] = opt.x
                sumCost += opt.fun
            # Beta = updated_param
            

            logger.info(f"Iteration {iteration+1} has total cost {sumCost}.")
            #if sumCostOld - sumCost < 0.0001:
            improvement = (sumCostOld - sumCost)/sumCostOld
            if iteration >= 1:
                logger.info(f"Iteration {iteration+1} has improved by {improvement}.")
            if np.absolute(improvement) < 0.001:
                logger.info(f"ended early because improvement of {sumCostOld - sumCost} is only a {improvement} fraction.")
                #break
            sumCostOld = sumCost
            
        t1_stop = perf_counter()
        print("######## Running time (s):", t1_stop-t1_start)

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
        # reduced_matrix is only a vector of indexes!!!
        return reduced_matrix, weights