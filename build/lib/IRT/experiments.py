import abc
import csv
import io
import logging
import math
import os
from io import StringIO
from time import perf_counter

import subprocess
import numpy as np
import pandas as pd
import scipy.stats
from joblib import Parallel, delayed
from sklearn.linear_model import SGDClassifier

from . import optimizer, settings, datasets
from .datasets import Dataset
from .l2s_sampling import l2s_sampling
from eval_k_means_coresets_main.xrun import gen2, go2
import pathlib

logger = logging.getLogger(settings.LOGGER_NAME)

_rng = np.random.default_rng()


class BaseExperiment(abc.ABC):
    def __init__(
            self,
            num_runs,
            sizes,
            dataset: Dataset,
            results_filename,
    ):
        self.num_runs = num_runs
        self.sizes = sizes
        self.dataset = dataset
        self.results_filename = results_filename
        self.optimizer = optimizer

    @abc.abstractmethod
    def get_reduced_matrix_and_weights(self, Z, size):
        pass

    def get_config_grid(self):
        """
        Returns a list of configurations that are used to run the experiments.
        """
        grid = []
        for run in range(1, self.num_runs + 1):
            grid.append({"run": run, "sizes": self.sizes})

        return grid

    def IRT(self, X, ThreePL=False, config=None):
        n = X.shape[1]  # number of students
        m = X.shape[0]  # number of items
        logger.info(f"Running IRT with {n} students and {m} items.")
        if config is not None:
            sizes = config["sizes"]
            logger.info(f"Working with coreset of sizes {sizes}.")

        theta = np.random.standard_normal(X.shape[1])  # np.zeros(X.shape[1]) +
        Alpha = np.vstack((theta, -np.ones(X.shape[1]))).T

        # Alpha = pd.read_csv(settings.DATA_DIR / "Theta_3PL_mirt.csv", delimiter=";", decimal=",", header=0, index_col=0)
        # Alpha["2"] = -np.ones(X.shape[1])
        # Alpha = Alpha.to_numpy()
        # Alpha[:,0] = (Alpha[:,0] - np.mean(Alpha[:,0])) / np.std(Alpha[:,0])

        # Beta = np.vstack((np.ones(X.shape[0]) + np.random.standard_normal(X.shape[0]), np.random.standard_normal(X.shape[0]))).T
        # Beta = np.vstack((scipy.stats.norm.ppf((X == 1).mean(axis=1)) / (np.sqrt(np.absolute(1-(0.5)**2)) / 1.702), np.ones(X.shape[0]) * 0.15)).T
        # Beta = np.vstack((scipy.stats.norm.ppf(((X+1)/2).mean(axis=1)) * 1.702 / np.sqrt(0.75), np.ones(X.shape[0]) * 0.851)).T
        # Beta = np.vstack((np.ones(X.shape[0]) * 0.851, scipy.stats.norm.ppf(((X+1)/2).mean(axis=1)) * 1.702 / np.sqrt(0.75))).T
        Beta = np.vstack((np.ones(X.shape[0]) * 2.75, np.zeros(X.shape[0]))).T
        if ThreePL is True:
            # append third column with c
            Beta = np.hstack((Beta, 0.25 * np.ones((Beta.shape[0], 1))))

        Alpha_core = None  # Alpha
        Beta_core = None  # Beta
        X_core = None  # X

        sumCostOld = math.inf
        sumCostBest = math.inf
        bestIteration = None
        bestAlpha = None
        bestBeta = None
        runtimes_df = pd.DataFrame(columns=['Alpha', 'Beta'])
        for iteration in range(
                5):  ###############################  here the number of iterations is set  ##############
            sumCost = 0
            weights = None
            coreset = None

            t1_start = perf_counter()

            if config is not None:
                if sizes[0] == 0:
                    Beta_core = Beta
                    X_core = X
                else:
                    coreset, weights = self.get_reduced_matrix_and_weights(Beta, sizes[0])
                    Beta_core = Beta[coreset]
                    X_core = X[coreset, :]

            for i in range(n):
                # for i in range(0):
                if config is not None:
                    y = X_core[:, i]
                    Z = datasets.make_Z(Beta_core[:, 0:2], y)  # without third column of c's
                else:
                    y = X[:, i]
                    Z = datasets.make_Z(Beta[:, 0:2], y)  # without third column of c's

                if ThreePL is True:
                    if config is not None:
                        c = Beta_core[:, 2]
                    else:
                        c = Beta[:, 2]
                    opt = optimizer.optimize_3PL(Z, y=y, c=c, opt_beta=False, w=weights, bnds=((-6, 6), (-1.0, -1.0)),
                                                 theta_init=Alpha[i, :])
                else:
                    opt = optimizer.optimize_2PL(Z, w=weights, bnds=((-6, 6), (-1.0, -1.0)), theta_init=Alpha[i, :])
                Alpha[i,] = opt.x
                sumCost += opt.fun

            # print(Alpha[:,0])
            # Alpha[:,0] = (Alpha[:,0] - np.mean(Alpha[:,0])) / (np.std(Alpha[:,0])+0.01)
            t1_stop = perf_counter()
            print("######## Alpha Running time (s):", t1_stop - t1_start)

            t2_start = perf_counter()

            if config is not None:
                if sizes[1] == 0:
                    Alpha_core = Alpha
                    X_core = X
                else:
                    coreset, weights = self.get_reduced_matrix_and_weights(Alpha, sizes[1])
                    Alpha_core = Alpha[coreset]
                    X_core = X[:, coreset]

            for i in range(m):
                if config is not None:
                    y = X_core[i, :]
                    Z = datasets.make_Z(Alpha_core, y)
                else:
                    y = X[i, :]
                    Z = datasets.make_Z(Alpha, y)

                if ThreePL is True:
                    # c = Beta[i, 2]
                    opt = optimizer.optimize_3PL(Z, y=y, c=None, opt_beta=True, w=weights,
                                                 bnds=((0, 5), (-6, 6), (0.001, 0.499)), theta_init=Beta[i, :])
                else:
                    opt = optimizer.optimize_2PL(Z, w=weights, bnds=((0, 5), (-6, 6)), theta_init=Beta[i, :])
                Beta[i, :] = opt.x
                # print(Beta[i, ])
                sumCost += opt.fun

            t2_stop = perf_counter()
            print("######## Beta Running time (s):", t2_stop - t2_start)
            runtimes_df.loc[iteration] = [t1_stop - t1_start, t2_stop - t2_start]

            logger.info(f"Iteration {iteration + 1} has total cost {sumCost}")

            if sumCost < sumCostBest:
                bestIteration = iteration
                bestAlpha = Alpha
                bestBeta = Beta
                sumCostBest = sumCost

            if iteration >= 1:
                improvement = (sumCostOld - sumCost) / sumCostOld
                logger.info(f"Iteration {iteration + 1} has improved by {improvement}")
                if np.absolute(improvement) < 0.001:
                    logger.info(
                        f"ended early because improvement of {sumCostOld - sumCost} is only a {improvement} fraction.")
                    # break
            sumCostOld = sumCost

        if config is None:
            result_filename = self.results_filename
            # self.dataset.get_name() #+ f"_M2" #for not overwriting the file in the case of repetition
        else:
            sizes = config["sizes"]
            result_filename = self.results_filename + f"_{sizes}"

        logger.info(f"Saving best Alpha and Beta from iteration {bestIteration + 1} with cost {sumCostBest}")

        # the following lines were added to store the optimal reached cost, to compare among multiple iterations
        cumulative_res_filename = self.dataset.get_name() + f"_goal_values.csv"
        f = open(settings.RESULTS_DIR / f"{cumulative_res_filename}", "at")
        if config is None:
            f.write("full," + f"{sumCostBest}\n")
        else:
            f.write("core," + f"{sumCostBest}\n")
        f.close()

        df = pd.DataFrame(bestAlpha)
        df.to_csv(settings.RESULTS_DIR / f"{result_filename}_Alpha.csv", header=False, index=False)
        df = pd.DataFrame(bestBeta)
        df.to_csv(settings.RESULTS_DIR / f"{result_filename}_Beta.csv", header=False, index=False)
        # df = pd.DataFrame(X)
        # df.to_csv(settings.RESULTS_DIR / f"{result_filename}_data.csv", header=False, index=False)
        # I have turned off the saving of Data.csv
        runtimes_df.to_csv(settings.RESULTS_DIR / f"{result_filename}_Alpha_Beta_runtime.csv", header=False,
                           index=False)

    def run(self, parallel=False, n_jobs=-3, add=False, ThreePL=False):
        X = self.dataset.get_X()
        logger.info("Computing IRT on full dataset...")
        self.IRT(X, ThreePL=ThreePL)

        logger.info("Running experiments...")

        def job_function(cur_config):
            logger.info(f"Current experimental config: {cur_config}")
            self.IRT(X, ThreePL=ThreePL, config=cur_config)

        for cur_config in self.get_config_grid():
            job_function(cur_config)

        logger.info("Done.")


class L2SExperiment(BaseExperiment):
    def __init__(
            self,
            dataset: Dataset,
            results_filename,
            sizes,
            num_runs
    ):
        super().__init__(
            num_runs=num_runs,
            sizes=sizes,
            dataset=dataset,
            results_filename=results_filename
        )

    def get_reduced_matrix_and_weights(self, Z, size):
        reduced_matrix, weights = l2s_sampling(Z, size=size)
        # reduced_matrix is only a vector of indexes!!!
        return reduced_matrix, weights


class UniformSamplingExperiment(BaseExperiment):
    def __init__(
            self,
            dataset: Dataset,
            results_filename,
            sizes,
            num_runs
    ):
        super().__init__(
            num_runs=num_runs,
            sizes=sizes,
            dataset=dataset,
            results_filename=results_filename
        )

    def get_reduced_matrix_and_weights(self, Z, size):
        print(Z.shape[0])
        reduced_matrix = np.random.choice(Z.shape[0], size=size, replace=False)
        weights = np.ones(size)
        # reduced_matrix is only a vector of indexes!!!
        return reduced_matrix, weights


class SensitivitySamplingExperiment(BaseExperiment):
    def __init__(
            self,
            dataset: Dataset,
            results_filename,
            sizes,
            num_runs
    ):
        super().__init__(
            num_runs=num_runs,
            sizes=sizes,
            dataset=dataset,
            results_filename=results_filename
        )

    def get_reduced_matrix_and_weights(self, Z, size):

        # Save data input for subprocess
        print(f"Current working directory: {os.getcwd()}")
        np.savetxt(f"eval_k_means_coresets_main/data/input/{self.results_filename}.txt.gz", Z, delimiter=",")

        algorithm_exe_path = "eval_k_means_coresets_main/gs/build/gs"
        random_seed = gen2.generate_random_seed()
        cmd = [
            algorithm_exe_path,
            f"{self.results_filename}",  # Dataset
            pathlib.Path(f"eval_k_means_coresets_main/data/input/{self.results_filename}.txt.gz"),  # Input path
            str(10),  # Number of clusters
            str(size),  # Coreset size
            str(random_seed),  # Random Seed
            "eval_k_means_coresets_main/data/output/",  # Output dir
        ]

        # Remove temporary result files
        file = pathlib.Path("eval_k_means_coresets_main/data/output/results.txt.gz")
        if file.exists():
            file.unlink()
        file = pathlib.Path("eval_k_means_coresets_main/data/output/done.out")
        if file.exists():
            file.unlink()

        print(f"Launching experiment with command:\n '{cmd}'")
        result = subprocess.run(
            args=cmd,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE
        )
        if result.returncode != 0:
            print("Got an error from the C++ Process:")
            print(result.returncode, result.stderr)
            print(result.stdout)

        # Load result file
        selection = np.loadtxt("eval_k_means_coresets_main/data/output/results.txt.gz", delimiter=" ")

        # Fill up to size with random columns
        weights = np.zeros(size)
        temp = selection[:, 0]
        weights[0:temp.shape[0]] = temp
        reduced_matrix = np.random.choice(Z.shape[0], size=size, replace=False)
        reduced_matrix = np.setdiff1d(reduced_matrix, selection[:, 1])
        reduced_matrix = np.concatenate((selection[:, 1].astype(int), reduced_matrix))[0:size]

        return reduced_matrix, weights
