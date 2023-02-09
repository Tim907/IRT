import abc
import logging
import math
from pathlib import Path

import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale

from . import optimizer, settings

logger = logging.getLogger(settings.LOGGER_NAME)

_rng = np.random.default_rng()


def make_Z(X, y):

    # multiply row-wise by y
    Z = np.multiply(X, y[:, np.newaxis])

    return Z


class Dataset(abc.ABC):
    def __init__(self, use_caching, cache_dir=None):
        self.use_caching = use_caching
        if cache_dir is None:
            cache_dir = settings.DATA_DIR
        self.cache_dir = cache_dir

        if use_caching and not self.cache_dir.exists():
            self.cache_dir.mkdir()

        self.X = None
        self.y = None
        self.beta_opt = None

    @abc.abstractmethod
    def load_X_y(self):
        pass

    @abc.abstractmethod
    def get_name(self):
        pass

    def _load_X_y_cached(self):
        if not self.use_caching:
            logger.info("Loading X and y...")
            X, y = self.load_X_y()
            logger.info("Done.")
            return X, y

        X_path = self.get_binary_path_X()
        y_path = self.get_binary_path_y()
        if X_path.exists() and y_path.exists():
            logger.info(
                f"Loading cached versions of X and y found at {X_path} and {y_path}..."
            )
            X = np.load(X_path)
            y = np.load(y_path)
            logger.info("Done.")
            return X, y

        logger.info("Loading X and y...")
        X, y = self.load_X_y()
        logger.info("Done.")
        np.save(X_path, X)
        np.save(y_path, y)
        logger.info(f"Saved X and y at {X_path} and {y_path}.")

        return X, y

    def _get_beta_opt_cached(self):
        if not self.use_caching:
            logger.info("Computing beta_opt...")
            beta_opt = optimizer.optimize_2PL(self.get_X(), self.get_y()).x
            logger.info("Done.")
            return beta_opt

        beta_opt_path = self.get_binary_path_beta_opt()
        if beta_opt_path.exists():
            logger.info(
                f"Loading cached version of beta_opt found at {beta_opt_path}..."
            )
            beta_opt = np.load(beta_opt_path)
            logger.info("Done.")
            return beta_opt

        logger.info("Computing beta_opt...")
        beta_opt = optimizer.optimize_2PL(self.get_X(), self.get_y()).x
        logger.info("Done.")
        np.save(beta_opt_path, beta_opt)
        logger.info(f"Saved beta_opt at {beta_opt_path}.")

        return beta_opt

    def _assert_data_loaded(self):
        if self.X is None or self.y is None:
            self.X, self.y = self._load_X_y_cached()

    def get_binary_path_X(self) -> Path:
        return self.cache_dir / f"{self.get_name()}_X.npy"

    def get_binary_path_y(self) -> Path:
        return self.cache_dir / f"{self.get_name()}_y.npy"

    def get_binary_path_beta_opt(self) -> Path:
        return self.cache_dir / f"{self.get_name()}_beta_opt.npy"

    def get_X(self):
        self._assert_data_loaded()
        return self.X

    def get_y(self):
        self._assert_data_loaded()
        return self.y

    def get_n(self):
        self._assert_data_loaded()
        return self.X.shape[0]

    def get_d(self):
        self._assert_data_loaded()
        return self.X.shape[1]

    def get_beta_opt(self):
        if self.beta_opt is None:
            self.beta_opt = self._get_beta_opt_cached()

        return self.beta_opt



class BasicDataset(Dataset):

    def __init__(self, n, m, use_caching=True):
        super().__init__(use_caching=use_caching)
        self.X = np.reshape(random.choices([-1, 1], k=m*n, weights=(0.4, 0.6)), (m, n))

    def get_name(self):
        return "basic"

    def get_X(self):
        return self.X

    def load_X_y(self):
        pass


class RealisticDataset(Dataset):

    def get_name(self):
        return "realistic"

    def get_X(self):
        return pd.read_csv(settings.DATA_DIR / "Labels.csv", delimiter=";", header=0, index_col=0).T.to_numpy() * 2 - 1

    def load_X_y(self):
        pass