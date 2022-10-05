from setuptools import find_packages, setup

setup(
    name="IRT",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["joblib", "numba", "pandas", "scikit-learn", "scipy"],
)
