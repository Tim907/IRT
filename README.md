# Scalable Learning of Item Response Theory Models

This is the accompanying code repository for the AISTATS 2024 publication "**Scalable Learning of Item Response Theory Models**" by **Susanne Frick**, **Amer Krivošija** and **Alexander Munteanu**.

The complete version of the paper is in ArXiv: https://arxiv.org/abs/2403.00680

## How to install

1. Clone the repository and navigate into the new directory

   ```bash
   git clone https://github.com/Tim907/IRT
   cd oblivious-sketching-varreglogreg
   ```

2. Create and activate a new virtual environment
   
   on Unix:
   ```bash
   python -m venv venv
   . ./venv/bin/activate
   ```
   on Windows:
   ```bash
   python -m venv venv
   venv\Scripts\activate.bat
   ```

3. Install the package locally

   ```bash
   pip install .
   ```

## How to run the experiments

The `scripts` directory contains python scripts that can be
used to run the experiments.
Just make sure, that everything is installed properly.

## How to run 2PL and 3PL models
To change between 2PL and 3PL IRT models, you can use the script contained in the `scripts` directory as a template to modify which datasets are used. Change the boolean flag `ThreePL` to True/False as needed.
