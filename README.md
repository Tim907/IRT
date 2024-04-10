# Scalable Learning of Item Response Theory Models

This is the accompanying code repository for the AISTATS 2024 publication "**Scalable Learning of Item Response Theory Models**" by **Susanne Frick**, **Amer Krivošija** and **Alexander Munteanu**.

The complete version of the paper is in ArXiv: https://arxiv.org/abs/2403.00680

## How to install

1. Clone the repository and navigate into the new directory

   ```bash
   git clone https://github.com/Tim907/IRT
   cd IRT
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

## 1.	How to run the experiments?
   
   a)   Make sure everything is installed properly.
   b)   The Python scripts are copied into a single directory. It can be any directory on the computer.
  	c)   After setting the parameters as wished, the following orders are given in the console (e.g. Anaconda Powershell Prompt):
            pip install .
  	         python ./scripts/run_experiments.py
	d)   The resulting files will be saved in "./experimental-results". By default the "Data" files are not stored. If you want to turn this on, change the lines 201 and 202 in "./IRT/experiments.py".
	e)   After each experiment it is suggested to store the resulting files (the computed optima for the item parameters (file mit "Betas"), and the computed optima for the examinee parameter (file mit "Alphas")) to some other directory. Alternatively double-check before running the next experiment, if a different name basis of the resulting files is provided.
	
## 2.	How to choose if it is 2PL or 3PL experiment?
	
   a)   In the file "./scripts/run_experiments.py", Line 9, set the variable THREE_PL to True, if it is 3PL, otherwise False (for 2PL). 
	b)   In the same file, choose the corresponding labels file in lines 11 and 12 (outcomment the one not used).
	c)   (Optional) To improve the convergence of the 3PL models, it is advised to use a larger matrix for sampling in "./IRT/ls2_sampling.py", that is, replace *1 with *4 in lines 22 and 28. 
	
## 3.	How to set the coreset size?
 
   a)   In the file "./scripts/run_experiments.py",  within the call of an experiment, the line 28: "sizes=(0, 50)" means "use no coreset for the items, and use coreset of size 50 for the examinees. Adapt the first parameter value for the coresets over Items, and the second parameter value for the coresets over Examinees. "0" means no coreset.
	
## 4.	How to run repeated instances of the same experiment?

   In the file "./scripts/run_experiments.py", line 8, set the variable NUM_RUNS to the wished number of the repetitions.
	
## 5.	How to change the name of the labels file to be read?
	
   a)   The name of the input file is given in the file "./IRT/datasets.py", lines 158 and 169. The defaults are "Labels.csv" for 2PL, and Labels_3PL.csv" for 3PL.
	b)   The basis of the name of the output files is set in lines 155 and 166.
	c)   The file called in a) is copied into the directory "./.data-cache".
	
## 6.	How to turn off/on parts of the experiments?
	
   By default are both the complete input and the coreset part included. If some of these are to be skipped, then in the file  "./IRT/experiments.py"
   a)   to turn off the optimization on the complete input, comment out the line 210.
	b)   to turn off the optimization on coresets, comment out the lines 215 to 220.
			
## 7.	How to set the number of optimization iterations?
	
   In the file "./IRT/experiments.py", line 87, set the parameter value in the "range" function.
	
## 8.	How to change the ranges where the solutions are searched for?
	
   In the file "./IRT/experiments.py":
   a)   to set the range of the examinee ability parameters, change the values in lines 117 and 120 (default are (-6.0, 6.0) and (-1.0, -1.0)).
	b)   to set the range of the item parameters, change the values in lines 151 and 153 (default are (0, 5), (-6, 6), and (0.001, 0.499)).
		
## 9.	How to test other sampling techniques, instead of the IRT coresets?
   
   In the file "./IRT/utils.py", is the respective block to be (out)commented. That is:
	a)   Lines 38-43 are for the IRT coresets.
	b)   Lines 30-35 are for the uniform sampling technique.
	c)   Lines 46-51 are for the clustering coresets with sensitivity sampling.
	d)   Lines 54-59 are for the L1-Lewis sampling.
	e)   Lines 62-68 are for the L1-Sensitivity scores.

## 10. How to generate the labels' files?

