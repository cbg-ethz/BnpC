# BnpC
Bayesian non-parametric clustering (BnpC) of binary data with missing values and uneven error rates.

BnpC is a novel non-parametric method to cluster individual cells into clones and infer their genotypes based on their noisy mutation profiles.
BnpC employs a Chinese Restaurant Process prior to handle the unknown number of clonal populations. The model introduces a combination of Gibbs sampling, a modified non-conjugate split-merge move and Metropolis-Hastings updates to explore the joint posterior space of all parameters. Furthermore, it employs a novel estimator, which accounts for the shape of the posterior distribution, to predict the clones and genotypes.

A preprint version of the corresponsing paper can be found on [bioRxiv](https://doi.org/10.1101/2020.01.15.907345 "Borgsmueller et al.")

# Contents
- [Installation](#Installation)
- [Usage](#Usage)
- [Example data](#Example-data)         

# Requirements
- Python 3.X

# Installation
## Clone repository
First, download BnpC from github and change to the directory:
```bash
git clone https://github.com/cbg-ethz/BnpC
cd BnpC
```

## Create conda environment (optional)
First, create a new environment named "BnpC":
```bash
conda create --name BnpC python=3.7
```

Second, source it:
```bash
source activate BnpC
```

## Install requirements
Use pip to install the requirements:
```bash
pip install -r requirements.txt
``` 

Now you are ready to run **BnpC**!

# Usage
The BnpC wrapper script `run_BnpC.py` can be run with the following shell command: 
```bash
python run_BnpC.py <INPUT_DATA> [-t] [-ad] [-fd] [-ad_m] [-ad_sd] [-fd_m] [-fd_sd] [-s] [-r] [-ls] [-b] [-pa] [-pb] [-smp] [-cup] [-e] [-dpa] [-tc] [-td] [-tr] [-np] [-par] [-ci] [-seed] [-si] [-o]
```

## Input
BnpC requires a binary matrix as input, where each row corresponds with a mutations and each columns with a cell. 
All matrix entries must be of the following: 0|1|3/" ", where 0 indicates the absence of a mutation, 1 the presence, and a 3 or empty element a missing value.

> ## Note
> If your data is arranged in the transposed way (cells = columns, rows = mutations), use the `-t` argument.

## Arguments
### Input Data Arguments
- `<str>`, Path to the input data. 
- `-t <flag>`, If set, the input matrix is transposed. 
- `-ad <float>`, Replace <float\> with the fixed error rate for false negatives.
- `-fd <float>`, Replace <float\> with the fixed error rate for false positives.
- `-ad_m <float>`, Replace <float\> with the mean for the prior for the false negative rate. 
- `-ad_sd <float>`, Replace <float\> with the standard deviation for the prior for the false negative rate.
- `-fd_m <float>`, Replace <float\> with the mean for the prior for the false positive rate.
- `-fd_sd <float>`, Replace <float\> with the standard deviation for the prior for the false positive rate.
        

### MCMC Settings Arguments
- `-s <int>`, Number of MCMC steps.
- `-r <int>`, Runtime in minutes. If set, steps argument is overwritten.
- `-ls <float>`, Lugsail batch means estimator as convergence diagnostics [Vats and Flegal, 2018].
- `-b  <float>`, Ratio of MCMC steps discarded as burn-in.
- `-pa <float>`, Alpha value of the Beta function used as parameter prior. 
- `-pb <float>`, Beta value of the Beta function used as parameter prior. 
- `-smp <float>`, Probability to do a split/merge step instead of Gibbs sampling. 
- `-cup  <float>`, Probability of updating the CRP concentration parameter. 
- `-e +<str>`, Estimator(s) for inferrence. If more than one, seperate by whitespace. Options = posterior|ML|MAP|MPEAR.
- `-dpa <int>`, Alpha value of the Beta function used as prior for the concentration parameter of the CRP. 

### Simulation Arguments
- `-tc <str>`, Path to the true clusters assignments to compare clustering methods. 
- `-td <str>`, Path to the true/raw data/genotypes.  
- `-tr <str>`, Path to the tree file (in .gv format) used for data generation.

### Plotting Arguments  
- `-np <flag>`, If set, no plots are generated. 
- `-par <flag>`, If set, cluster parameter traces are plotted (time consuming).
- `-ci <flag>`, If set, the cluster incidence for each annotated sample containing a set of cells are plotted. 

### Other Arguments
- `--seed <int>`, Seed used for random number generation.
- `-si <flag>`, If set, no status messages are printed to stdout. 
- `-o <str>`, Path to an output directory.

# Example data

Lets employ the toy dataset that one can find in the `data` folder (data.csv) to understand the functionality of the different arguments. First go to the folder and activate the environment:
        
        cd /path/to/crp_clustering
        conda activate environment_name

BnpC can run in three different settings: 
1. Number of steps. Runs for the given number of MCMC steps. Arument: -s
2. Running time limit. Every MCMC the time is tracked and the method stops after the introduced time is achieved. Argument: -r 
3. Lugsail for convergence diagnosis. The chain is terminated if the estimator undercuts a threshold defined by a significance level of 0.05 and a user defined float between [0,1], comparable to the half-width of the confidence interval in sample size calculation for a one sample t-test. Reasonal values = 0.1, 0.2, 0.3. Argument: -ls

The simplest way to run the BnpC is to leave every argument as default and hence only the path to the data needs to be given. In this case BnpC runs in the setting 1. 
```bash
python CRP_clustering documentation/data/data.csv 
```
If the error rates are known for a particular sequenced data (e.g fd = 0.0001 and ad = 0.3), one can run BnpC with fixed error rates by: 
```bash
python CRP_clustering data/data.csv -fd 0.0001 -ad 0.3
```
On the other hand, if errors are not known one can leave it blank as in the first case or if there is some intuition add the mean and standard deviation priors for the method to learn them: 
```bash
python CRP_clustering documentation/data/data.csv -fd_m 0.0001 -ad_m 0.3 -fd_sd 0.000001 -ad_sd 0.05
```
Additional MCMC arguments can be employed to allow faster convergence. Among other options: 
- Reduce burnin to include more posterior samples in the estimation. Example: -b 0.2, discard 20 % of the total MCMC steps.
- Adapt split-merge probability to better explore the posterior landscape. Example: -smp 0.33, 1 out of every 3 steps will be a split-merge move. 
- Adjust the Dirchlet Process alpha which accounts for the probability of starting a new cluster. Example: -dpa 10. Increasing the value, leads to a larger probability of starting a new cluster in the cell assignment step. 


        
