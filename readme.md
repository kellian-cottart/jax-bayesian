# JAX Bayesian NN

## Description

This is the repository holding the commands to run for the results in [paper].

## Environment installation

To install the environment needed to run the code, please use conda. Make sure you are running either of the commands in the main directory.

```bash
conda env create -f environment.yml
conda activate binarized
```

## Commands

Commands for the paper:

- MNIST OOD PMNIST

```bash
python main.py --config mlp-bayesian-bgd-mnist.json --n_iterations 5 --extra
python main.py --config mlp-bayesian-mesu-mnist.json --n_iterations 5 --extra
```

- PMNIST 200

```bash
python main.py --config mlp-bayesian-bgd-pmnist-200.json --n_iterations 5 --extra
python main.py --config mlp-bayesian-mesu-pmnist-200.json --n_iterations 5 --extra
```
