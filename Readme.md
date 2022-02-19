# Utilities for Benchmarking EvoJAX Algorithms 

This repository contains benchmark results, helper scripts, ES configurations and logs for testing the performance of evolutionary strategies in [`evojax`](https://github.com/google/evojax/). These can come in handy, when aiming to merge a new JAX-based ES into the projects.

## Installation

```
pip install evojax mle-hyperopt
```

## Running the Benchmarks for an Evolution Strategy

1. Fork `evojax`. 
2. Add your strategy to `algo` and the `Strategies` wrapper in the `__init__.py` file.
3. Add the base task configurations for you ES to `configs/<es>/`.
4. Get compute access and execute the individual training runs for the base configurations via:

```
python train.py -config configs/<es>/cartpole_easy.yaml
python train.py -config configs/<es>/cartpole_hard.yaml
python train.py -config configs/<es>/waterworld.yaml
python train.py -config configs/<es>/waterworld_ma.yaml
python train.py -config configs/<es>/brax_ant.yaml
python train.py -config configs/<es>/mnist.yaml
```

5. [OPTIONAL] Tune hyperparameters using [`mle-hyperopt`](https://github.com/mle-infrastructure/mle-hyperopt). Here is an example for running a grid search for ARS over different learning rates and perturbation standard deviations via

```
mle-search train.py -base configs/ARS/mnist.yaml -search configs/ARS/search.yaml -iters 25
```

### Expected Runtimes on 4 A100s

- Cartpole (easy - 1000 iters) - ~5 Minutes
- Cartpole (hard - 1000 iters) - ~7 Minutes
- Waterworld (500 iters) - ~20 Minutes 
- Waterworld (MA - 2000 iters) - ~20 Minutes
- MNIST (2000 iters) - ~12 Minutes
- Brax Ant (300 iters) - ~25 Minutes 

## Benchmark Results

### Augmented Random Search


|   | Benchmarks | Parameters | Results |
|---|---|---|---|
CartPole (easy) | 	900 (max_iter=1000)|[Link](https://github.com/RobertTLange/evojax-benchmarks/blob/main/configs/ars/cartpole_easy.yaml)| 902.107 |
CartPole (hard)	| 600 (max_iter=1000)|[Link](https://github.com/RobertTLange/evojax-benchmarks/blob/main/configs/ars/cartpole_hard.yaml)| 666.6442 |
Waterworld	| 6 (max_iter=500)	 |[Link](https://github.com/RobertTLange/evojax-benchmarks/blob/main/configs/ars/waterworld.yaml)| 6.1300 |
Waterworld (MA)	| 2 (max_iter=2000)	| [Link](https://github.com/RobertTLange/evojax-benchmarks/blob/main/configs/ars/waterworld_ma.yaml)| 1.4831 |
Brax Ant |	3000 (max_iter=300) |[Link](https://github.com/RobertTLange/evojax-benchmarks/blob/main/configs/ars/brax_ant.yaml)| 3298.9746 |
MNIST	| 90.0 (max_iter=2000)	| [Link](https://github.com/RobertTLange/evojax-benchmarks/blob/main/configs/ars/mnist.yaml)| 0.9610 |


## TODOs
- [ ] Merge all training scripts into single one?
- [ ] Add more strategies to evojax
- [ ] Add additional configs/logs for other strategies

