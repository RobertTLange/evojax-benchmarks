# Utilities for Benchmarking EvoJAX Algorithms 

This repository contains benchmark results, helper scripts, ES configurations and logs for testing the performance of evolutionary strategies in [`evojax`](https://github.com/google/evojax/). These can come in handy, when aiming to merge a new JAX-based ES into the projects.

## Installation

```
pip install evojax
pip install git+https://github.com/mle-infrastructure/mle-hyperopt.git@main
pip install git+https://github.com/mle-infrastructure/mle-logging.git@main
```

## Running the Benchmarks for an Evolution Strategy

1. Fork `evojax`. 
2. Add your strategy to `algo` and the `Strategies` wrapper in the `__init__.py` file.
3. Add the base task configurations for you ES to `configs/<es>/`.
4. Get compute access and execute the individual training runs for the base/default configurations via:

```
python train.py -config configs/<es>/cartpole_easy.yaml
python train.py -config configs/<es>/cartpole_hard.yaml
python train.py -config configs/<es>/waterworld.yaml
python train.py -config configs/<es>/waterworld_ma.yaml
python train.py -config configs/<es>/brax_ant.yaml
python train.py -config configs/<es>/mnist.yaml
```

5. [OPTIONAL] Tune hyperparameters using [`mle-hyperopt`](https://github.com/mle-infrastructure/mle-hyperopt). Here is an example for running a grid search for ARS over different learning rates and perturbation standard deviations via:

```
mle-search train.py -base configs/ARS/mnist.yaml -search configs/ARS/search.yaml -iters 25 -log log/ARS/mnist/
```

This will sequentially execute 25 ARS-MNIST evolution runs for a grid of different learning rates and standard deviations. After the search has completed, you can access the search log at `log/ARS/mnist/search_log.yaml`

## Benchmark Results

### OpenES


|   | Benchmarks | Parameters | Results (Avg) |
|---|---|---|---|
CartPole (easy) | 	900 (max_iter=1000)|[Link](https://github.com/RobertTLange/evojax-benchmarks/blob/main/configs/OpenES/cartpole_easy.yaml)| 929.4153 |
CartPole (hard)	| 600 (max_iter=1000)|[Link](https://github.com/RobertTLange/evojax-benchmarks/blob/main/configs/OpenES/cartpole_hard.yaml)| 604.6940 |
MNIST	| 90.0 (max_iter=2000)	| [Link](https://github.com/RobertTLange/evojax-benchmarks/blob/main/configs/OpenES/mnist.yaml)| 0.9669 |
Brax Ant |	3000 (max_iter=1200) |[Link](https://github.com/RobertTLange/evojax-benchmarks/blob/main/configs/OpenES/brax_ant.yaml)| 6726.2100 |
Waterworld	| 6 (max_iter=500)	 | - | - |
Waterworld (MA)	| 2 (max_iter=2000)	| - | - |


*Note*: For the brax environment I reduced the population size from 1024 to 256 and increased the search iterations by the same factor (300 to 1200) in the main run. For the grid search I used a population size of 256 but with 500 iterations.


| Cartpole-Easy  | Cartpole-Hard | MNIST | Brax-Ant
|---|---|---|---|
![](figures/OpenES/cartpole_easy.png) | ![](figures/OpenES/cartpole_hard.png) | ![](figures/OpenES/mnist.png) |![](figures/OpenES/brax.png) |

### Augmented Random Search


|   | Benchmarks | Parameters | Results |
|---|---|---|---|
CartPole (easy) | 	900 (max_iter=1000)|[Link](https://github.com/RobertTLange/evojax-benchmarks/blob/main/configs/ARS/cartpole_easy.yaml)| 902.107 |
CartPole (hard)	| 600 (max_iter=1000)|[Link](https://github.com/RobertTLange/evojax-benchmarks/blob/main/configs/ARS/cartpole_hard.yaml)| 666.6442 |
Waterworld	| 6 (max_iter=500)	 |[Link](https://github.com/RobertTLange/evojax-benchmarks/blob/main/configs/ARS/waterworld.yaml)| 6.1300 |
Waterworld (MA)	| 2 (max_iter=2000)	| [Link](https://github.com/RobertTLange/evojax-benchmarks/blob/main/configs/ARS/waterworld_ma.yaml)| 1.4831 |
Brax Ant |	3000 (max_iter=300) |[Link](https://github.com/RobertTLange/evojax-benchmarks/blob/main/configs/ARS/brax_ant.yaml)| 3298.9746 |
MNIST	| 90.0 (max_iter=2000)	| [Link](https://github.com/RobertTLange/evojax-benchmarks/blob/main/configs/ARS/mnist.yaml)| 0.9610 |


| Cartpole-Easy  | Cartpole-Hard | MNIST |
|---|---|---|
![](figures/ARS/cartpole_easy.png) | ![](figures/ARS/cartpole_hard.png) | ![](figures/ARS/mnist.png) |