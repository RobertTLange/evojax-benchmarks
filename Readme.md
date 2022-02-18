# Utilities for Benchmarking EvoJAX Algorithms 

## TODOs
- [x] Rewrite waterworld envs for config setup
- [x] Rewrite MNIST for config setup
- [ ] Add training script for brax from notebook example
- [ ] Tune ARS on waterworld/brax/mnist
- [ ] Open PR for ARS & share repository
- [ ] Add `mle-hyperopt` simple pipeline
- [ ] Add more strategies to evojax
- [ ] Add additional configs/logs for other strategies

## Installation

```
pip install evojax mle-logging
```

## Running the Benchmark for an Evolution Strategy

1. Fork `evojax`. And add your strategy to `algo` and the `Strategies` wrapper.
2. Add the task configurations to `configs/<es>/` and execute the training runs via:

```
python train/cartpole.py -config configs/<es>/cartpole_easy.yaml
python train/cartpole.py -config configs/<es>/cartpole_hard.yaml
python train/waterworld.py -config configs/<es>/waterworld.yaml
python train/waterworld_ma.py -config configs/<es>/waterworld_ma.yaml
python train/brax_env.py -config configs/<es>/brax_ant.yaml
python train/mnist.py -config configs/<es>/mnist.yaml
```

### Expected Runtimes on 4 A100 GPUs

- Cartpole (easy - 1000 iters) - 5 Minutes
- Cartpole (hard - 1000 iters) - 7 Minutes
- Waterworld (1000 iters) - 30 Minutes 
- Waterworld (MA - 2000 iters) - 20 Minutes
- MNIST (2000 iters) - 12 Minutes
- Brax Ant - 

## Hyperparameter Tuning with `mle-hyperopt`

- TBC

## Benchmark Results

### Augmented Random Search


|   | Benchmarks | Parameters | Results |
|---|---|---|---|
CartPole (easy) | 	900 (max_iter=1000)|[Link](https://github.com/RobertTLange/evojax-benchmarks/blob/main/configs/ars/cartpole_easy.yaml)| 902.107 |
CartPole (hard)	| 600 (max_iter=1000)|[Link](https://github.com/RobertTLange/evojax-benchmarks/blob/main/configs/ars/cartpole_hard.yaml)| 666.6442 |
Waterworld	| 6 (max_iter=2000)	 |[Link](https://github.com/RobertTLange/evojax-benchmarks/blob/main/configs/ars/waterworld.yaml)||
Waterworld (MA)	| 2 (max_iter=5000)	| [Link](https://github.com/RobertTLange/evojax-benchmarks/blob/main/configs/ars/waterworld_ma.yaml)||
Brax Ant |	3000 (max_iter=1000) |[Link](https://github.com/RobertTLange/evojax-benchmarks/blob/main/configs/ars/brax_ant.yaml)||
MNIST	| 90.0 (max_iter=2000)	| [Link](https://github.com/RobertTLange/evojax-benchmarks/blob/main/configs/ars/mnist.yaml)||
