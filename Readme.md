# Utilities for Benchmarking EvoJAX Algorithms 

## TODOs
- [ ] Rewrite waterworld envs for config setup
- [ ] Rewrite MNIST for config setup
- [ ] Add training script for brax from notebook example
- [ ] Tune ARS on waterworld/brax
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
python train/brax_ant.py -config configs/<es>/brax_ant.yaml
python train/mnist.py -config configs/<es>/mnist.yaml
```

### Expected Runtimes on 4 A100 GPUs

- Cartpole (easy) - 5 Minutes
- Cartpole (hard) - 7 Minutes
- Waterworld (1000 iters) - 30 Minutes 
- Waterworld (MA - 2000 iters) - 20 Minutes
- MNIST (5000 iters) - 
- Brax Ant - 

## Hyperparameter Tuning with `mle-hyperopt`

- TBC

## Benchmark Results

### Augmented Random Search


|   | Benchmarks | Parameters | Results |
|---|---|---|---|
CartPole (easy) | 	900 (max_iter=2000)|[Link]()| 902.107 |
CartPole (hard)	| 600 (max_iter=2000)|[Link]()| 666.6442 |
Waterworld	| 6 (max_iter=2000)	 |[Link]()||
Waterworld (MA)	| 2 (max_iter=5000)	| [Link]()||
Brax Ant |	3000 (max_iter=1000) |[Link]()||
MNIST	| 90.0 (max_iter=5000)	| [Link]()||
