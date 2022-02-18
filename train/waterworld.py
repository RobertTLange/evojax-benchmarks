# Copyright 2022 The EvoJAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train an agent to solve the WaterWorld task.

In this task, an agent (yellow) tries to catch as much food (green) as possible
while avoiding poisons (red).
This task is based on:
https://cs.stanford.edu/people/karpathy/reinforcejs/waterworld.html

Example command to run this script: `python train_waterworld.py --gpu-id=0`
"""

import argparse
import os
import shutil
import jax

from evojax.task.waterworld import WaterWorld
from evojax.policy.mlp import MLPPolicy
from evojax.algo import Strategies
from evojax import Trainer
from evojax import util


def main(config):
    log_dir = f"./log/{config.es_name}/water_world"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    logger = util.create_logger(
        name="WaterWorld", log_dir=log_dir, debug=config.debug
    )
    logger.info("EvoJAX WaterWorld")
    logger.info("=" * 30)

    max_steps = 500
    train_task = WaterWorld(test=False, max_steps=max_steps)
    test_task = WaterWorld(test=True, max_steps=max_steps)
    policy = MLPPolicy(
        input_dim=train_task.obs_shape[0],
        hidden_dims=[
            config.hidden_size,
        ],
        output_dim=train_task.act_shape[0],
        output_act_fn="softmax",
    )
    solver = Strategies[config.es_name](
        **config.es_config.toDict(),
        param_size=policy.num_params,
        seed=config.seed,
    )

    # Train.
    trainer = Trainer(
        policy=policy,
        solver=solver,
        train_task=train_task,
        test_task=test_task,
        max_iter=config.max_iter,
        log_interval=config.log_interval,
        test_interval=config.test_interval,
        n_repeats=config.n_repeats,
        n_evaluations=config.num_tests,
        seed=config.seed,
        log_dir=log_dir,
        logger=logger,
    )
    trainer.run(demo_mode=False)

    # Test the final model.
    src_file = os.path.join(log_dir, "best.npz")
    tar_file = os.path.join(log_dir, "model.npz")
    shutil.copy(src_file, tar_file)
    trainer.model_dir = log_dir
    trainer.run(demo_mode=True)


if __name__ == "__main__":
    from mle_logging import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-config",
        "--config_fname",
        type=str,
        default="configs/ARS/cartpole_easy.yaml",
        help="Path to configuration yaml.",
    )
    args, _ = parser.parse_known_args()
    config = load_config(args.config_fname, return_dotmap=True)

    if config.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(i) for i in config.gpu_id]
        )
    main(config)
