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

"""Train a population of agents to solve the WaterWorld task."""

import argparse
import os
import shutil
import jax
import jax.numpy as jnp

from evojax.task.ma_waterworld import MultiAgentWaterWorld
from evojax.policy.mlp import MLPPolicy
from evojax.algo import Strategies
from evojax import Trainer
from evojax import util


def main(config):
    log_dir = f"./log/{config.es_name}/water_world_ma"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    logger = util.create_logger(
        name="MultiAgentWaterWorld", log_dir=log_dir, debug=config.debug
    )

    logger.info("EvoJAX MultiAgentWaterWorld")
    logger.info("=" * 30)

    num_agents = 16
    max_steps = 500
    train_task = MultiAgentWaterWorld(
        num_agents=num_agents, test=False, max_steps=max_steps
    )
    test_task = MultiAgentWaterWorld(
        num_agents=num_agents, test=True, max_steps=max_steps
    )
    policy = MLPPolicy(
        input_dim=train_task.obs_shape[-1],
        hidden_dims=[
            config.hidden_size,
        ],
        output_dim=train_task.act_shape[-1],
        output_act_fn="softmax",
    )
    solver = Strategies[config.es_name](
        **config.es_config.toDict(),
        pop_size=num_agents,
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
        n_evaluations=num_agents,
        n_repeats=config.n_repeats,
        test_n_repeats=config.num_tests,
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

    # Visualize the policy.
    task_reset_fn = jax.jit(test_task.reset)
    policy_reset_fn = jax.jit(policy.reset)
    step_fn = jax.jit(test_task.step)
    action_fn = jax.jit(policy.get_actions)
    best_params = jnp.repeat(
        trainer.solver.best_params[None, :], num_agents, axis=0
    )
    key = jax.random.PRNGKey(0)[None, :]

    task_state = task_reset_fn(key)
    policy_state = policy_reset_fn(task_state)
    screens = []
    for _ in range(max_steps):
        num_tasks, num_agents = task_state.obs.shape[:2]
        task_state = task_state.replace(
            obs=task_state.obs.reshape((-1, *task_state.obs.shape[2:]))
        )
        action, policy_state = action_fn(task_state, best_params, policy_state)
        action = action.reshape(num_tasks, num_agents, *action.shape[1:])
        task_state = task_state.replace(
            obs=task_state.obs.reshape(
                num_tasks, num_agents, *task_state.obs.shape[1:]
            )
        )
        task_state, reward, done = step_fn(task_state, action)
        screens.append(MultiAgentWaterWorld.render(task_state))

    gif_file = os.path.join(log_dir, "water_world_ma.gif")
    screens[0].save(
        gif_file, save_all=True, append_images=screens[1:], duration=40, loop=0
    )
    logger.info("GIF saved to {}.".format(gif_file))


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
    main(config)
