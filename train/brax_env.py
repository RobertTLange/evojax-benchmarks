"""Train an agent to solve a Brax task."""

import argparse
import os
import shutil

from evojax import Trainer
from evojax.task.brax_task import BraxTask
from evojax.policy import MLPPolicy
from evojax.algo import Strategies
from evojax import util


def main(config):
    log_dir = f"./log/{config.es_name}/brax_{config.env_name}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    logger = util.create_logger(
        name=f"Brax-{config.env_name}", log_dir=log_dir, debug=config.debug
    )

    logger.info(f"EvoJAX Brax ({config.env_name}) Demo")
    logger.info("=" * 30)

    train_task = BraxTask(env_name=config.env_name, test=False)
    test_task = BraxTask(env_name=config.env_name, test=True)
    policy = MLPPolicy(
        input_dim=train_task.obs_shape[0],
        output_dim=train_task.act_shape[0],
        hidden_dims=[32, 32, 32, 32],
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
        normalize_obs=True,
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
        default="configs/ARS/brax_ant.yaml",
        help="Path to configuration yaml.",
    )
    args, _ = parser.parse_known_args()
    config = load_config(args.config_fname, return_dotmap=True)

    if config.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(i) for i in config.gpu_id]
        )
    main(config)
