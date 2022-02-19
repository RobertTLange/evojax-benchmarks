from evojax.policy import MLPPolicy
from evojax.policy.convnet import ConvNetPolicy


def setup_problem(config, logger):
    if config.problem_type == "cartpole_easy":
        return setup_cartpole(config, False)
    elif config.problem_type == "cartpole_hard":
        return setup_cartpole(config, True)
    elif config.problem_type == "brax":
        return setup_brax(config)
    elif config.problem_type == "mnist":
        return setup_mnist(config, logger)
    elif config.problem_type == "waterworld":
        return setup_waterworld(config)
    elif config.problem_type == "waterworld_ma":
        return setup_waterworld_ma(config)


def setup_cartpole(config, hard=False):
    from evojax.task.cartpole import CartPoleSwingUp

    train_task = CartPoleSwingUp(test=False, harder=hard)
    test_task = CartPoleSwingUp(test=True, harder=hard)
    policy = MLPPolicy(
        input_dim=train_task.obs_shape[0],
        hidden_dims=[config.hidden_size] * 2,
        output_dim=train_task.act_shape[0],
    )
    return train_task, test_task, policy


def setup_brax(config):
    from evojax.task.brax_task import BraxTask

    train_task = BraxTask(env_name=config.env_name, test=False)
    test_task = BraxTask(env_name=config.env_name, test=True)
    policy = MLPPolicy(
        input_dim=train_task.obs_shape[0],
        output_dim=train_task.act_shape[0],
        hidden_dims=[32, 32, 32, 32],
    )
    return train_task, test_task, policy


def setup_mnist(config, logger):
    from evojax.task.mnist import MNIST

    policy = ConvNetPolicy(logger=logger)
    train_task = MNIST(batch_size=config.batch_size, test=False)
    test_task = MNIST(batch_size=config.batch_size, test=True)
    return train_task, test_task, policy


def setup_waterworld(config, max_steps=500):
    from evojax.task.waterworld import WaterWorld

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
    return train_task, test_task, policy


def setup_waterworld_ma(config, num_agents=16, max_steps=500):
    from evojax.task.ma_waterworld import MultiAgentWaterWorld

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
    return train_task, test_task, policy
