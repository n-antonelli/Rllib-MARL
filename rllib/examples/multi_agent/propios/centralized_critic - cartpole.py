
from pettingzoo.sisl import waterworld_v4
import argparse
import numpy as np
import os
import ray

from ray.rllib.evaluation.postprocessing import compute_advantages
from ray.rllib.examples._old_api_stack.models.centralized_critic_models import (
    YetAnotherCentralizedCriticModel,
    YetAnotherTorchCentralizedCriticModel,
)
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.utils.tf_utils import make_tf_callable
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.tune.registry import get_trainable_cls, register_env
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv


from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from gymnasium.spaces import Dict, Discrete, MultiDiscrete, Box
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.air.constants import TRAINING_ITERATION
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
)
from ray.rllib.examples.envs.classes.multi_agent import MultiAgentCartPole
tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

def central_critic_observer(agent_obs, **kw):
    """Rewrites the agent obs to include opponent data for training."""

    new_obs = {
        0: {
            "own_obs": agent_obs[0],
            "opponent_obs": agent_obs[1],
            "opponent_action": 0,  # filled in by FillInActions
        },
        1: {
            "own_obs": agent_obs[1],
            "opponent_obs": agent_obs[0],
            "opponent_action": 0,  # filled in by FillInActions
        },
    }
    return new_obs

parser = add_rllib_example_script_args(
    default_iters=200,
    default_timesteps=100000,
    default_reward=400.0,
)
# TODO (sven): This arg is currently ignored (hard-set to 2).
parser.add_argument("--num-policies", type=int, default=2)

if __name__ == "__main__":
    args = parser.parse_args()
    args.num_agents = 3



    # Prueba para testear
    args.as_test = True




    """ Espacio de acción y de observación escrito a mano
    action_space = Discrete(1)
    observer_space = Dict(
        {
            "own_obs": Box(low = -2.4, high = 2.4,shape = [1,4]),
            # These two fields are filled in by the CentralCriticObserver, and are
            # not used for inference, only for training.
            "opponent_obs": Box(low = -2.4, high = 2.4,shape = [1,4]),
            "opponent_action": Discrete(1),
        }
    )
    """
    register_env("env",lambda _: MultiAgentCartPole(config={"num_agents": args.num_agents}))

    base_config = (
        get_trainable_cls(args.algo)
        .get_default_config()
        .environment("env")
        .env_runners(
            # TODO (sven): MAEnvRunner does not support vectorized envs yet
            #  due to gym's env checkers and non-compatability with RLlib's
            #  MultiAgentEnv API.
            num_envs_per_env_runner=1
            if args.num_agents > 0
            else 20,
        )
        .multi_agent(
            policies={f"p{i}" for i in range(args.num_agents)},
            policy_mapping_fn=lambda aid, *a, **kw: f"p{aid}",
            observation_fn=central_critic_observer,
        )
        .training(
            model={
                "vf_share_layers": True,
            },
            vf_loss_coeff=0.005,
        )
        .api_stack(enable_rl_module_and_learner=True)
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
               module_specs={p: RLModuleSpec() for p in {f"p{i}" for i in range(args.num_agents)}},
            ),
        )
    )
    run_rllib_example_script_experiment(base_config, args) # success_metric = "env_runners/episode_return_mean" # la métrica a evaluar
