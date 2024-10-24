# @OldAPIStack

# ***********************************************************************************
# IMPORTANT NOTE: This script uses the old API stack and will soon be replaced by
# `ray.rllib.examples.multi_agent.pettingzoo_shared_value_function.py`!
# ***********************************************************************************


"""An example of implementing a centralized critic with ObservationFunction.

The advantage of this approach is that it's very simple and you don't have to
change the algorithm at all -- just use callbacks and a custom model.
However, it is a bit less principled in that you have to change the agent
observation spaces to include data that is only used at train time.

See also: centralized_critic.py for an alternative approach that instead
modifies the policy to add a centralized value function.
"""

import numpy as np
from gymnasium.spaces import Dict, Discrete
import argparse
import os

from ray import air, tune
from ray.air.constants import TRAINING_ITERATION
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.examples._old_api_stack.models.centralized_critic_models import (
    YetAnotherCentralizedCriticModel,
    YetAnotherTorchCentralizedCriticModel,
)
from ray.rllib.examples.envs.classes.two_step_game import TwoStepGame
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
)
from ray.rllib.utils.test_utils import check_learning_achieved

parser = argparse.ArgumentParser()
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "torch"],
    default="torch",
    help="The DL framework specifier.",
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--stop-iters", type=int, default=80, help="Number of iterations to train." #100
)
parser.add_argument(
    "--stop-timesteps", type=int, default=100000, help="Number of timesteps to train." #100000
)
parser.add_argument(
    "--stop-reward", type=float, default=7.99, help="Reward at which we stop training." #7.99
)


class FillInActions(DefaultCallbacks):
    """Fills in the opponent actions info in the training batches."""

    def on_postprocess_trajectory(
        self,
        worker,
        episode,
        agent_id,
        policy_id,
        policies,
        postprocessed_batch,
        original_batches,
        **kwargs,
    ):
        to_update = postprocessed_batch[SampleBatch.CUR_OBS]
        other_id = 1 if agent_id == 0 else 0
        action_encoder = ModelCatalog.get_preprocessor_for_space(Discrete(2))

        # set the opponent actions into the observation
        _, opponent_batch = original_batches[other_id]
        opponent_actions = np.array(
            [action_encoder.transform(a) for a in opponent_batch[SampleBatch.ACTIONS]]
        )
        to_update[:, -2:] = opponent_actions


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


if __name__ == "__main__":
    args = parser.parse_args()

    ModelCatalog.register_custom_model(
        "cc_model",
        YetAnotherTorchCentralizedCriticModel
        if args.framework == "torch"
        else YetAnotherCentralizedCriticModel,
    )

    action_space = Discrete(2)
    observer_space = Dict(
        {
            "own_obs": Discrete(6), # Estado + ¿acción tomada?
            # These two fields are filled in by the CentralCriticObserver, and are
            # not used for inference, only for training.
            "opponent_obs": Discrete(6), # Estado + ¿acción tomada?
            "opponent_action": Discrete(2),
        }
    )

    config = (
        PPOConfig()
        .environment(TwoStepGame)
        .framework(args.framework)
        .env_runners(
            batch_mode="complete_episodes",
            num_env_runners=0,
            # TODO(avnishn) make a new example compatible w connectors.
            enable_connectors=False,
        )
        .callbacks(FillInActions)
        .training(model={"custom_model": "cc_model"})
        .multi_agent(
            policies={
                "pol1": (None, observer_space, action_space, {}),
                "pol2": (None, observer_space, action_space, {}),
            },
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: "pol1"
            if agent_id == 0
            else "pol2",
            observation_fn=central_critic_observer,
        )
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )

    stop = {
        TRAINING_ITERATION: args.stop_iters,
        NUM_ENV_STEPS_SAMPLED_LIFETIME: args.stop_timesteps,
        f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}": args.stop_reward,
    }

    tuner = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=air.RunConfig(stop=stop, verbose=1),
    )
    results = tuner.fit()

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)
