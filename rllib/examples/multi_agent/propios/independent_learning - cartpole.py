"""Runs the PettingZoo Waterworld env in RLlib using independent multi-agent learning.

See: https://pettingzoo.farama.org/environments/sisl/waterworld/
for more details on the environment.


How to run this script
----------------------
`python [script file name].py --enable-new-api-stack --num-agents=2`

Control the number of agents and policies (RLModules) via --num-agents and
--num-policies.

This works with hundreds of agents and policies, but note that initializing
many policies might take some time.

For debugging, use the following additional command line options
`--no-tune --num-env-runners=0`
which should allow you to set breakpoints anywhere in the RLlib code and
have the execution stop there for inspection and debugging.

For logging to your WandB account, use:
`--wandb-key=[your WandB API key] --wandb-project=[some project name]
--wandb-run-name=[optional: WandB run name (within the defined project)]`


Results to expect
-----------------
The above options can reach a combined reward of 0.0 or more after about 500k env
timesteps. Keep in mind, though, that due to the separate value functions (and
learned policies in general), one agent's gain (in per-agent reward) might cause the
other agent's reward to decrease at the same time. However, over time, both agents
should simply improve.

+---------------------+------------+-----------------+--------+------------------+
| Trial name          | status     | loc             |   iter |   total time (s) |
|---------------------+------------+-----------------+--------+------------------+
| PPO_env_a82fc_00000 | TERMINATED | 127.0.0.1:28346 |    124 |          363.599 |
+---------------------+------------+-----------------+--------+------------------+

+--------+-------------------+--------------------+--------------------+
|     ts |   combined reward |   reward pursuer_1 |   reward pursuer_0 |
+--------+-------------------+--------------------+--------------------|
| 496000 |           2.24542 |           -34.6869 |            36.9324 |
+--------+-------------------+--------------------+--------------------+

Note that the two agents (`pursuer_0` and `pursuer_1`) are optimized on the exact same
objective and thus differences in the rewards can be attributed to weight initialization
(and sampling randomness) only.
"""

from pettingzoo.sisl import waterworld_v4

from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.tune.registry import get_trainable_cls, register_env
from ray.rllib.examples.envs.classes.multi_agent import MultiAgentCartPole

parser = add_rllib_example_script_args(
    default_iters=200,
    default_timesteps=100000,
    default_reward=600.0,
)
# TODO (sven): This arg is currently ignored (hard-set to 2).
parser.add_argument("--num-policies", type=int, default=2)

if __name__ == "__main__":
    args = parser.parse_args()
    args.num_agents = 2
#    args.enable_new_api_stack = True

#    assert args.num_agents > 0, "Must set --num-agents > 0 when running this script!"
#    assert (
#        args.enable_new_api_stack
#    ), "Must set --enable-new-api-stack when running this script!"

    register_env("env",lambda _: MultiAgentCartPole(config={"num_agents": args.num_agents}))

    # Policies are called just like the agents (exact 1:1 mapping).
    #policies = {f"p{i}" for i in range(args.num_agents)}   # Crea una política para cada agente

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
            # Exact 1:1 mapping from AgentID to ModuleID.
            policy_mapping_fn=lambda aid, *a, **kw: f"p{aid}",  # Mapea la política con el agente
        )
        .training(
            vf_loss_coeff=0.005,
        )
        .rl_module(
            model_config_dict={"vf_share_layers": True},
            rl_module_spec=MultiRLModuleSpec(
                module_specs={p: RLModuleSpec() for p in {f"p{i}" for i in range(args.num_agents)}},
            ),
        )
    )

    run_rllib_example_script_experiment(base_config, args)
