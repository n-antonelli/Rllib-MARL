import matplotlib.pyplot as plt
import numpy as np
import json

MARL = 'critic'    # 'independent' 'sharing' 'critic'
ENV = 'cartpole'    # 'cartpole' 'waterworld' '
# CRITIC --> 2 ag - obs mod - PPO_2024-11-01_10-11-07\\PPO_env_c19a9_00000_0_2024-11-01_10-11-07
# CRITIC - PPOConf --> 2 ag - obs mod - PPOConf - PPO_2024-11-04_11-11-32\\PPO_env_b1411_00000_0_2024-11-04_11-11-32
# SHARING --> 2 ag - sharing - PPO_2024-11-01_12-05-31\\PPO_env_bcba8_00000_0_2024-11-01_12-05-31
# IND --> 2 ag - ind - PPO_2024-11-04_09-12-00\\PPO_env_fe8ba_00000_0_2024-11-04_09-12-00
with open('C:\\Users\\Usuario\\ray_results\\PPO_2024-11-05_10-23-38\PPO_env_2cd92_00000_0_2024-11-05_10-23-41\\result.json', 'r') as file:
    data = []
    for episode in file:
        data.append(json.loads(episode))
    #data = json.load(file)
figr, axr = plt.subplots()
figl, axl = plt.subplots()
reward_episode = []
reward_agents = {}
loss_episode = []
loss_agents = {}
for agent in range(len(data[0]['env_runners']['policy_reward_mean'])):
    reward_episode = [] # vacío la recompensa general para que solo quede una al final
    reward_episode_agent = []
    loss_episode = []  # vacío la pérdida general para que solo quede una al final
    loss_episode_agent = []
    for episode in range(len(data)):
        reward_episode.append(data[episode]['env_runners']['episode_reward_mean'])
        reward_episode_agent.append(data[episode]['env_runners']['policy_reward_mean'][f'p{agent}'])
        if MARL == 'critic':
            loss_episode.append(data[episode]['info']['learner']['__all_modules__']['total_loss'])
            loss_episode_agent.append(data[episode]['info']['learner'][f'p{agent}']['policy_loss'])
        else:
            loss_episode.append(data[episode]['info']['learner'][f'p{agent}']['learner_stats']['total_loss'])
            loss_episode_agent.append(data[episode]['info']['learner'][f'p{agent}']['learner_stats']['policy_loss'])

    reward_agents[f'pol_{agent}'] = reward_episode_agent

    loss_agents[f'pol_{agent}'] = loss_episode_agent

    # Para guardar todos los valores de un episodio
    # episodio = 0
    #if ENV == 'cartpole':
    #    recompensa_episodio_p0 = data[episodio]['env_runners']['hist_stats']['policy_p0_reward']
    #    if MARL != 'sharing':
    #        recompensa_episodio_p1 = data[episodio]['env_runners']['hist_stats']['policy_p1_reward']
    #        recompensa_episodio_p2 = data[episodio]['env_runners']['hist_stats']['policy_p2_reward']

    #elif ENV == 'waterworld':


    axr.plot(range(0, len(reward_agents[f'pol_{agent}'])), np.asarray(reward_agents[f'pol_{agent}']), label=f'rew_pol_{agent}')
    axl.plot(range(0,len(loss_agents[f'pol_{agent}'])),np.asarray(loss_agents[f'pol_{agent}']), label=f'loss_pol_{agent}')
# Reward
axr.plot(range(0,len(reward_episode)),np.asarray(reward_episode), label='total_rew')
# Loss
axl.plot(range(0,len(loss_episode)),np.asarray(loss_episode), label='total_loss')
axr.legend(loc='best')
axl.legend(loc='best')
plt.show()

"""
agent_obs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
import copy
new_obs = {}
for agente in range(len(agent_obs)):
    opponent_obs = copy.deepcopy(agent_obs)  # Crear una nueva variable para no modificar original
    #print(opponent_obs, ' ', agent_obs)
    del opponent_obs[agente]  # Modificar nueva variable para que no tenga su propia observación
    print(opponent_obs)
    new_obs[agente] = {"own_obs": agent_obs[agente],
                       "opponent_obs": opponent_obs,
                       "opponent_action": 0,
                       }

print(new_obs)"""