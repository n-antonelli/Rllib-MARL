import matplotlib.pyplot as plt
import numpy as np
import json

MARL = 'sharing'    # 'independent' 'sharing' 'critic'
ENV = 'cartpole'    # 'cartpole' 'waterworld' ' '
with open('C:\\Users\\Usuario\\ray_results\\PPO_2024-10-29_12-41-20\\PPO_env_3ed6e_00000_0_2024-10-29_12-41-21\\result.json', 'r') as file:
    data = []
    for episode in file:
        data.append(json.loads(episode))
    #data = json.load(file)

recompensa_episodio_p0 = []
for episode in range(len(data)):
    recompensa_episodio_p0.append(data[episode]['env_runners']['policy_reward_min']['p0'])
recompensa_episodio = []
for episode in range(len(data)):
    recompensa_episodio.append(data[episode]['env_runners']['episode_reward_mean'])

# Para guardar todos los valores de un episodio
#episodio = 0
#if ENV == 'cartpole':
#    recompensa_episodio_p0 = data[episodio]['env_runners']['hist_stats']['policy_p0_reward']
#    if MARL != 'sharing':
#        recompensa_episodio_p1 = data[episodio]['env_runners']['hist_stats']['policy_p1_reward']
#        recompensa_episodio_p2 = data[episodio]['env_runners']['hist_stats']['policy_p2_reward']

#elif ENV == 'waterworld':


plt.plot(range(0,len(recompensa_episodio)),np.asarray(recompensa_episodio))
#plt.plot(range(0,len(recompensa_episodio_p1)),np.asarray(recompensa_episodio_p1))
#plt.plot(range(0,len(recompensa_episodio_p2)),np.asarray(recompensa_episodio_p2))
plt.show()
#print(np.linspace(0,len(recompensa_episodio),endpoint=False))
#print(np.asarray(recompensa_episodio))
#print(data[2]['env_runners']['hist_stats']['episode_reward'])