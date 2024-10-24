import matplotlib.pyplot as plt
import numpy as np
import json


with open('C:\\Users\\Usuario\\ray_results\\PPO_2024-10-24_11-45-30\\PPO_env_9d878_00000_0_2024-10-24_11-45-30\\result.json', 'r') as file:
    data = json.load(file)
recompensa_episodio = data['env_runners']['hist_stats']['episode_reward']
recompensa_episodio_p0 = data['env_runners']['hist_stats']['policy_p0_reward']
recompensa_episodio_p1 = data['env_runners']['hist_stats']['policy_p1_reward']
recompensa_episodio_p2 = data['env_runners']['hist_stats']['policy_p2_reward']

#plt.plot(range(0,len(recompensa_episodio_p0)),np.asarray(recompensa_episodio_p0))
#plt.plot(range(0,len(recompensa_episodio_p1)),np.asarray(recompensa_episodio_p1))
#plt.plot(range(0,len(recompensa_episodio_p2)),np.asarray(recompensa_episodio_p2))
#plt.show()
#print(np.linspace(0,len(recompensa_episodio),endpoint=False))
#print(np.asarray(recompensa_episodio))
print(data['env_runners']['hist_stats'])