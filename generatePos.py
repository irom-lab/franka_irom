import numpy as np

seed = 10000
np.random.seed(seed)

numTrials = 15
obj_x = np.random.uniform(low=0.55, 
                            high=0.65, 
                            size=(numTrials, 1))
obj_y = np.random.uniform(low=-0.10, 
                        high=0.10, 
                        size=(numTrials, 1))
obj_yaw = np.random.uniform(low=-45, 
                            high=45, 
                            size=(numTrials, 1))
all_config = np.hstack((obj_x, obj_y, obj_yaw))
for i in range(numTrials):
    print(i)
    print(all_config[i])

