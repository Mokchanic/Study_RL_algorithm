class PPO_define():
    learning_rate = 0.0005
    gamma = 0.98
    lmbda = 0.95
    eps_clip = 0.1
    K_epoch = 3
    T_horizon = 20
    INPUT = 4
    MLP_DIMS_PI = [256, 2]
    MLP_DIMS_V  = [256, 1]