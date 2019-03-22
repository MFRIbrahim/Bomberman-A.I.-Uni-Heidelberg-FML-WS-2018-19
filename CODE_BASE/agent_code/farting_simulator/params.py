"""
This file stores all the hyperparameters and settings for our agent
"""

params = {
    "CUDA": True,
    "target_newtork_update_freq": 10000,
    "save_network_freq": 10,

    # -------------------- hyperparameters --------------------
    "initial_eps": 0.24,
    "final_eps": 0.01,
    "learning_rate":1e-6,
    "gamma": 0.99,

    "replay_mem_capacity": 10000,
    "replay_mem_batchsize": 64,

    "framestack_capacity": 4

}

GRAYSCALE_VALUES = {
    "empty": 0,
    "wall": 25,
    "crate": 50,
    "bomb": 75,
    "coin": 100,
    "explosion": 125,
    "enemy": 150,
    "enemy_and_bomb": 175,
    "self": 200,
    "self_and_bomb": 225

}