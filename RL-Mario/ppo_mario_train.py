import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
import os
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv, WarpFrame
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.logger import configure
MARIO_ENV='SuperMarioBros-1-1-v0'

MODEL_DIR='models/'

####### new directories added
LOG_DIR = 'logs/'
new_logger = configure(LOG_DIR, ["stdout", "csv", "tensorboard"])

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)


################################################################## For PPO trainning parameters ##################################################################
LEARNING_RATE = 2.5e-5
GAMMA = 0.9
LAMBDA = 0.9

MAX_STEPS = 15e6
#### Mario environments

def mario_env(train:bool = False) -> gym.Env:
    env = gym_super_mario_bros.make(MARIO_ENV)
    # Get action... can change into RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = MaxAndSkipEnv(env, skip=4)
    env = WarpFrame(env, width=84, height=84)
    if train:
        return Monitor(env, LOG_DIR)
    else:
        return env
# ?PPO
model = PPO(
    "CnnPolicy",
    mario_env(train=True), 
    learning_rate=LEARNING_RATE,
    batch_size=32,
    gamma=GAMMA,
    gae_lambda=LAMBDA,
    create_eval_env=True,
    ent_coef=0.02,
    vf_coef=1.0,
    tensorboard_log=f'{LOG_DIR}_PPO_v2',
    verbose=1)

model.set_logger(new_logger)
checkpoint_callback = CheckpointCallback(save_freq=50000, save_path='checkpointcallback/')

eval_callback = EvalCallback(
    mario_env(train=True),
    best_model_save_path=f'./{MODEL_DIR}/best_model/',
    log_path=f'./{MODEL_DIR}/best_model/results/',
    eval_freq=5e4,
    deterministic=False,
    verbose=1)

callback = CallbackList([checkpoint_callback, eval_callback])


model.learn(total_timesteps=MAX_STEPS, callback=callback, reset_num_timesteps=False, tb_log_name='PPO_v2')
model.save(f'{MODEL_DIR}/{MAX_STEPS}')