import numpy as np
import gym
import gym_ple
import tensorflow as tf
import random
from collections import deque
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from keras.layers import Dense, Flatten
from keras import backend as K
import ast
from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from IPython.display import display
from utils import get_state, get_frames, display_frames_as_gif

EPISODES = 20000

def huber_loss(y_true, y_pred, clip_delta=1.0):
    error = y_true - y_pred
    cond  = tf.keras.backend.abs(error) < clip_delta

    squared_loss = 0.5 * tf.keras.backend.square(error)
    linear_loss  = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)
    
    return tf.where(cond, squared_loss, linear_loss)

def huber_loss_mean(y_true, y_pred, clip_delta=1.0):
    return tf.keras.backend.mean(huber_loss(y_true, y_pred, clip_delta))
    
    

class DQNAgent:
    def __init__(self, state_size, action_size, exploration_step=1000000, update_target_rate = 1000, gamma=0.95,
                 lr=0.000001, init_eps=1.0, end_eps=0.01, batch_size=64, train_start=1000):
        #state_size and action_size of env
        self.state_size = state_size
        self.action_size = action_size
        
        
        # reward and score
        self.r = {0: 0.1, 1: 1, -5: -1, -4: -1}
        self.score = {0: 0, 1: 1, -5: 0, -4: 1}
        
        
        #Hyperparamter for DQN
        self.exploration_step = exploration_step
        self.gamma = gamma
        self.learning_rate = lr
        self.eps = 1.0
        self.init_eps = init_eps
        self.end_eps = end_eps
        self.epsilon_decay = (self.init_eps - self.end_eps) / self.exploration_step
        self.batch_size = batch_size
        self.train_start = train_start
        
        # create replay memory using deque
        self.memory = deque(maxlen=100000)

        
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_rate = update_target_rate
        # initialize target model
        self.update_target_model()
        
        self.avg_q = 0.
        
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True,
                                           gpu_options=gpu_options))
        
        K.set_session(self.sess)
        
        self.sess.run(tf.global_variables_initializer())


    def build_model(self):
        model = Sequential()
        model.add(Dense(256, input_dim=self.state_size, activation='relu'))
                        
        model.add(Dense(256, activation='relu'))
                        
        model.add(Dense(self.action_size, activation='linear'))
                        
        model.summary()
        model.compile(loss=huber_loss, optimizer=Adam(lr=self.learning_rate))
        
        return model

    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # Training enable greedy, Testing disable greedy
    def get_action(self, state, greedy=True):
        if greedy:
            if np.random.rand() <= self.eps:
                return random.randrange(self.action_size)
            else:
                q_value = self.model.predict(state)
                return np.argmax(q_value[0])
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])



  
    def train_model(self):
        if len(self.memory) < self.train_start:
            return
        
        # pick random batches in memory
        mini_batch = random.sample(self.memory, self.batch_size)
        
        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        target = self.model.predict(update_input)
        target_val = self.target_model.predict(update_target)
        
        # Getting the q value by using the target model value
        for i in range(self.batch_size):

            target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_val[i]))

        self.model.fit(update_input, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)
        
if __name__ == "__main__":
    
    env = gym.make('FlappyBird-v0')
    # get size of state and action from environment
    state_size = 3
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)

    record, episodes = np.zeros(EPISODES), []
    global_t = 0

    for e in range(1,EPISODES):
        
        reward_sum = 0
        total_score = 0
        state = env.reset()
        step = 0

        while vars(env)['env'].game_state.game_over()==False:
            
            old_state = get_state(env)
            old_state = np.reshape(old_state, [1, state_size])
            action = agent.get_action(old_state)
            
            _, reward, _, _ = env.step(action)
            total_score += agent.score[reward]
            #Redine reward
            real_reward = agent.r[reward]
            reward_sum += real_reward
            
            global_t += 1
            step += 1
            
            agent.avg_q += np.amax(agent.model.predict(old_state)[0])
            
            next_state = get_state(env)
            next_state = np.reshape(next_state, [1, state_size])
            
            # save the sample <s, a, r, s'> to the replay memory
            agent.memory.append((old_state, action, real_reward, next_state,
                               vars(env)['env'].game_state.game_over()))
            
            if agent.eps > agent.end_eps:
                agent.eps -= agent.epsilon_decay
            
            if len(agent.memory) > len(agent.memory):
                agent.replay_memory.popleft()
            
            # every time step do the training
            agent.train_model()
            
            if global_t % agent.update_target_rate == 0:
                    agent.update_target_model()
            
            

            if vars(env)['env'].game_state.game_over():
                # every episode update the target model to be same with model

                # every episode, plot the play time
                record[e] = total_score
                episodes.append(e)
                
                
                print("episode:", e, "  total_score:", total_score, " real_reward", reward_sum, 
                      "  memory length:", len(agent.memory), 
                      "  epsilon:", agent.eps, "  average_q:",agent.avg_q / float(step))
                
                agent.avg_q =  0.


        # save the model every 50 episodes
        if e % 50 == 0:
            agent.model.save_weights("./save_model/flappy.h5")
            
#plt.scatter(np.arange(EPISODES), record, c='r', s=5)
#plt.xlabel('Episodes')
#plt.ylabel('Score for each episode')
#plt.savefig("flappy_dqn.png")