from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import random
import numpy as np
import pandas as pd
from operator import add # + operasyonu, state'lerde mpa yaparken kullanmış

# Deep Q Network implementasyonu
class DQNAgent(object):

    def __init__(self):
        self.reward = 0
        self.gamma = 0.9 # "discounted reward factor": sonraki state'in reward'ın etkisini azaltıyor?
        self.learning_rate = 0.0005
        self.model = self.network()
        self.epsilon = 0 # epsilon-greedy
        self.memory = []
        # self.model = self.network("weights.hdf5")
        # self.dataframe = pd.DataFrame()
        # self.short_memory = np.array([])
        # self.agent_target = 1
        # self.agent_predict = 0
        # self.actual = []

    # Önce boolean assertion'lar yapıp, daha sonra kolayca okuyyabilmek için binary'e çeviriyor.
    # Output array: [danger front, danger left, danger left, move left, move right, move up, move down, food left, food right, food up, food down]
    def get_state(self, game, player, food):
        # state'in ilk üç elemanı ilerde, sağda ve solda tehlike olup olmadığına bakıyor. Tehlikeden kastına kendi kuyruğu dahil mi bilmiyorum.
        # bu kod basitleştirilebilir
        state = [
            (player.x_change == 20 and player.y_change == 0 and ((list(map(add, player.position[-1], [20, 0])) in player.position) or
            player.position[-1][0] + 20 >= (game.game_width - 20))) or (player.x_change == -20 and player.y_change == 0 and ((list(map(add, player.position[-1], [-20, 0])) in player.position) or
            player.position[-1][0] - 20 < 20)) or (player.x_change == 0 and player.y_change == -20 and ((list(map(add, player.position[-1], [0, -20])) in player.position) or
            player.position[-1][-1] - 20 < 20)) or (player.x_change == 0 and player.y_change == 20 and ((list(map(add, player.position[-1], [0, 20])) in player.position) or
            player.position[-1][-1] + 20 >= (game.game_height-20))),  # danger straight

            (player.x_change == 0 and player.y_change == -20 and ((list(map(add,player.position[-1],[20, 0])) in player.position) or
            player.position[ -1][0] + 20 > (game.game_width-20))) or (player.x_change == 0 and player.y_change == 20 and ((list(map(add,player.position[-1],
            [-20,0])) in player.position) or player.position[-1][0] - 20 < 20)) or (player.x_change == -20 and player.y_change == 0 and ((list(map(
            add,player.position[-1],[0,-20])) in player.position) or player.position[-1][-1] - 20 < 20)) or (player.x_change == 20 and player.y_change == 0 and (
            (list(map(add,player.position[-1],[0,20])) in player.position) or player.position[-1][
             -1] + 20 >= (game.game_height-20))),  # danger right

             (player.x_change == 0 and player.y_change == 20 and ((list(map(add,player.position[-1],[20,0])) in player.position) or
             player.position[-1][0] + 20 > (game.game_width-20))) or (player.x_change == 0 and player.y_change == -20 and ((list(map(
             add, player.position[-1],[-20,0])) in player.position) or player.position[-1][0] - 20 < 20)) or (player.x_change == 20 and player.y_change == 0 and (
            (list(map(add,player.position[-1],[0,-20])) in player.position) or player.position[-1][-1] - 20 < 20)) or (
            player.x_change == -20 and player.y_change == 0 and ((list(map(add,player.position[-1],[0,20])) in player.position) or
            player.position[-1][-1] + 20 >= (game.game_height-20))), # danger left


            player.x_change == -20, # move left
            player.x_change == 20,  # move right
            player.y_change == -20, # move up
            player.y_change == 20,  # move down
            food.x_food < player.x, # food left
            food.x_food > player.x, # food right
            food.y_food < player.y, # food up
            food.y_food > player.y  # food down
            ]

        for i in range(len(state)):
            if state[i]:
                state[i]=1
            else:
                state[i]=0

        return np.asarray(state)

    def set_reward(self, player, crash):
        self.reward = 0
        if crash:
            self.reward = -10
            return self.reward
        if player.eaten:
            self.reward = 10
        return self.reward

    def network(self, weights=None):
        model = Sequential()
        model.add(Dense(output_dim=120, activation='relu', input_dim=11))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=120, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=120, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=3, activation='softmax'))
        opt = Adam(self.learning_rate)
        model.compile(loss='mse', optimizer=opt)

        # elimizde eğitilmiş model varsa onu yükle
        if weights:
            model.load_weights(weights)
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Telegram 
    # Experience replay
    def replay_new(self, memory):
        if len(memory) > 1000:
            minibatch = random.sample(memory, 1000)
        else:
            minibatch = memory
        
        # aslında train_short_memory fonksiyonu
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]))[0])
            target_f = self.model.predict(np.array([state]))
            target_f[0][np.argmax(action)] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)

    def train_short_memory(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape((1, 11)))[0])
        target_f = self.model.predict(state.reshape((1, 11)))
        target_f[0][np.argmax(action)] = target
        self.model.fit(state.reshape((1, 11)), target_f, epochs=1, verbose=0)