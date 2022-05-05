from typing import TYPE_CHECKING
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import backend as K

# Hyperparameters
VID_CONV_FILTERS = 128
HIST_CONV_FILTERS = 128
HIDDEN_LAYER_SIZE = 128

LR = 1e-4  # Lower lr stabilises training greatly
ACTION_EPS = 1e-4
GAMMA = 0.99
LOSS_CLIPPING=0.1
ENTROPY_LOSS =0.1


def proximal_policy_optimization_loss(advantage, old_prediction):
    def loss(y_true, y_pred):
        prob = K.sum(y_true * y_pred, axis=-1)
        old_prob = K.sum(y_true * old_prediction, axis=-1)
        r = prob/(old_prob + 1e-10)
        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage) + ENTROPY_LOSS * -(prob * K.log(prob + 1e-10)))
    return loss


class PPOAgent:
    def __init__(self, network_history_len:int, available_video_sizes_count:int):
        self.network_history_len = network_history_len
        self.available_video_sizes_count = available_video_sizes_count
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        

    def build_actor(self):
        # network throughput measurements for the last k video chunks
        feature_historical_network_throughput = keras.layers.Input(shape=(self.network_history_len,1))
        # chunk download times for the last k video chunks
        feature_historical_chunk_download_time = keras.layers.Input(shape=(self.network_history_len,1))
        # a vector of m available sizes for the next video chunk
        feature_available_video_sizes = keras.layers.Input(shape=(self.available_video_sizes_count,1))
        # the current buffer level
        feature_buffer_level = keras.layers.Input(shape=(1,))
        # number of chunks remaining in the video
        feature_remaining_chunk_count = keras.layers.Input(shape=(1,))
        # bitrate at which the last chunk was downloaded
        feature_last_chunk_bitrate = keras.layers.Input(shape=(1,))

        convolved_historical_network_throughput = keras.layers.Conv1D(HIST_CONV_FILTERS, 4, activation='relu')(feature_historical_network_throughput)
        convolved_historical_chunk_download_time = keras.layers.Conv1D(HIST_CONV_FILTERS, 4, activation='relu')(feature_historical_chunk_download_time)
        convolved_available_video_sizes = keras.layers.Conv1D(VID_CONV_FILTERS, 4, activation='relu')(feature_available_video_sizes)

        hidden_layer_input = keras.layers.Concatenate(axis=0)([
          keras.layers.Flatten()(convolved_historical_network_throughput),
          keras.layers.Flatten()(convolved_historical_chunk_download_time),
          keras.layers.Flatten()(convolved_available_video_sizes),
          feature_buffer_level,
          feature_remaining_chunk_count,
          feature_last_chunk_bitrate
        ])

        hidden_layer = keras.layers.Dense(HIDDEN_LAYER_SIZE, activation='relu')(hidden_layer_input)

        out_actions = keras.layers.Dense(self.available_video_sizes_count, activation='softmax')(hidden_layer)

        # old prediction is a 1-hot vector
        old_prediction = keras.layers.Input(shape=self.available_video_sizes_count)

        # advantage captures how better an action is compared to the others at a given state
        # https://theaisummer.com/Actor_critics/
        advantage = keras.layers.Input(shape=(1,))

        model = keras.Model(
            inputs=[
                # ppo features
                old_prediction,
                advantage,
                # real features
                feature_historical_network_throughput,
                feature_historical_chunk_download_time,
                feature_available_video_sizes,
                feature_buffer_level,
                feature_remaining_chunk_count,
                feature_last_chunk_bitrate
            ],
            outputs=[out_actions]
        )

        model.compile(
            optimizer=keras.optimizer.Adam(lr=LR),
            loss=[
                proximal_policy_optimization_loss(
                    advantage=advantage,
                    old_prediction=old_prediction
                 )
            ]
        )
        model.summary()

        return model


    def build_critic(self):
        # network throughput measurements for the last k video chunks
        feature_historical_network_throughput = keras.layers.Input(shape=(self.network_history_len,1))
        # chunk download times for the last k video chunks
        feature_historical_chunk_download_time = keras.layers.Input(shape=(self.network_history_len,1))
        # a vector of m available sizes for the next video chunk
        feature_available_video_sizes = keras.layers.Input(shape=(self.available_video_sizes_count,1))
        # the current buffer level
        feature_buffer_level = keras.layers.Input(shape=(1,))
        # number of chunks remaining in the video
        feature_remaining_chunk_count = keras.layers.Input(shape=(1,))
        # bitrate at which the last chunk was downloaded
        feature_last_chunk_bitrate = keras.layers.Input(shape=(1,))

        convolved_historical_network_throughput = keras.layers.Conv1D(HIST_CONV_FILTERS, 4, activation='relu')(feature_historical_network_throughput)
        convolved_historical_chunk_download_time = keras.layers.Conv1D(HIST_CONV_FILTERS, 4, activation='relu')(feature_historical_chunk_download_time)
        convolved_available_video_sizes = keras.layers.Conv1D(VID_CONV_FILTERS, 4, activation='relu')(feature_available_video_sizes)

        hidden_layer_input = keras.layers.Concatenate(axis=0)([
          keras.layers.Flatten()(convolved_historical_network_throughput),
          keras.layers.Flatten()(convolved_historical_chunk_download_time),
          keras.layers.Flatten()(convolved_available_video_sizes),
          feature_buffer_level,
          feature_remaining_chunk_count,
          feature_last_chunk_bitrate
        ])

        hidden_layer = keras.layers.Dense(HIDDEN_LAYER_SIZE, activation='relu')(hidden_layer_input)

        out_actions = keras.layers.Dense(self.available_video_sizes_count, activation='linear')(hidden_layer)

        model = keras.Model(
            inputs=[
                feature_historical_network_throughput,
                feature_historical_chunk_download_time,
                feature_available_video_sizes,
                feature_buffer_level,
                feature_remaining_chunk_count,
                feature_last_chunk_bitrate
            ],
            outputs=[out_actions]
        )

        model.compile(optimizer=keras.optimizer.Adam(lr=LR), loss='mse')

        return model

    def predict(self):
        self.actor.predict(

    def get_action(self):
        p = self.actor.predict([self.observation.reshape(1, NUM_STATE), DUMMY_VALUE, DUMMY_ACTION])
        if self.val is False:

            action = np.random.choice(NUM_ACTIONS, p=np.nan_to_num(p[0]))
        else:
            action = np.argmax(p[0])
        action_matrix = np.zeros(NUM_ACTIONS)
        action_matrix[action] = 1
        return action, action_matrix, p

    def transform_reward(self):
        if self.val is True:
            self.writer.add_scalar('Val episode reward', np.array(self.reward).sum(), self.episode)
        else:
            self.writer.add_scalar('Episode reward', np.array(self.reward).sum(), self.episode)
        for j in range(len(self.reward) - 2, -1, -1):
            self.reward[j] += self.reward[j + 1] * GAMMA

    def get_batch(self):
        batch = [[], [], [], []]

        tmp_batch = [[], [], []]
        while len(batch[0]) < BUFFER_SIZE:
            if CONTINUOUS is False:
                action, action_matrix, predicted_action = self.get_action()
            else:
                action, action_matrix, predicted_action = self.get_action_continuous()
            observation, reward, done, info = self.env.step(action)
            self.reward.append(reward)

            tmp_batch[0].append(self.observation)
            tmp_batch[1].append(action_matrix)
            tmp_batch[2].append(predicted_action)
            self.observation = observation

            if done:
                self.transform_reward()
                if self.val is False:
                    for i in range(len(tmp_batch[0])):
                        obs, action, pred = tmp_batch[0][i], tmp_batch[1][i], tmp_batch[2][i]
                        r = self.reward[i]
                        batch[0].append(obs)
                        batch[1].append(action)
                        batch[2].append(pred)
                        batch[3].append(r)
                tmp_batch = [[], [], []]
                self.reset_env()

        obs, action, pred, reward = np.array(batch[0]), np.array(batch[1]), np.array(batch[2]), np.reshape(np.array(batch[3]), (len(batch[3]), 1))
        pred = np.reshape(pred, (pred.shape[0], pred.shape[2]))
        return obs, action, pred, reward

