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
GAMMA = 0.99

# PPO2
LOSS_CLIPPING=0.1
ENTROPY_LOSS =0.1

class PPOAgent:
    def __init__(self, network_history_len:int, available_video_sizes_count:int):
        self.network_history_len = network_history_len
        self.available_video_sizes_count = available_video_sizes_count
        self.actor = self.__build_actor()
        self.critic = self.__build_critic()

    # Private
    def __build_actor(self):
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
            inputs=[(
                # ppo features
                advantage,
                old_prediction,
                # real features
                feature_historical_network_throughput,
                feature_historical_chunk_download_time,
                feature_available_video_sizes,
                feature_buffer_level,
                feature_remaining_chunk_count,
                feature_last_chunk_bitrate
            )],
            outputs=[out_actions]
        )

        def actor_loss(y_true, y_pred):
            prob = K.sum(y_true * y_pred, axis=-1)
            old_prob = K.sum(y_true * old_prediction, axis=-1)
            r = prob/(old_prob + 1e-10)
            return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage) + ENTROPY_LOSS * -(prob * K.log(prob + 1e-10)))

        model.compile(
            optimizer=keras.optimizer.Adam(lr=LR),
            loss=actor_loss
        )
        model.summary()

        return model

    # Private
    def __build_critic(self):
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

        value = keras.layers.Dense(1, activation='linear')(hidden_layer)

        model = keras.Model(
            inputs=[(
                feature_historical_network_throughput,
                feature_historical_chunk_download_time,
                feature_available_video_sizes,
                feature_buffer_level,
                feature_remaining_chunk_count,
                feature_last_chunk_bitrate
            )],
            outputs=[value]
        )

        model.compile(optimizer=keras.optimizer.Adam(lr=LR), loss='mse')

        return model

    def save(self, actor_path:str, critic_path:str):
        self.actor.save(actor_path);
        self.critic.save(critic_path);

    def load(self, actor_path:str, critic_path:str):
        self.actor = keras.models.load_model(actor_path);
        self.critic = keras.models.load_model(critic_path);

    def get_network_params(self):
        return [
          self.actor.get_weights(),
          self.critic.get_weights(),
        ]

    def set_network_params(self, weights):
        self.actor.set_weights(weights[0])
        self.critic.set_weights(weights[1])

    def predict(
        self,
        state_batch: list[tuple[
            list[float], # historical_network_throughput
            list[float], # historical_chunk_download_time
            list[float], # available_video_sizes
            float, # buffer_level
            float, # remaining_chunk_count
            float # last_chunk_bitrate
        ]],
    ):
        p = self.actor.predict((
          tf.None
          state_batch

        ))
        return p

    def train(
        self,
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
    ):
        pass


    # value function
    def compute_v(
        self,
        state_batch: list[tuple[
            list[float], # historical_network_throughput
            list[float], # historical_chunk_download_time
            list[float], # available_video_sizes
            float, # buffer_level
            float, # remaining_chunk_count
            float # last_chunk_bitrate
        ]],
        reward_batch: list[float],
        terminal: bool
    ):
        assert len(state_batch) == len(reward_batch)

        ba_size= len(state_batch)
        R_batch = np.zeros([len(reward_batch), 1])

        if terminal:
            R_batch[-1, 0] = 0  # terminal state
        else:    
            v_batch = self.critic.predict(state_batch)
            R_batch[-1, 0] = v_batch[-1, 0]  # boot strap from last state
        # Use GAMMA to decay value 
        for t in reversed(range(ba_size - 1)):
            R_batch[t, 0] = reward_batch[t] + GAMMA * R_batch[t + 1, 0]

        return list(R_batch)

