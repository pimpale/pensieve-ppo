from typing import TYPE_CHECKING
import math
import numpy as np
import numpy.typing as npt
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
    def __init__(self, network_history_len:int, available_video_bitrates_count:int):
        self.network_history_len = network_history_len
        self.available_video_bitrates_count = available_video_bitrates_count
        self.actor = self.__build_actor()
        self.critic = self.__build_critic()

    # Private
    def __build_actor(self):
        # network throughput measurements for the last k video chunks
        feature_historical_network_throughput = keras.layers.Input(shape=(self.network_history_len,1))
        # chunk download times for the last k video chunks
        feature_historical_chunk_download_time = keras.layers.Input(shape=(self.network_history_len,1))
        # a vector of m available sizes for the next video chunk
        feature_available_video_bitrates = keras.layers.Input(shape=(self.available_video_bitrates_count,1))
        # the current buffer level
        feature_buffer_level = keras.layers.Input(shape=(1,))
        # number of chunks remaining in the video
        feature_remaining_chunk_count = keras.layers.Input(shape=(1,))
        # bitrate at which the last chunk was downloaded
        feature_last_chunk_bitrate = keras.layers.Input(shape=(1,))

        convolved_historical_network_throughput = keras.layers.Conv1D(HIST_CONV_FILTERS, 4, activation='relu')(feature_historical_network_throughput)
        convolved_historical_chunk_download_time = keras.layers.Conv1D(HIST_CONV_FILTERS, 4, activation='relu')(feature_historical_chunk_download_time)
        convolved_available_video_bitrates = keras.layers.Conv1D(VID_CONV_FILTERS, 4, activation='relu')(feature_available_video_bitrates)

        hidden_layer_input = keras.layers.Concatenate(axis=0)([
          keras.layers.Flatten()(convolved_historical_network_throughput),
          keras.layers.Flatten()(convolved_historical_chunk_download_time),
          keras.layers.Flatten()(convolved_available_video_bitrates),
          feature_buffer_level,
          feature_remaining_chunk_count,
          feature_last_chunk_bitrate
        ])

        hidden_layer = keras.layers.Dense(HIDDEN_LAYER_SIZE, activation='relu')(hidden_layer_input)

        out_actions = keras.layers.Dense(self.available_video_bitrates_count, activation='softmax')(hidden_layer)

        # old prediction is a 1-hot vector
        old_prediction = keras.layers.Input(shape=self.available_video_bitrates_count)

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
                feature_available_video_bitrates,
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
            optimizer=keras.optimizers.Adam(lr=LR),
            loss=actor_loss
        )
        model.summary()

        return model

    # Private
    # The critic attempts to learn the advantage
    def __build_critic(self):
        # network throughput measurements for the last k video chunks
        feature_historical_network_throughput = keras.layers.Input(shape=[self.network_history_len])
        # chunk download times for the last k video chunks
        feature_historical_chunk_download_time = keras.layers.Input(shape=[self.network_history_len])
        # a vector of m available sizes for the next video chunk
        feature_available_video_bitrates = keras.layers.Input(shape=[self.available_video_bitrates_count])
        # the current buffer level
        feature_buffer_level = keras.layers.Input(shape=[1])
        # number of chunks remaining in the video
        feature_remaining_chunk_count = keras.layers.Input(shape=[1])
        # bitrate at which the last chunk was downloaded
        feature_last_chunk_bitrate = keras.layers.Input(shape=[1])

        convolved_historical_network_throughput = keras.layers.Conv1D(HIST_CONV_FILTERS, 4, activation='relu')(feature_historical_network_throughput)
        convolved_historical_chunk_download_time = keras.layers.Conv1D(HIST_CONV_FILTERS, 4, activation='relu')(feature_historical_chunk_download_time)
        convolved_available_video_bitrates = keras.layers.Conv1D(VID_CONV_FILTERS, 4, activation='relu')(feature_available_video_bitrates)

        hidden_layer_input = keras.layers.Concatenate()([
          keras.layers.Flatten()(convolved_historical_network_throughput),
          keras.layers.Flatten()(convolved_historical_chunk_download_time),
          keras.layers.Flatten()(convolved_available_video_bitrates),
          feature_buffer_level,
          feature_remaining_chunk_count,
          feature_last_chunk_bitrate
        ])

        hidden_layer = keras.layers.Dense(HIDDEN_LAYER_SIZE, activation='relu')(hidden_layer_input)

        value = keras.layers.Dense(1, activation='linear')(hidden_layer)

        model = keras.Model(
            inputs=[
                feature_historical_network_throughput,
                feature_historical_chunk_download_time,
                feature_available_video_bitrates,
                feature_buffer_level,
                feature_remaining_chunk_count,
                feature_last_chunk_bitrate
            ],
            outputs=[value]
        )

        model.compile(optimizer=keras.optimizers.Adam(lr=LR), loss='mse')

        return model

    def save(self, actor_path:str, critic_path:str):
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)

    def load(self, actor_path:str, critic_path:str):
        self.actor.load_weights(actor_path)
        self.critic.load_weights(critic_path)

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
            npt.NDArray[np.float32], # historical_network_throughput
            npt.NDArray[np.float32], # historical_chunk_download_time
            npt.NDArray[np.float32], # available_video_bitrates
            np.float32, # buffer_level
            np.float32, # remaining_chunk_count
            np.float32  # last_chunk_bitrate
        ]],
    ):
        # Convert state batch into correct format
        historical_network_throughput = np.zeros((len(state_batch), self.network_history_len)) 
        historical_chunk_download_time = np.zeros((len(state_batch), self.network_history_len)) 
        available_video_bitrates = np.zeros((len(state_batch), self.available_video_bitrates_count))
        buffer_level = np.zeros((len(state_batch), 1))
        remaining_chunk_count  = np.zeros((len(state_batch), 1))
        last_chunk_bitrate  = np.zeros((len(state_batch), 1))

        # Create other PPO2 things (needed for training, but during inference we dont care)
        advantage = np.zeros((len(state_batch), 1))
        old_prediction = np.zeros((len(state_batch), self.available_video_bitrates_count))

        for (i, s) in enumerate(state_batch):
            historical_network_throughput[i],
            historical_chunk_download_time[i],
            available_video_bitrates[i],
            buffer_level[i],
            remaining_chunk_count[i],
            last_chunk_bitrate[i] = s

        p = self.actor.predict([
            advantage,
            old_prediction,
            historical_network_throughput,
            historical_chunk_download_time,
            available_video_bitrates,
            buffer_level,
            remaining_chunk_count,
            last_chunk_bitrate,
        ])
        return p

    def train(
        self,
        state_batch: list[tuple[
            npt.NDArray[np.float32], # historical_network_throughput
            npt.NDArray[np.float32], # historical_chunk_download_time
            npt.NDArray[np.float32], # available_video_bitrates
            np.float32, # buffer_level
            np.float32, # remaining_chunk_count
            np.float32  # last_chunk_bitrate
        ]],
        old_prediction_batch: list[float],
        advantage_batch:list[float],
    ):
        assert len(state_batch) == len(advantage_batch) 
        assert len(state_batch) == len(old_prediction_batch)

        # Convert state batch into correct format
        historical_network_throughput = np.zeros((len(state_batch), self.network_history_len)) 
        historical_chunk_download_time = np.zeros((len(state_batch), self.network_history_len)) 
        available_video_bitrates = np.zeros((len(state_batch), self.available_video_bitrates_count))
        buffer_level = np.zeros((len(state_batch), 1))
        remaining_chunk_count  = np.zeros((len(state_batch), 1))
        last_chunk_bitrate  = np.zeros((len(state_batch), 1))

        # Create other PPO2 things (needed for training, but during inference we dont care)
        advantage = np.reshape(advantage_batch, (len(advantage_batch), 1))
        old_prediction = np.reshape(old_prediction_batch, (len(advantage_batch), 1))

        for (i, s) in enumerate(state_batch):
            historical_network_throughput[i],
            historical_chunk_download_time[i],
            available_video_bitrates[i],
            buffer_level[i],
            remaining_chunk_count[i],
            last_chunk_bitrate[i] = s

        # Train Actor
        self.actor.fit([
            advantage,
            old_prediction,
            historical_network_throughput,
            historical_chunk_download_time,
            available_video_bitrates,
            buffer_level,
            remaining_chunk_count,
            last_chunk_bitrate,
        ])

        # Train Critic
        self.critic.fit(
            [
                historical_network_throughput,
                historical_chunk_download_time,
                available_video_bitrates,
                buffer_level,
                remaining_chunk_count,
                last_chunk_bitrate,
            ],
            advantage
        )

    # value function
    def compute_v(
        self,
        state_batch: list[tuple[
            npt.NDArray[np.float32], # historical_network_throughput
            npt.NDArray[np.float32], # historical_chunk_download_time
            npt.NDArray[np.float32], # available_video_bitrates
            np.float32, # buffer_level
            np.float32, # remaining_chunk_count
            np.float32  # last_chunk_bitrate
        ]],
        reward_batch: list[float],
        terminal: bool
    ) -> list[float]:
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

