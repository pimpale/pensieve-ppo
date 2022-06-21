from typing import TYPE_CHECKING
import math
import numpy as np
import numpy.typing as npt
import tensorflow as tf
from tensorflow import keras

from env import Observation

# Hyperparameters
VID_CONV_FILTERS = 128
HIST_CONV_FILTERS = 128
HIDDEN_LAYER_SIZE = 128

LR = 1e-5  # Lower lr stabilises training greatly
GAMMA = 0.99

# PPO2
ACTOR_PPO_LOSS_CLIPPING=0.2
PPO_TRAINING_EPO = 5
H_TARGET = 0.1

class PPOAgent:
    def __init__(self, network_history_len:int, available_video_bitrates_count:int):
        self.network_history_len = network_history_len
        self.available_video_bitrates_count = available_video_bitrates_count
        self.actor = self.__build_actor()
        self.critic = self.__build_critic()
        self._entropy_weight:float = np.log(available_video_bitrates_count)


    # Private
    def __build_actor(self):
        # network throughput measurements for the last k video chunks
        feature_historical_network_throughput = keras.layers.Input(shape=[self.network_history_len, 1])
        # chunk download times for the last k video chunks
        feature_historical_chunk_download_time = keras.layers.Input(shape=[self.network_history_len, 1])
        # a vector of m available sizes for the next video chunk
        feature_available_video_bitrates = keras.layers.Input(shape=[self.available_video_bitrates_count, 1])
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

        action_probs = keras.layers.Dense(self.available_video_bitrates_count, activation='softmax')(hidden_layer)

        # oldpolicy probs is the vector of predictions made by the old actor model
        oldpolicy_probs = keras.layers.Input(shape=self.available_video_bitrates_count)
        # is a 1-hot vector
        action_chosen = keras.layers.Input(shape=self.available_video_bitrates_count)

        # advantage captures how better an action is compared to the others at a given state
        # https://theaisummer.com/Actor_critics/
        advantage = keras.layers.Input(shape=(1,))

        # Needed for entropy regularization
        entropy_weight = keras.layers.Input(shape=(1,))


        model = keras.Model(
            inputs=[
                # ppo features
                advantage,
                oldpolicy_probs,
                action_chosen,
                entropy_weight,
                # real features
                feature_historical_network_throughput,
                feature_historical_chunk_download_time,
                feature_available_video_bitrates,
                feature_buffer_level,
                feature_remaining_chunk_count,
                feature_last_chunk_bitrate
            ],
            outputs=[action_probs]
        )


        def actor_ppo_loss(newpolicy_probs, oldpolicy_probs, action_chosen, advantage, entropy_weight):
            # Reward???? IDK what this function does
            def r(pi_new, pi_old, acts):
                return tf.reduce_sum(tf.multiply(pi_new, acts), axis=1, keepdims=True) / \
                        tf.reduce_sum(tf.multiply(pi_old, acts), axis=1, keepdims=True)

            # Unsure what this value represents??
            rv = r(newpolicy_probs, oldpolicy_probs, action_chosen)
            # PPO2 Loss Clipping
            ppo2loss = tf.minimum(rv, tf.clip_by_value(rv, 1-ACTOR_PPO_LOSS_CLIPPING, 1+ACTOR_PPO_LOSS_CLIPPING)) * advantage

            # Dual Loss (Unsure what the advantage of this is???)
            dual_loss = tf.where(tf.less(advantage, 0.), tf.maximum(ppo2loss, 3. * advantage), ppo2loss)

            # Calculate Entropy (how uncertain the prediction is)
            entropy = -tf.reduce_sum(tf.multiply(newpolicy_probs, tf.math.log(newpolicy_probs)), axis=1, keepdims=True)

            # actor loss
            return -tf.reduce_sum(dual_loss) - entropy_weight * entropy

        model.add_loss(actor_ppo_loss(
          newpolicy_probs=action_probs,
          oldpolicy_probs=oldpolicy_probs,
          action_chosen=action_chosen,
          advantage=advantage,
          entropy_weight=entropy_weight
        ))

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LR),
        )

        return model

    # Private
    # The critic attempts to learn the advantage
    def __build_critic(self):
        # network throughput measurements for the last k video chunks
        feature_historical_network_throughput = keras.layers.Input(shape=[self.network_history_len, 1])
        # chunk download times for the last k video chunks
        feature_historical_chunk_download_time = keras.layers.Input(shape=[self.network_history_len, 1])
        # a vector of m available sizes for the next video chunk
        feature_available_video_bitrates = keras.layers.Input(shape=[self.available_video_bitrates_count, 1])
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

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=LR), loss='mse')

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
        state_batch: list[Observation],
    ):

        batch_len = len(state_batch)

        # Convert state batch into correct format
        historical_network_throughput = np.zeros((batch_len, self.network_history_len)) 
        historical_chunk_download_time = np.zeros((batch_len, self.network_history_len)) 
        available_video_bitrates = np.zeros((batch_len, self.available_video_bitrates_count))
        buffer_level = np.zeros((batch_len, 1))
        remaining_chunk_count  = np.zeros((batch_len, 1))
        last_chunk_bitrate  = np.zeros((batch_len , 1))

        # Create other PPO2 things (needed for training, but during inference we dont care)
        advantage = np.zeros((batch_len, 1))
        oldpolicy_probs = np.zeros((batch_len, self.available_video_bitrates_count))
        chosen_action= np.zeros((batch_len, self.available_video_bitrates_count))
        entropy_weight = np.zeros((batch_len, 1))

        for (i, (hnt, hcdt, avb, bl, rcc, lcb)) in enumerate(state_batch):
            historical_network_throughput[i] = hnt
            historical_chunk_download_time[i] = hcdt
            available_video_bitrates[i] = avb
            buffer_level[i] = bl
            remaining_chunk_count[i] = rcc
            last_chunk_bitrate[i] = lcb

        p = self.actor(
            [
                # Dummy
                advantage,
                oldpolicy_probs,
                chosen_action,
                entropy_weight,
                # Real
                historical_network_throughput,
                historical_chunk_download_time,
                available_video_bitrates,
                buffer_level,
                remaining_chunk_count,
                last_chunk_bitrate,
            ],
        )
        return p

    def train(
        self,
        state_batch: list[Observation],
        action_batch: list[npt.NDArray[np.float32]],
        advantage_batch:list[float],
        old_prediction_batch: list[npt.NDArray[np.float32]],
        base_step:int
    ):
        batch_len = len(state_batch)
        assert batch_len == len(action_batch)
        assert batch_len == len(advantage_batch) 
        assert batch_len == len(old_prediction_batch)

        # Convert state batch into correct format
        historical_network_throughput = np.zeros((len(state_batch), self.network_history_len)) 
        historical_chunk_download_time = np.zeros((len(state_batch), self.network_history_len)) 
        available_video_bitrates = np.zeros((len(state_batch), self.available_video_bitrates_count))
        buffer_level = np.zeros((len(state_batch), 1))
        remaining_chunk_count  = np.zeros((len(state_batch), 1))
        last_chunk_bitrate  = np.zeros((len(state_batch), 1))

        # Create other PPO2 things (needed for training, but during inference we dont care)
        advantage = np.reshape(advantage_batch, (batch_len, 1))
        oldpolicy_probs= np.reshape(old_prediction_batch, (batch_len, self.available_video_bitrates_count))
        chosen_action = np.reshape(action_batch, (batch_len, self.available_video_bitrates_count))

        entropy_weight = np.full((batch_len, 1), self._entropy_weight);

        for (i, (hnt, hcdt, avb, bl, rcc, lcb)) in enumerate(state_batch):
            historical_network_throughput[i] = hnt
            historical_chunk_download_time[i] = hcdt
            available_video_bitrates[i] = avb
            buffer_level[i] = bl
            remaining_chunk_count[i] = rcc
            last_chunk_bitrate[i] = lcb

        class PrintLoss(keras.callbacks.Callback):
            def __init__(self, base_step:int, name:str):
                self.base_step = base_step
                self.name = name
            def on_epoch_end(self, epoch, logs):
                tf.summary.scalar(self.name, logs['loss'], step=self.base_step+epoch)

        # Train Actor
        self.actor.fit(
            [
                # Required to compute loss
                advantage,
                oldpolicy_probs,
                chosen_action,
                entropy_weight,
                # Real values
                historical_network_throughput,
                historical_chunk_download_time,
                available_video_bitrates,
                buffer_level,
                remaining_chunk_count,
                last_chunk_bitrate,
            ],
            epochs=PPO_TRAINING_EPO,
            callbacks=[PrintLoss(base_step, 'loss_actor')]
        )

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
            advantage,
            epochs=PPO_TRAINING_EPO,
            callbacks=[PrintLoss(base_step, 'critic_loss')]
        )

        p_batch = np.clip(oldpolicy_probs, ACTOR_PPO_LOSS_CLIPPING, 1. - ACTOR_PPO_LOSS_CLIPPING)
        H = np.mean(np.sum(-np.log(p_batch) * p_batch, axis=1))
        g = H - H_TARGET
        self._entropy_weight -= LR * g * 0.1

        return PPO_TRAINING_EPO

    def get_entropy_weight(self) -> float:
        return self._entropy_weight

    # value function
    def compute_v(
        self,
        state_batch: list[Observation],
        reward_batch: list[float],
        terminal: bool
    ) -> list[float]:
        assert len(state_batch) == len(reward_batch)

        ba_size= len(state_batch)
        R_batch = np.zeros([len(reward_batch), 1])

        if terminal:
            R_batch[-1, 0] = 0  # terminal state
        else:    
            v_batch = self.critic(state_batch)
            R_batch[-1, 0] = v_batch[-1, 0]  # boot strap from last state
        # Use GAMMA to decay value 
        for t in reversed(range(ba_size - 1)):
            R_batch[t, 0] = reward_batch[t] + GAMMA * R_batch[t + 1, 0]

        return list(R_batch)

