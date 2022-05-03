import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

# Hyperparameters
VID_CONV_FILTERS = 128
HIST_CONV_FILTERS = 128
HIDDEN_LAYER_SIZE = 128

LR = 1e-4  # Lower lr stabilises training greatly
ACTION_EPS = 1e-4
GAMMA = 0.99

def exponential_average(old:float, new:float, b1:float) -> float:
    return old * b1 + (1-b1) * new

def proximal_policy_optimization_loss(advantage:float, old_prediction):
    def loss(y_true:float, y_pred:float)->float:
        prob = K.sum(y_true * y_pred, axis=-1)
        old_prob = K.sum(y_true * old_prediction, axis=-1)
        r = prob/(old_prob + 1e-10)
        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage) + ENTROPY_LOSS * -(prob * K.log(prob + 1e-10)))
    return loss


def proximal_policy_optimization_loss_continuous(advantage, old_prediction):
    def loss(y_true, y_pred):
        var = K.square(NOISE)
        denom = K.sqrt(2 * math.pi * var)
        prob_num = K.exp(- K.square(y_true - y_pred) / (2 * var))
        old_prob_num = K.exp(- K.square(y_true - old_prediction) / (2 * var))

        prob = prob_num/denom
        old_prob = old_prob_num/denom
        r = prob/(old_prob + 1e-10)

        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage))
    return loss



class Agent:
    def __init__(self, network_history_len:int, available_video_sizes_count:int):
        self.network_history_len = network_history_len
        self.available_video_sizes_count = available_video_sizes_count
        self.actor = self.build_actor()
        self.critic = self.build_critic()

    def build_actor():
        # network throughput measurements for the last k video chunks
        feature_historical_network_throughput = keras.layers.Input(shape=(self.network_hist_feature_len,1))
        # chunk download times for the last k video chunks
        feature_historical_chunk_download_time = keras.layers.Input(shape=(self.network_hist_feature_len,1))
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

        model = Model(
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

        return model

    def build_critic():
        # network throughput measurements for the last k video chunks
        feature_historical_network_throughput = keras.layers.Input(shape=(self.network_hist_feature_len,1))
        # chunk download times for the last k video chunks
        feature_historical_chunk_download_time = keras.layers.Input(shape=(self.network_hist_feature_len,1))
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

        model = Model(
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

        model.compile(

        return model
