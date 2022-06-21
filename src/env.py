# add queuing delay into halo
import os
import numpy as np
import numpy.typing as npt
import core as abrenv
import load_trace
from typing import TypedDict

# bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
NETWORK_HISTORY_LEN  = 8  # take how many frames in the past
TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 100
VIDEO_BIT_RATES:npt.NDArray[np.float32] = np.array([300., 750., 1200., 1850., 2850., 4300.])  # Kbps
AVAILABLE_VIDEO_BITRATES_COUNT = len(VIDEO_BIT_RATES)
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY_IDX = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
EPS = 1e-6

Observation = tuple[
    npt.NDArray[np.float32], # historical_network_throughput
    npt.NDArray[np.float32], # historical_chunk_download_time
    npt.NDArray[np.float32], # available_video_bitrates
    np.float32, # buffer_level
    np.float32, # remaining_chunk_count
    np.float32  # last_chunk_bitrate
]


class ABREnv():
    def __init__(
        self,
        random_seed=RANDOM_SEED,
    ):
        np.random.seed(random_seed)
        all_cooked_time, all_cooked_bw, _ = load_trace.load_trace()
        self.net_env = abrenv.Environment(all_cooked_time=all_cooked_time,
                                          all_cooked_bw=all_cooked_bw,
                                          random_seed=random_seed)
        self.reset()

    def seed(self, num):
        np.random.seed(num)

    def reset(self) -> Observation :
        # self.net_env.reset_ptr()

        # Reset variables
        self.buffer_size = 0.0
        self.last_bit_rate_idx = DEFAULT_QUALITY_IDX
        self.historical_network_throughput = np.zeros(NETWORK_HISTORY_LEN, dtype="float32")
        self.historical_chunk_download_time = np.zeros(NETWORK_HISTORY_LEN, dtype="float32")

        # Initial default
        bit_rate_idx = DEFAULT_QUALITY_IDX

        delay, sleep_time, self.buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain = self.net_env.get_video_chunk(bit_rate_idx )


        last_chunk_bitrate = VIDEO_BIT_RATES[bit_rate_idx] / np.float32(np.max(VIDEO_BIT_RATES))  # last quality
        buffer_level = self.buffer_size / BUFFER_NORM_FACTOR  # 10 sec
        self.historical_network_throughput[0] = np.float32(video_chunk_size) / np.float32(delay) / M_IN_K  # kilo byte / ms
        self.historical_chunk_download_time[1] = np.float32(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        available_video_bitrates = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
        remaining_chunk_count = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / np.float32(CHUNK_TIL_VIDEO_END_CAP)

        return (
          self.historical_network_throughput,
          self.historical_chunk_download_time,
          available_video_bitrates,
          buffer_level,
          remaining_chunk_count,
          last_chunk_bitrate
        )

    class StepInfo(TypedDict):
        bitrate: float
        rebuffer: bool

    def step(self, action:np.intp) -> tuple[Observation, float, bool, StepInfo]:
        bit_rate_idx = action
        # the action is from the last decision
        # this is to make the framework similar to the real
        delay, sleep_time, self.buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain = \
            self.net_env.get_video_chunk(bit_rate_idx)

        # reward is video quality - rebuffer penalty - smooth penalty
        reward = VIDEO_BIT_RATES[bit_rate_idx] / M_IN_K \
            - REBUF_PENALTY * rebuf \
            - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATES[bit_rate_idx] - VIDEO_BIT_RATES[self.last_bit_rate_idx]) / M_IN_K

        self.last_bit_rate_idx = bit_rate_idx

        self.historical_network_throughput = np.roll(self.historical_network_throughput, -1)
        self.historical_chunk_download_time = np.roll(self.historical_chunk_download_time, -1)

        # Calculate reqs
        last_chunk_bitrate = VIDEO_BIT_RATES[bit_rate_idx] / float(np.max(VIDEO_BIT_RATES))  # last quality
        buffer_level = self.buffer_size / BUFFER_NORM_FACTOR  # 10 sec
        self.historical_network_throughput[0] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
        self.historical_chunk_download_time[0] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        available_video_bitrates = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
        remaining_chunk_count = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

        state = (
          self.historical_network_throughput,
          self.historical_chunk_download_time,
          available_video_bitrates,
          buffer_level,
          remaining_chunk_count,
          last_chunk_bitrate
        )

        return state, reward, end_of_video, {'bitrate': VIDEO_BIT_RATES[bit_rate_idx], 'rebuffer': rebuf}
