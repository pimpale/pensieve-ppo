import os
import sys
import numpy as np
import numpy.typing as npt
import tensorflow as tf
import load_trace
import ppo as network
import fixed_env as env
from env import Observation


NETWORK_HISTORY_LEN = 8
VIDEO_BIT_RATES = [300,750,1200,1850,2850,4300]  # Kbps
AVAILABLE_VIDEO_BITRATES_COUNT = len(VIDEO_BIT_RATES)

BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
LOG_FILE = './test_results/log_sim_ppo'
TEST_TRACES = './test/'
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward
    
def main():
    np.random.seed(RANDOM_SEED)

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TEST_TRACES)

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw)

    log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
    log_file = open(log_path, 'w')

    actor = network.PPOAgent(NETWORK_HISTORY_LEN, AVAILABLE_VIDEO_BITRATES_COUNT)

    # restore neural net parameters
    actor_model_path = sys.argv[1]
    critic_model_path = sys.argv[2]
    actor.load(actor_model_path, critic_model_path)
    print("Testing model restored.")

    time_stamp = 0

    last_bit_rate_idx:int= DEFAULT_QUALITY
    bit_rate_idx:int = DEFAULT_QUALITY

    action_vec = np.zeros(AVAILABLE_VIDEO_BITRATES_COUNT, dtype="float32")
    action_vec[bit_rate_idx] = 1

    entropy_record = []
    entropy_ = 0.5
    video_count = 0

    historical_network_throughput = np.zeros(NETWORK_HISTORY_LEN, dtype="float32")
    historical_chunk_download_time = np.zeros(NETWORK_HISTORY_LEN, dtype="float32")

    while True:  # serve video forever
        # the action is from the last decision
        # this is to make the framework similar to the real
        delay, sleep_time, buffer_size, rebuf, \
        video_chunk_size, next_video_chunk_sizes, \
        end_of_video, video_chunk_remain = \
            net_env.get_video_chunk(bit_rate_idx)

        time_stamp += delay  # in ms
        time_stamp += sleep_time  # in ms

        # reward is video quality - rebuffer penalty - smoothness
        reward = VIDEO_BIT_RATES[bit_rate_idx] / M_IN_K \
                 - REBUF_PENALTY * rebuf \
                 - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATES[bit_rate_idx] -
                                           VIDEO_BIT_RATES[last_bit_rate_idx]) / M_IN_K

        last_bit_rate_idx = bit_rate_idx

        # log time_stamp, bit_rate, buffer_size, reward
        log_file.write(str(time_stamp / M_IN_K) + '\t' +
                       str(VIDEO_BIT_RATES[bit_rate_idx]) + '\t' +
                       str(buffer_size) + '\t' +
                       str(rebuf) + '\t' +
                       str(video_chunk_size) + '\t' +
                       str(delay) + '\t' +
                       str(entropy_) + '\t' + 
                       str(reward) + '\n')
        log_file.flush()

        # dequeue history record
        historical_network_throughput = np.roll(historical_network_throughput, -1)
        historical_chunk_download_time = np.roll(historical_chunk_download_time, -1)

        # Calculate reqs
        last_chunk_bitrate = VIDEO_BIT_RATES[bit_rate_idx] / np.float32(np.max(VIDEO_BIT_RATES))  # last quality
        buffer_level = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
        historical_network_throughput[0] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
        historical_chunk_download_time[0] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        available_video_bitrates = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
        remaining_chunk_count = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

        # Predict using our observation
        action_prob = actor.predict([(
          historical_network_throughput,
          historical_chunk_download_time,
          available_video_bitrates,
          buffer_level,
          remaining_chunk_count,
          last_chunk_bitrate
        )])[0]

        noise = np.random.gumbel(size=len(action_prob))
        bit_rate = np.argmax(np.log(action_prob) + noise)
        
        entropy_ = -np.dot(action_prob, np.log(action_prob))
        entropy_record.append(entropy_)

        if end_of_video:
            log_file.write('\n')
            log_file.close()

            last_bit_rate_idx = DEFAULT_QUALITY
            bit_rate_idx = DEFAULT_QUALITY

            action_vec = np.zeros(AVAILABLE_VIDEO_BITRATES_COUNT, dtype="float32")
            action_vec[bit_rate_idx] = 1

            entropy_record = []
            entropy_ = 0.5

            historical_network_throughput = np.zeros(NETWORK_HISTORY_LEN, dtype="float32")
            historical_chunk_download_time = np.zeros(NETWORK_HISTORY_LEN, dtype="float32")

            video_count += 1
            print(video_count)

            if video_count >= len(all_file_names):
                break

            log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
            log_file = open(log_path, 'w')


if __name__ == '__main__':
    main()
