from io import TextIOWrapper
import multiprocessing as mp
import numpy as np
import numpy.typing as npt
import logging
import tensorflow as tf
import os
import sys
from env import ABREnv, Observation, AVAILABLE_VIDEO_BITRATES_COUNT, NETWORK_HISTORY_LEN
import ppo as network

NUM_AGENTS = 16
TRAIN_SEQ_LEN = 1000  # take as a train batch
TRAIN_EPOCH = 500000
MODEL_SAVE_INTERVAL = 100
RANDOM_SEED = 42
SUMMARY_DIR = './ppo'
MODEL_DIR = './models'
TRAIN_TRACES = './train/'
TEST_LOG_FOLDER = './test_results/'
LOG_FILE = SUMMARY_DIR + '/log'

# create result directory
if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)

def testing(epoch:int, nn_actor_model_path:str, nn_critic_model_path:str, log_file:TextIOWrapper):
    # clean up the test results folder
    os.system('rm -r ' + TEST_LOG_FOLDER)
    #os.system('mkdir ' + TEST_LOG_FOLDER)
    if not os.path.exists(TEST_LOG_FOLDER):
        os.makedirs(TEST_LOG_FOLDER)
    # run test script
    os.system(f'python test.py {nn_actor_model_path} {nn_critic_model_path}')
    # append test performance to the log
    rewards_list:list[float] = []
    entropies_list:list[float] = []
    test_log_files = os.listdir(TEST_LOG_FOLDER)
    for test_log_file in test_log_files:
        reward, entropy = [], []
        with open(TEST_LOG_FOLDER + test_log_file, 'rb') as f:
            for line in f:
                parse = line.split()
                try:
                    entropy.append(float(parse[-2]))
                    reward.append(float(parse[-1]))
                except IndexError:
                    break
        rewards_list.append(np.mean(reward[1:]))
        entropies_list.append(np.mean(entropy[1:]))
    rewards = np.array(rewards_list)
    rewards_min = np.min(rewards)
    rewards_5per = np.percentile(rewards, 5)
    rewards_mean = np.mean(rewards)
    rewards_median = np.percentile(rewards, 50)
    rewards_95per = np.percentile(rewards, 95)
    rewards_max = np.max(rewards)
    log_file.write(str(epoch) + '\t' +
                   str(rewards_min) + '\t' +
                   str(rewards_5per) + '\t' +
                   str(rewards_mean) + '\t' +
                   str(rewards_median) + '\t' +
                   str(rewards_95per) + '\t' +
                   str(rewards_max) + '\n')
    log_file.flush()
    return rewards_mean, np.mean(entropies_list)
        
def central_agent(net_params_queues, exp_queues):
    assert len(net_params_queues) == NUM_AGENTS
    assert len(exp_queues) == NUM_AGENTS

    # TODO: restore neural net parameters
    actor = network.PPOAgent(NETWORK_HISTORY_LEN, AVAILABLE_VIDEO_BITRATES_COUNT)

    # Get Writer
    writer = tf.summary.create_file_writer(SUMMARY_DIR);

    step=0

    with open(LOG_FILE + '_test.txt', 'w') as test_log_file, writer.as_default():
        for epoch in range(TRAIN_EPOCH):
            # synchronize the network parameters of work agent
            actor_net_params = actor.get_network_params()
            for i in range(NUM_AGENTS):
                net_params_queues[i].put(actor_net_params)

            s_batch:list[Observation] = []
            a_batch:list[npt.NDArray[np.float32]] = []
            p_batch:list[npt.NDArray[np.float32]]  = []
            v_batch:list[float] = []
            for i in range(NUM_AGENTS):
                s_, a_, p_, g_ = exp_queues[i].get()
                s_batch += s_
                a_batch += a_
                p_batch += p_
                v_batch += g_

            # Train Actor
            steps_trained = actor.train(s_batch, a_batch, v_batch, p_batch, step)
            step += steps_trained

            if epoch % MODEL_SAVE_INTERVAL == 0:
                # Save the neural net parameters to disk.
                actor_path = f"{SUMMARY_DIR}/nn_model_ep_{epoch}_actor.ckpt"
                critic_path = f"{SUMMARY_DIR}/nn_model_ep_{epoch}_critic.ckpt"
                save_path = actor.save(actor_path, critic_path)

                # Write to Log File
                avg_reward, avg_entropy = testing(epoch, actor_path, critic_path, test_log_file)
                tf.summary.scalar('avg_reward', avg_reward, step=step)
                tf.summary.scalar('avg_entropy', avg_entropy, step=step)


def agent(agent_id, net_params_queue, exp_queue):
    env = ABREnv(agent_id)
    actor = network.PPOAgent(NETWORK_HISTORY_LEN, AVAILABLE_VIDEO_BITRATES_COUNT)

    # initial synchronization of the network parameters from the coordinator
    actor_net_params = net_params_queue.get()
    actor.set_network_params(actor_net_params)

    for epoch in range(TRAIN_EPOCH):
        obs = env.reset()
        s_batch:list[Observation] = []
        a_batch:list[npt.NDArray[np.float32]] = []
        p_batch:list[npt.NDArray[np.float32]]  = []
        r_batch:list[float] = []
        done = True
        for step in range(TRAIN_SEQ_LEN):
            s_batch.append(obs)

            action_prob = actor.predict([obs])[0]

            # gumbel noise
            noise = np.random.gumbel(size=len(action_prob))
            chosen_vid_bitrate_idx = np.argmax(np.log(action_prob) + noise)

            obs, reward, done, info = env.step(chosen_vid_bitrate_idx)

            action_vec = np.zeros(AVAILABLE_VIDEO_BITRATES_COUNT, dtype="float32")
            action_vec[chosen_vid_bitrate_idx] = 1
            a_batch.append(action_vec)
            r_batch.append(reward)
            p_batch.append(action_prob)
            if done:
                break
        v_batch = actor.compute_v(s_batch, r_batch, done)
        exp_queue.put([s_batch, a_batch, p_batch, v_batch])

        actor_net_params = net_params_queue.get()
        actor.set_network_params(actor_net_params)

def main():
    np.random.seed(RANDOM_SEED)

    # inter-process communication queues
    net_params_queues = []
    exp_queues = []
    for i in range(NUM_AGENTS):
        net_params_queues.append(mp.Queue(1))
        exp_queues.append(mp.Queue(1))

    # create a coordinator and multiple agent processes
    # (note: threading is not desirable due to python GIL)
    coordinator = mp.Process(target=central_agent,
                             args=(net_params_queues, exp_queues))
    coordinator.start()

    agents = []
    for i in range(NUM_AGENTS):
        agents.append(mp.Process(target=agent,
                                 args=(i,
                                       net_params_queues[i],
                                       exp_queues[i])))
    for i in range(NUM_AGENTS):
        agents[i].start()

    # wait unit training is done
    coordinator.join()


if __name__ == '__main__':
    main()
