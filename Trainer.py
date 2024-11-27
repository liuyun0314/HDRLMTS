import time
import torch
import numpy as np
import simpy
import copy
import datetime
import argparse
import openpyxl
from normalization import Normalization, RewardScaling
from PPO.task_agent import Agent
from PPO.Controller_PPO import PPO
from numpy import random
from PPO.replaybuffer import ReplayBuffer, ReplayBuffer_LSTM
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dispatchingRules.dispatchingRules import *
from sklearn.preprocessing import OneHotEncoder
from Environment.hy_jobSop import JobShop

DRs = ['SPT_SPTM', 'SPT_LWT', 'LPT_SPTM', 'LPT_LWT', 'MOPNR_SPTM', 'MOPNR_LWT', 'MWKR_SPTM', 'MWKR_LWT']


def main(args, seed, model_path, control_agent_path, opt_control_agent_path, optimal_model_path):
    optimi = 0
    total_steps = 0  # Record the total steps during the training
    test_machine_fault_time = 50
    avg = np.average(args.processing_time_range) - 0.5
    beta = avg / args.E_utliz
    np.random.seed(seed)
    arrival_interval = np.random.exponential(beta, args.num_new_jobs).round() + 1
    validation_arrival_interval = np.random.exponential(beta, args.num_new_jobs).round() + 1
    env = simpy.Environment()
    sim = JobShop(env, args, arrival_interval, seed)
    validation_sim = JobShop(env, args, validation_arrival_interval, 123)
    action_data = np.array([[i] for i in range(args.n_action)])

    encoder = OneHotEncoder()
    encoded_data = encoder.fit_transform(action_data).toarray()

    global_action_data = np.array([[i] for i in range(args.num_tasks)])
    encoder = OneHotEncoder()
    global_encoded_data = encoder.fit_transform(global_action_data).toarray()

    objectives = np.zeros((3, args.max_train_steps))
    rewards = np.zeros((2, args.max_train_steps))
    agent = Agent(0, args)
    control_agent = PPO(args)
    ##########################初始化#############################
    # agent.load_model(os.path.join(args.testModels_dir, 'twoDynamic/HPPO/agent_LSTM.pkl'))
    # control_agent.load_model(os.path.join(args.testModels_dir, 'twoDynamic/HPPO/controller_agent.pkl'))
    ############################################################

    global_replay_buffer = ReplayBuffer(args, args.global_state_dim)
    replay_buffers = ReplayBuffer_LSTM(args, args.RP_dim)
    # for id in range(args.num_tasks):
    #     replay_buffer = ReplayBuffer_LSTM(args, args.RP_dim)
    #     replay_buffers.append(replay_buffer)

    # print("------Check job shop environment------")
    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    global_state_norm = Normalization(shape=args.global_state_dim)
    reward_norm = Normalization(shape=1)  # Trick 3:reward normalization
    reward_scaling = RewardScaling(shape=1, gamma=args.gamma)  # Trick 4:reward scaling

    start_time = time.time()
    for epoch in range(args.max_train_steps):
        # print('***************epoch:', epoch)
        seed = random.randint(0, 1000000)
        # seed = 12
        avg = np.average(args.processing_time_range) - 0.5
        beta = avg / args.E_utliz
        np.random.seed(seed)
        machine_fault_time = np.random.randint(1, 2000)
        arrival_interval = np.random.exponential(beta, args.num_new_jobs).round() + 1
        sim.reset(seed, arrival_interval)
        sim.starTime = machine_fault_time

        validation_sim.reset(12,
                             validation_arrival_interval)
        validation_sim.starTime = test_machine_fault_time
        sim.decision_points.append(0)
        sim.decision_points.append(machine_fault_time)
        sim.decision_points.append(sim.arrival_interval[0])
        sim.decision_points = sorted(sim.decision_points)
        episode_steps = 0

        if args.use_reward_scaling:
            reward_scaling.reset()

        init_JSP_state(sim)
        while not sim.done:
            episode_steps += 1
            sim.dynamic_event()
            sim.machine_failure()
            sim.machine_repair()
            global_s, selected_task, global_action_logprob, idle_machines, last_obj = Resource_decision(control_agent, sim, global_encoded_data)
            jobs = sim.tasks_list[selected_task].jobsList
            if len(jobs) == 0 or len(idle_machines) == 0:
                sim.env.timeout(1)
                sim.env.run()
                continue
            reward_function, func = select_reward_function(sim, selected_task)
            r = eval(func)(sim.env.now, jobs, sim.machines)
            obj = three_objection(sim)
            state = sim.get_local_state(jobs, sim.machines, sim.env.now)
            inputs = copy.deepcopy(state)
            action, action_logprob = agent.select_action(inputs)
            agent.last_action = action
            DR = DRs[action]
            _ = sim.step(jobs, idle_machines, DR)
            reward = Reward(sim, obj, reward_function, r, jobs)
            global_reward = reward

            s_ = sim.get_local_state(jobs, sim.machines, sim.env.now)

            global_s_next = sim.get_global_features()
            global_action_oneHot = global_encoded_data[control_agent.last_action]
            global_s_next.append(global_action_oneHot)

            if args.use_state_norm:
                s_ = state_norm(s_)
                global_s_next = global_state_norm(global_s_next)

            if args.use_reward_norm:
                reward = reward_norm(reward)
                global_reward = reward_norm(global_reward)
            elif args.use_reward_scaling:
                reward = reward_scaling(reward)
                global_reward = reward_scaling(global_reward)

            if sim.done:
                dw = True
            else:
                dw = False

            replay_buffers.store(state, action, action_logprob, reward, s_, dw, sim.done)
            global_replay_buffer.store(global_s, selected_task, global_action_logprob, global_reward, global_s_next,
                                       dw, sim.done)
            total_steps += 1

            if replay_buffers.count == args.batch_size:
                agent.update(replay_buffers, total_steps)
                replay_buffers.clear()

            if global_replay_buffer.count == args.batch_size:
                control_agent.update(global_replay_buffer, total_steps)
                global_replay_buffer.clear()

            sim.env.timeout(1)
            sim.env.run()

        # if total_steps == 0:
        #     control_agent.save_model(opt_control_agent_path, optimi)
        #     agent.save_model(optimal_model_path, optimi, 0)
        #     optimi += 1
        if total_steps % args.save_freq == 0:
            control_agent.save_model(control_agent_path, total_steps)
            agent.save_model(model_path, total_steps, 0)
    end_time = time.time()
    exection_time = end_time - start_time
    avg_time = exection_time / args.max_train_steps
    control_agent.save_model(control_agent_path, total_steps)
    agent.save_model(model_path, total_steps, 0)
    return avg_time


def select_reward_function(sim, selected_task):
    reward_function = ''
    func = ''
    if sim.tasks_list[selected_task].objective == 'WTmean':
        reward_function = 'reward_WT_mean'
        func = 'estimated_WT_mean'
    if sim.tasks_list[selected_task].objective == 'WFmean':
        reward_function = 'reward_WT_max'
        func = 'estimated_WF_mean'
    if sim.tasks_list[selected_task].objective == 'WTmax':
        reward_function = 'reward_WF_mean'
        func = 'estimated_WT_max'
    return reward_function, func


def Resource_decision(agent, sim, encoded_data):
    all_jobs = sim.jobs
    all_machines = sim.machines
    # step 1: find the idle machines
    idle_machines = []
    for m in all_machines:
        if m.currentTime <= sim.env.now:
            idle_machines.append(m)
    state = sim.get_global_features()
    last_obj = copy.deepcopy(state)
    one_hot = encoded_data[agent.last_action]
    state.append(one_hot)
    inputs = copy.deepcopy(state)
    action, action_logprob = agent.select_action(inputs)
    agent.last_action = action
    return state, action, action_logprob, idle_machines, last_obj


# ref {A Reinforcement Learning Approach for Flexible Job Shop Scheduling Problem With Crane Transportation and Setup Times}
def gaussian_weighted_moving_average(data, window_size=100, sigma=1):
    weights = np.exp(-0.5 * (np.arange(window_size) - (window_size - 1) / 2) ** 2 / sigma ** 2)
    weights /= np.sum(weights)
    smoothed = np.convolve(data, weights, mode='valid')
    return np.concatenate(([data[0]], smoothed))


def init_JSP_state(sim):
    # 考虑到初始时m个machine都分配给一个task是不合理的，所以初始时随机分配每个machine给一个task
    PT = sim.step(sim.jobs, sim.machines, 'SPT_SPTM')
    unique_list = [x for i, x in enumerate(PT) if x not in PT[:i]]  # 去除相同的元素
    for i in unique_list:
        if i not in sim.decision_points:
            sim.decision_points.append(i)
    # sim.decision_points.extend(unique_list)
    sim.decision_points = sorted(sim.decision_points)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument("--max_train_steps", type=int, default=int(10),
                        help=" Maximum number of training steps")  # 2e5
    parser.add_argument("--evaluate_freq", type=float, default=5e3,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=5000, help="Save frequency")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--global_batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=48, help="Minibatch size")
    parser.add_argument("--state_dim", type=int, default=12, help="The dimension of local state")
    parser.add_argument("--task_state_dim", type=int, default=5, help="The dimension of task")
    parser.add_argument("--num_input", type=int, default=5, help="The types of input")
    parser.add_argument("--machine_state_dim", type=int, default=5, help="The dimension of machine")
    parser.add_argument("--action_state_dim", type=int, default=3, help="The dimension of action")
    parser.add_argument("--global_state_dim", type=int, default=24, help="The dimension of global state")
    parser.add_argument("--RP_dim", type=int, default=5, help="The dimension of replay buffer")
    parser.add_argument("--machine_embedding_dim", type=int, default=5, help="The embedding dimension of machine")
    parser.add_argument("--jobs_embedding_dim", type=int, default=5, help="The embedding dimension of jobs")
    parser.add_argument("--action_embedding_width", type=int, default=1,
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--num_hidden_layer", type=int, default=2, help="The number of hidden layer")
    parser.add_argument("--embedding_dim", type=int, default=20, help="The number of hidden layer")  # 50
    parser.add_argument("--hidden_width", type=int, default=10,
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--task_hidden_width", type=int, default=30,
                        help="The number of neurons in hidden layers of the neural network")
    # parser.add_argument("--actor_hidden_layers", type=int, default=5, help="The number of hidden layers in the actor policy network")
    # parser.add_argument("--critic_hidden_layers", type=int, default=3, help="The number of hidden layers in the critic policy network")
    parser.add_argument("--lr_a", type=float, default=1e-2, help="Learning rate of actor")  # 6e-3
    parser.add_argument("--lr_c", type=float, default=1e-1, help="Learning rate of critic")  # 1e-3
    parser.add_argument("--control_lr_a", type=float, default=1e-6, help="Learning rate of control_actor")  # 1e-4
    parser.add_argument("--control_lr_c", type=float, default=1e-5, help="Learning rate of control_critic")  # 1e-3
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")

    parser.add_argument("--num_tasks", type=int, default=3, help="The number of tasks")
    parser.add_argument("--num_total_jobs", type=int, default=20, help="The number of jobs in system")
    parser.add_argument("--num_new_jobs", type=int, default=6, help="The number of new jobs")  # 100
    parser.add_argument("--num_machines", type=int, default=10, help="The number of machines")  # 10
    parser.add_argument("--n_action", type=int, default=8, help="The number of dispatching rules")
    parser.add_argument("--E_utliz", type=float, default=0.95, help="The machine utilization ")
    parser.add_argument("--num_ops_range", type=tuple, default=(1, 3), help="The range of operations in a job")  # 10
    parser.add_argument("--num_cand_machines_range", type=tuple, default=(1, 10),
                        help="The range of candidate machines for each operation")
    parser.add_argument("--weights", type=list, default=[0.2, 0.6, 0.2], help="The weight of each job")
    parser.add_argument("--processing_time_range", type=tuple, default=(1, 99),
                        help="The processing time of an operation")
    parser.add_argument("--due_time_multiplier", type=float, default=1.5, help="The due time multiplier of a job")
    parser.add_argument("--num_warmup_jobs", type=int, default=1, help="The number of warmup jobs")  # 10
    parser.add_argument("--seed", type=int, default=12, help="seed")
    parser.add_argument("--C", type=int, default=10, help="the update frequencey of global policy")
    parser.add_argument("--testModels_dir", type=str, default='E:/Phd_work/myWork/Code/2/testModels/',
                        help="Save path of the model")

    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=True, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=False, help="Trick 10: tanh activation function")
    parser.add_argument("--use_attention_fusion", type=float, default=True,
                        help="attention fusion for controller agent")
    args = parser.parse_args()
    now = datetime.datetime.now()
    now_time = str(now.month) + '_' + str(now.day) + '_' + str(now.hour) + '_' + str(now.minute)
    model_path = 'E:/Phd_work/myWork/Code/2/trained_models/' + now_time
    model_path = 'E:/Phd_work/myWork/Code/2/trained_models/Task_agent/' + now_time
    control_agent_path = 'E:/Phd_work/myWork/Code/2/trained_models/Controller_agent/' + now_time
    optimal_model_path = 'E:/Phd_work/myWork/Code/2/trained_models/Task_agent/HPPO/opt/' + now_time
    opt_control_agent_path = 'E:/Phd_work/myWork/Code/2/trained_models/Controller_agent/HPPO/opt/' + now_time
    figure_file = 'E:/Phd_work/myWork/Code/2/figures/'

    seed = random.randint(0, 1000000)
    exection_time = main(args, seed, model_path, control_agent_path, opt_control_agent_path, optimal_model_path)
