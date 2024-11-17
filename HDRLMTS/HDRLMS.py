import os
import torch
import copy
import random
from PPO.rewards import *
from PPO.Controller_PPO import PPO
# from PPO.V1_Agent import Agent
from PPO.task_agent import Agent
from sklearn.preprocessing import OneHotEncoder
from normalization import Normalization, RewardScaling
from Environment.hy_jobSop import JobShop, Exponential_arrival, select_task

DRs = ['SPT_SPTM', 'SPT_LWT', 'LPT_SPTM', 'LPT_LWT', 'MOPNR_SPTM', 'MOPNR_LWT', 'MWKR_SPTM', 'MWKR_LWT']

######################### with machine failure###########################
def HiPPOS(sim, args):
    objectives = np.zeros(4)

    global_action_data = np.array([[i] for i in range(args.num_tasks)])
    encoder = OneHotEncoder()
    global_encoded_data = encoder.fit_transform(global_action_data).toarray()

    action_data = np.array([[i] for i in range(args.n_action)])
    # 创建OneHotEncoder对象
    encoder = OneHotEncoder()
    # 使用fit_transform()方法将分类变量转换为二进制向量
    encoded_data = encoder.fit_transform(action_data).toarray()
    control_agent = PPO(args)
    control_agent.load_model(os.path.join(args.testModels_dir, 'twoDynamic/HPPO/controller_agent.pkl'))
    agent = Agent(0, args)
    agent.load_model(os.path.join(args.testModels_dir, 'twoDynamic/HPPO/agent_LSTM.pkl'))

    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    global_state_norm = Normalization(shape=args.global_state_dim)
    reward_norm = Normalization(shape=1)  # Trick 3:reward normalization
    reward_scaling = RewardScaling(shape=1, gamma=args.gamma)  # Trick 4:reward scaling

    init_JSP_state(sim)
    while not sim.done:
        sim.dynamic_event()
        sim.machine_failure()
        sim.machine_repair()
        if sim.env.now == sim.decision_points[0]:
            sim.decision_points.remove(sim.decision_points[0])
            global_s, selected_task, idle_machines = Resource_decision(control_agent, sim, global_encoded_data)
            jobs = sim.tasks_list[selected_task].jobsList
            if len(jobs) == 0 or len(idle_machines) == 0:
                if len(sim.decision_points) == 0:
                    sim.env.timeout(1)
                else:
                    sim.env.timeout(sim.decision_points[0] - sim.env.now)
                sim.env.run()
                continue
            state = sim.get_local_state(jobs, sim.machines, sim.env.now)
            last_action = agent.last_action

            action, action_logprob = agent.select_action(state)
            agent.last_action = action
            DR = DRs[action]
            _ = sim.step(sim.jobs, idle_machines, DR)
            if len(sim.decision_points) == 0:
                sim.env.timeout(1)
            else:
                sim.env.timeout(sim.decision_points[0] - sim.env.now)
            sim.env.run()
            if sim.env.now not in sim.decision_points:
                sim.decision_points.append(sim.env.now)

        if len(sim.decision_points) == 0:
            sim.decision_points.append(sim.env.now)
        sim.decision_points = sorted(sim.decision_points)
    WTmean = WT_mean_func(sim.tasks_list[0].jobsList)
    objectives[0] = WTmean
    WTmax = WT_max_func(sim.tasks_list[1].jobsList)
    objectives[1] = WTmax
    WFmean = WF_mean_func(sim.tasks_list[2].jobsList)
    objectives[2] = WFmean
    machine_UR = machine_utilization(sim.env.now, sim.machines)
    objectives[3] = machine_UR
    return objectives

def Resource_decision(agent, sim, encoded_data):
    all_machines = sim.machines
    idle_machines = []
    for m in all_machines:
        if m.currentTime <= sim.env.now:
            idle_machines.append(m)

    state = sim.get_global_features()
    one_hot = encoded_data[agent.last_action]
    state.append(one_hot)
    # inputs = torch.Tensor(state)
    inputs = copy.deepcopy(state)
    # action, action_logprob = agent.select_action(inputs)
    r = random.random()
    if r < 0.99:
        action, _ = agent.select_action(inputs)
    else:
        action = random.randint(0, 2)
    return state, action, idle_machines

def Resource_decision_forTesting(agent, sim, encoded_data):
    all_machines = sim.machines
    idle_machines = []
    for m in all_machines:
        if m.currentTime <= sim.env.now:
            idle_machines.append(m)

    state = sim.get_global_features()
    one_hot = encoded_data[agent.last_action]
    state.append(one_hot)
    inputs = copy.deepcopy(state)
    r = random.random()
    if r < 0.9:
        action, _ = agent.select_action(inputs)
    else:
        action = random.randint(0, 2)
    return state, action, idle_machines

def init_JSP_state(sim):
    index = list(range(len(sim.machines)))
    random.shuffle(index)
    group_size = len(index) // sim.num_tasks
    group_1 = index[:group_size]
    group_2 = index[group_size:group_size * 2]
    group_3 = index[group_size * 2:]
    machines1 = [sim.machines[i] for i in group_1]
    machines2 = [sim.machines[i] for i in group_2]
    machines3 = [sim.machines[i] for i in group_3]
    _ = sim.step(sim.tasks_list[0].jobsList, machines1, 'SPT_SPTM')
    _ = sim.step(sim.tasks_list[1].jobsList, machines2, 'SPT_SPTM')
    _ = sim.step(sim.tasks_list[2].jobsList, machines3, 'SPT_SPTM')


