import numpy as np

######################design_1######################

def reward_WT_mean(r, jobs, machines, currentTime):

    r_next = estimated_WT_mean(currentTime, jobs, machines)
    if r_next < r:
        reward = 1
    else:
        if r_next < r * 1.05:
            reward = 0
        else:
            reward = -1
    return reward

def reward_WT_max(r, jobs, machines, currentTime):

    r_next = estimated_WT_max(currentTime, jobs, machines)
    if r_next < r:
        reward = 1
    else:
        if r_next < r * 1.05:
            reward = 0
        else:
            reward = -1
    return reward

def reward_WF_mean(r, jobs, machines, currentTime):

    r_next = estimated_WF_mean(currentTime, jobs, machines)
    if r_next < r:
        reward = 1
    else:
        if r_next < r *1.01:
            reward = 0
        else:
            reward = -1
    return reward

def globalReward(r, sim):
    r_next = machine_utilizatioin_ratio(sim)
    # if r_next > 0.9:
    #     reward1 = 1
    # else:
    #     reward1 = -1

    if r_next > r:
        reward2 = 1
    else:
        if r_next > 0.9 * r:
            reward2 = 0
        else:
            reward2 = -1
    return reward2

def high_level_r(r, sim):
    r_next = machine_utilizatioin_ratio(sim)
    if r_next > 0.95:
        reward = r_next - 0.95
    else:
        reward = r_next - r
    return reward

def reward_global_U(r, currentTime, machines):

    r_next = machine_utilization(currentTime, machines)
    if r_next > 0.9:
        reward1 = 1
    else:
        reward1 = -1

    if r_next > r:
        reward2 = 1
    else:
        if r_next > 0.9 * r:
            reward2 = 0
        else:
            reward2 = -1
        # if r_next >= 0.97 * r:   # ref{Real-Time Scheduling for Dynamic Partial-No-Wait Multiobjective Flexible Job Shop by Deep Reinforcement Learning}
        #     reward = 0
        # else:
        #     reward = -1
    return reward1 + reward2

def reward_idel_time(r, currentTime, machines):

    r_next = machine_idle_time(currentTime, machines)
    # if r == r_next:
    #     reward = 0
    # else:
    if r_next < r:
        reward = 1
    else:
        if r_next == r:   
            reward = 0
        else:
            reward = -1
    return reward

def estimated_WT_mean(currentTime, jobs, machines):
    CKs = []
    ETWTs = []
    num_unfinished_jobs = 0
    for m in machines:
        CKs.append(m.currentTime)

    T_cure = np.mean(CKs) 
    for j in jobs:
        if not j.completed:
            num_unfinished_jobs += 1
            all_t_avg = []
            C_last = 0
            for index in range(len(j.operation_list)):
                if j.operation_list[index].completed:
                    C_last = j.operation_list[index].endTime
                # if j.operation_list[index].completed and j.operation_list[index].endTime <= currentTime:
                #     C_last = j.operation_list[index].endTime
                if not j.operation_list[index].completed:
                    cMachines = j.operation_list[index].cMachines
                    t_avg = np.mean(list(cMachines.values()))
                    all_t_avg.append(t_avg)
            sum_t_avg = sum(all_t_avg)
            ETWT = j.weight * max(0, max(currentTime, C_last) + sum_t_avg - j.DT)
        else:
            ETWT = j.weight * max(0, j.endTime - j.DT)
        ETWTs.append(ETWT)
    return np.mean(ETWTs)     # sum(ETWTs) / len(ETWTs)

def estimated_WT_max(currentTime, jobs, machines):
    CKs = []
    ETWTs = []
    num_unfinished_jobs = 0
    for m in machines:
        CKs.append(m.currentTime)

    T_cure = np.mean(CKs)
    for j in jobs:
        if not j.completed:
            num_unfinished_jobs += 1
            all_t_avg = []
            C_last = 0
            for index in range(len(j.operation_list)):
                if j.operation_list[index].completed:
                    C_last = j.operation_list[index].endTime
                # if j.operation_list[index].completed and j.operation_list[index].endTime <= currentTime:
                #     C_last = j.operation_list[index].endTime
                if not j.operation_list[index].completed:
                    cMachines = j.operation_list[index].cMachines
                    t_avg = np.mean(list(cMachines.values()))
                    all_t_avg.append(t_avg)
            sum_t_avg = sum(all_t_avg)
            ETWT = j.weight * max(0, max(currentTime, C_last) + sum_t_avg - j.DT)
        else:
            ETWT = j.weight * max(0, j.endTime - j.DT)
        ETWTs.append(ETWT)
    return np.max(ETWTs)
    
def estimated_WF_mean(currentTime, jobs, machines):
    CKs = []
    ETWFs = []
    num_unfinished_jobs = 0
    for m in machines:
        CKs.append(m.currentTime)
    T_cure = np.mean(CKs)
    for j in jobs:
        if not j.completed:
            num_unfinished_jobs += 1
            all_t_avg = []
            C_last = 0
            pre_op_dedTime = 0
            for index in range(len(j.operation_list)):
                if j.operation_list[index].completed:
                    C_last = j.operation_list[index].endTime
                # if j.operation_list[index].completed and j.operation_list[index+1].completed:
                #     if j.operation_list[index].endTime <= currentTime and j.operation_list[index+1].endTime > currentTime:
                #         C_last = j.operation_list[index].endTime
                if not j.operation_list[index].completed:
                    cMachines = j.operation_list[index].cMachines
                    t_avg = np.mean(list(cMachines.values()))
                    all_t_avg.append(t_avg)
            sum_t_avg = sum(all_t_avg)
            ETWF = j.weight * (max(currentTime, C_last) + sum_t_avg - j.RT)
        else:
            ETWF = j.weight * (j.endTime - j.RT)
        ETWFs.append(ETWF)
    return np.mean(ETWFs)
    
def re(sim, last_Cmax):
    Cmax = makespan(sim.machines)
    gap = Cmax - last_Cmax   # gap不可能为负，因为生产时间只会越来越长；当gap为0时说明在该决策点处选择的任务没有被进一步处理
    if gap == 0:
        reward = 0
    else:
        reward = 1 / (Cmax - last_Cmax)
    return reward

def makespan(machines):
    all_Cmax = []
    for m in machines:
        all_Cmax.append(m.currentTime)
        # Cmax_m = 0
        # for o in m.assignedOpera:
        #     Cmax_m += o.duration
        # all_Cmax.append(Cmax_m)
    Cmax = max(all_Cmax)
    return Cmax

def rewardGlobal(obj, r_MR, sim):
    r_WT_mean = obj[0]
    r_WT_max = obj[1]
    r_WF_mean = obj[2]

    objectives = three_objection(sim)
    r_WT_mean_next = objectives[0]
    r_WT_max_next = objectives[1]
    r_WF_mean_next = objectives[2]
    r_MR_next = machine_utilizatioin_ratio(sim)
    if r_WT_mean_next < r_WT_mean and r_WT_max_next < r_WT_max and r_WF_mean_next < r_WF_mean:
        reward = 3
    else:
        if (r_WT_mean_next < r_WT_mean and r_WT_max_next < r_WT_max) or (r_WT_mean_next < r_WT_mean and r_WF_mean_next < r_WF_mean) or (r_WT_max_next < r_WT_max and r_WF_mean_next < r_WF_mean):
            reward = 2
        else:
            if r_WT_mean_next < r_WT_mean or r_WT_max_next < r_WT_max or r_WF_mean_next < r_WF_mean:
                reward = 1
            else:
                reward = -3
    return reward

def reward_global(obj, r_MR, sim):
    r_WT_mean = obj[0]
    r_WT_max = obj[1]
    r_WF_mean = obj[2]

    objectives = three_objection(sim)
    r_WT_mean_next = objectives[0]
    r_WT_max_next = objectives[1]
    r_WF_mean_next = objectives[2]
    if r_WT_mean_next < r_WT_mean and r_WT_max_next < r_WT_max and r_WF_mean_next < r_WF_mean:
        reward = 3
    else:
        if (r_WT_mean_next < r_WT_mean and r_WT_max_next < r_WT_max) or (r_WT_mean_next < r_WT_mean and r_WF_mean_next < r_WF_mean) or (r_WT_max_next < r_WT_max and r_WF_mean_next < r_WF_mean):
            reward = 2
        else:
            if r_WT_mean_next < r_WT_mean or r_WT_max_next < r_WT_max or r_WF_mean_next < r_WF_mean:
                reward = 1
            else:
                reward = -3
    reward = reward - (-3) / (3 - (-3))
    return reward

def highLevel_reward(sim, last_Cmax, obj):

    reward = 0
    Cmax = makespan(sim.machines)
    r_WT_mean = obj[0]
    r_WT_max = obj[1]
    r_WF_mean = obj[2]

    objectives = three_objection(sim)
    r_WT_mean_next = objectives[0]
    r_WT_max_next = objectives[1]
    r_WF_mean_next = objectives[2]

    if r_WT_mean_next < r_WT_mean and r_WT_max_next < r_WT_max and r_WF_mean_next < r_WF_mean:
        reward = 3
    else:
        if (r_WT_mean_next < r_WT_mean and r_WT_max_next < r_WT_max) or (
                r_WT_mean_next < r_WT_mean and r_WF_mean_next < r_WF_mean) or (
                r_WT_max_next < r_WT_max and r_WF_mean_next < r_WF_mean):
            reward = 2
        else:
            if r_WT_mean_next < r_WT_mean or r_WT_max_next < r_WT_max or r_WF_mean_next < r_WF_mean:
                reward = 1
            else:
                reward = -3
    reward = reward - (-3) / (3 - (-3))
    reward = reward + (Cmax - last_Cmax) / last_Cmax
    return reward

def reward_PPO_withoutMultitask(r_WT_mean, r_WT_max, r_WF_mean, sim):
    objectives = three_objection(sim)
    r_WT_mean_next = objectives[0]
    r_WT_max_next = objectives[1]
    r_WF_mean_next = objectives[2]

    if r_WT_mean_next < r_WT_mean or r_WT_max_next < r_WT_max or r_WF_mean_next < r_WF_mean:
        reward = 1
    else:
        reward = 0
    return reward

def reward_DDQN(UK_avg, Tard_e, Tard_a, last_UK_avg, last_Tard_e, last_Tard_a):
    if Tard_a < last_Tard_a:
        reward = 1
    else:
        if Tard_a > last_Tard_a:
            reward = -1
        else:
            if Tard_e < last_Tard_e:
                reward = 1
            else:
                if Tard_e > last_Tard_e:
                    reward = -1
                else:
                    if UK_avg > last_UK_avg:
                        reward = 1
                    else:
                        if UK_avg > last_UK_avg * 0.95:
                            reward = 0
                        else:
                            reward = -1
    return reward
######################design_2######################

def reward_WT_mean1(r, jobs, machines, currentTime):

    r_next = estimated_WT_mean(currentTime, jobs, machines)
    if r_next < r:
        return 1
    else:
        if r_next == r:
            return 0
        else:
            return 1 / (r_next - r)

def reward_WT_max1(r, jobs, machines, currentTime):

    r_next = estimated_WT_max(currentTime, jobs, machines)
    if r_next < r:
        return 1
    else:
        if r_next == r:
            return 0
        else:
            return 1 / (r_next - r)

def reward_WF_mean1(r, jobs, machines, currentTime):

    r_next = estimated_WF_mean(currentTime, jobs, machines)
    if r_next < r:
        return 1
    else:
        if r_next == r:
            return 0
        else:
            return 1 / (r_next - r)
            
def three_objection(sim):
    EWT_mean = estimated_WT_mean(sim.env.now, sim.tasks_list[0].jobsList, sim.machines)
    EWT_max = estimated_WT_max(sim.env.now, sim.tasks_list[1].jobsList, sim.machines)
    EWF_mean = estimated_WF_mean(sim.env.now, sim.tasks_list[2].jobsList, sim.machines)
    return [EWT_mean, EWT_max, EWF_mean]
# ref{Real-Time Scheduling for Dynamic Partial-No-Wait Multiobjective Flexible Job Shop by Deep Reinforcement Learning}
# Three tasks are from {Task Relatedness Based Multitask Genetic Programming for Dynamic Flexible Job Shop Scheduling}

def machine_idle_time(currentTime, machines):
    idle_times = []
    for m in machines:
        pt = 0
        operations = m.assignedOpera
        for o in operations:
            if o.completed:
                if o.endTime <= currentTime:
                    pt += o.duration
                else:
                    pt += currentTime - o.startTime
        idle_time = currentTime - pt
        idle_times.append(idle_time)
    return np.sum(idle_times)

def idle_time(sim):
    idelTime = []
    Cmax = []
    for ta in range(sim.num_tasks):
        Cmax.append(sim.completed_jobs[ta][-1].endTime)
    makespan = max(Cmax)

    for m in sim.machines:
        pt = 0
        assigned_opera = m.assignedOpera
        if len(assigned_opera):
            for o in assigned_opera:
                pt += o.duration
        idelTime.append(makespan - pt)
    return np.average(idelTime)


def WT_mean_func(jobs):
    WTs = []
    for j in jobs:
        WT = j.weight * max(0, j.endTime - j.DT)
        WTs.append(WT)
    return np.mean(WTs)

def WT_max_func(jobs):
    WTs = []
    for j in jobs:
        WT = j.weight * max(0, j.endTime - j.DT)
        WTs.append(WT)
    return max(WTs)

def WF_mean_func(jobs):
    WFs = []
    for j in jobs:
        WF = j.weight * max(0, j.endTime - j.RT)
        WFs.append(WF)
    return np.mean(WFs)

def num_assigned_jobs(machines):
    num_all_assigned = 0
    for m in machines:
        o = m.assignedOpera
        assign_num = len(o)
        num_all_assigned += assign_num
    return num_all_assigned
