import copy
import math
import random

import torch
from dispatchingRules.dispatchingRules import *
from PPO.rewards import *
from Environment.machine import Machine
from Environment.Task import Task
from Environment.Job import Job
from Environment.Operation import Operation

DRs = ['SPT_SPTM', 'SPT_LWT', 'LPT_SPTM', 'LPT_LWT', 'MOPNR_SPTM', 'MOPNR_LWT', 'MWKR_SPTM', 'MWKR_LWT']

# Define job shop simulation
class JobShop:
    def __init__(self, env, args, arrival_interval, seed):
        # job shop
        self.env = env
        self.random_state = random.Random(seed)
        # if seed is not None:
        #     self.seed = seed
        self.num_tasks = args.num_tasks
        self.num_jobs = args.num_new_jobs + self.num_tasks * args.num_warmup_jobs   # the number of jobs
        self.num_warm_up = args.num_warmup_jobs  # the number of initial jobs in a task at beginning
        self.num_new_job = args.num_new_jobs
        self.dynamic_contor = 0
        self.num_machines = args.num_machines    # the number of machines
        self.num_ops_range = args.num_ops_range
        self.num_cand_machines_range = args.num_cand_machines_range
        self.weights = args.weights    # the number of operations of a job is randomly generated from the range
        self.processing_time_range = args.processing_time_range    # the processing time of rach operation is assigned by a range
        self.due_time_multiplier = args.due_time_multiplier
        # self.lmbda = lmbda
        # self.E_ave = 25
        self.arrival_interval = arrival_interval
        self.next_new_job_AT = self.arrival_interval[0]
        self.num_warmup_jobs = args.num_warmup_jobs
        self.index_job = 0      # the index of the jobs that arrives in the job shop system
        self.in_system_job_num = 0
        self.in_system_job_dic = {}        # {'arrival_time': self.in_system_job_num}
        self.tasks_list = []
        self.jobs = []
        self.machines = []
        self.C = args.C
        self.starTime = -1
        self.finishTime = -1
        self.failure_machine = None
        self.decision_points = []
        self.done = False
        self.num_finished = 0
        self.span = 0
        self.Task_Initializationr()
        self.Machine_Initializationr()
        self.completed_jobs = {}
        for i in range(self.num_tasks):
            self.completed_jobs[i] = []


    def reset(self, seed, arrival_interval):
        '''
        # reset the job shop environment
        :return:
        '''
        self.done = False
        self.env = simpy.Environment()
        self.random_state = random.Random(seed)
        self.decision_points = []
        self.index_job = 0  # the index of the jobs that arrives in the job shop system
        self.dynamic_contor = 0
        self.in_system_job_num = 0
        self.in_system_job_dic = {}  # {'arrival_time': self.in_system_job_num}
        self.tasks_list = []
        self.jobs = []
        self.machines = []
        self.span = 0
        self.starTime = -1
        self.finishTime = -1
        self.failure_machine = None
        self.num_finished = 0
        self.arrival_interval = arrival_interval
        self.next_new_job_AT = self.arrival_interval[0]
        for i in range(self.num_tasks):
            self.completed_jobs[i] = []
        self.Task_Initializationr()
        self.Machine_Initializationr()
        self.completed_jobs = {}
        for i in range(self.num_tasks):
            self.completed_jobs[i] = []


    def Task_Initializationr(self):
        for i in range(self.num_tasks):
            task = self.Task_Generator(i)
            self.tasks_list.append(task)
            for i in range(self.num_warmup_jobs):
                job = self.Jobs_Initializationr(task.idTask,0)
                # task.job_counter += 1
                self.jobs.append(job)
                task.jobsList.append(job)


    def Jobs_Initializationr(self, idTask, arrival_time):
        if idTask == 0 or idTask == 2:
            job_weight = self.random_state.choices([1, 2, 4], weights=self.weights)[0]
        else:
            job_weight = self.random_state.choices([1, 5, 10], weights=self.weights)[0]
        job = self.Job_Generator(idTask, job_weight, arrival_time, self.num_ops_range[1])
        return job

    def Job_Generator(self, task_id, weight, arrival_time, max_operation_num):
        # generate the operation list
        operations_list = []
        operations_num = self.random_state.randint(1, max_operation_num)
        for i in range(operations_num):
            operation = self.Operation_Generator(i, self.tasks_list[task_id].job_counter, task_id, arrival_time)
            operations_list.append(operation)
        job = Job(task_id, self.tasks_list[task_id].job_counter, weight, arrival_time, operations_list)
        self.tasks_list[task_id].job_counter += 1
        return job


    def Machine_Initializationr(self):
        current_time = 0
        for m in range(self.num_machines):
            machine = Machine(m, current_time)
            self.machines.append(machine)

    def Task_Generator(self, id):
        # tasks_list = []
        objectives = ['WTmean', 'WFmean', 'WTmax']
        objective = self.random_state.choices(objectives)
        task = Task(id, objectives[id])
        return task


    def Operation_Generator(self, id_operation, id_job, taskID, arrival_time):

        candidate_machines = {}
        # generate the information of candidate machine
        candidate_machine_num = self.random_state.randint(1, self.num_machines)
        sample_index = self.random_state.sample(range(self.num_machines), candidate_machine_num)
        # process_time = self.random_state.randint(self.processing_time_range[0], self.processing_time_range[1], candidate_machine_num)
        process_time = self.random_state.sample(range(self.processing_time_range[0], self.processing_time_range[1]),
                                                candidate_machine_num)
        count = 0
        for m in sample_index:
            machine_name = 'M' + str(m)
            candidate_machines[machine_name] = process_time[count]
            count += 1
        operation = Operation(id_operation, id_job, taskID, candidate_machines, arrival_time)
        return operation

    def getState_DDQN(self):
        states = []
        idel_machines = []
        jobs_list = self.jobs
        machines_list = self.machines
        currentTime = self.env.now
        n_jobs = len(jobs_list)
        num_machines = len(machines_list)
        WKs = []
        UKs = []
        for m in machines_list:
            if m.currentTime <= currentTime:
                idel_machines.append(m)
            wk = 0
            uk = 0
            operation_list = m.assignedOpera
            if len(operation_list) != 0:
                for op in operation_list:
                    wk += op.duration
                uk = wk / m.currentTime
            WKs.append(wk)
            UKs.append(uk)
        idel_rate = len(idel_machines) / num_machines

        # Average normalized machine workload
        max_WK = max(WKs)
        WKs = [wk / max_WK for wk in WKs]
        WK_avg = sum(WKs) / num_machines

        wk_sum = 0
        for w in WKs:
            wk_sum += pow(w - WK_avg, 2)
        WK_std = math.sqrt(wk_sum / num_machines)

        # Standard deviation of machine utilization rate
        UK_avg = sum(UKs) / num_machines
        sum_U = 0
        for u in UKs:
            sum_U += pow(u - UK_avg, 2)
        # Standard deviation of normalized machine workload
        UK_std = math.sqrt(sum_U / num_machines)

        OP = np.zeros(len(jobs_list))
        CRJ = np.zeros(len(jobs_list))
        opera_num = []
        for job in jobs_list:
            counter = 0
            n = len(job.operation_list)
            opera_num.append(n)
            for opera in job.operation_list:
                if opera.completed:
                    OP[counter] += 1
                else:
                    break
            CRJ[counter] = OP[counter] / len(job.operation_list)
            counter += 1
        # Completion rate of operations
        if n_jobs == 0:
            CRO = 0
            CRJ_avg = 0
            CRJ_std = 0
        else:
            CRO = sum(OP) / sum(opera_num)
            # Average job completion rate
            CRJ_avg = sum(CRJ) / n_jobs
            # Standard deviation of job completion rate
            sum_CRJ = 0
            for i in range(n_jobs):
                sum_CRJ += pow(CRJ[i] - CRJ_avg, 2)
            CRJ_std = math.sqrt(sum_CRJ / n_jobs)

        # Estimated Tardiness Rate Tard_e and Actual Tardiness Rate Tard_a
        CKs = []
        ETO = []
        for m in self.machines:
            CKs.append(m.currentTime)
        T_cure = np.mean(CKs)
        min_CK = min(CKs)
        num_unfinished_jobs = 0
        sum_unOpera = []
        ATJ = []
        UC = []
        ATJ_num_inOpera = []
        UC_num_inOpera = []
        for j in jobs_list:
            C_last = 0
            if j.completed:
                C_last = j.endTime
                uncompleted_num = 0
            if not j.completed:
                UC.append(j)
                num_unfinished_jobs += 1
                all_t_avg = []
                sum_t_avg = 0
                uncompleted_num = 0
                for o in j.operation_list:
                    if o.completed:
                        C_last = o.endTime
                    if not o.completed:
                        uncompleted_num += 1
                        cMachines = o.cMachines
                        t_avg = np.mean(list(cMachines.values()))
                        sum_t_avg += t_avg
                        if max(T_cure, C_last) + sum_t_avg > j.DT:
                            ETO.append(o)
                sum_unOpera.append(uncompleted_num)
                if max(C_last, min_CK) > j.DT:
                    ATJ.append(j)
                    ATJ_num_inOpera.append(uncompleted_num)
            # UC_num_inOpera.append(num_unfinished_jobs)
        if sum(sum_unOpera) == 0:
            Tard_e = 0
        else:
            Tard_e = len(ETO) / sum(sum_unOpera)  # Estimated Tardiness Rate Tard_e
        if sum(sum_unOpera) == 0 or len(sum_unOpera) or len(ATJ_num_inOpera) == 0:
            Tard_a = 0
        else:
            Tard_a = sum(ATJ_num_inOpera) / sum(sum_unOpera)  # Actual Tardiness Rate Tard_a
        return [UK_avg, UK_std, CRO, CRJ_avg, CRJ_std, Tard_e, Tard_a]

    def get_observation(self, jobs_list):
        # total number of machines for the task
        states = []
        idel_machines = []
        n_jobs = len(jobs_list)

        OP = np.zeros(len(jobs_list))
        CRJ = np.zeros(len(jobs_list))
        opera_num = []
        for job in jobs_list:
            counter = 0
            n = len(job.operation_list)
            opera_num.append(n)
            for opera in job.operation_list:
                if opera.completed:
                    OP[counter] += 1
                else:
                    break
            CRJ[counter] = OP[counter] / len(job.operation_list)
            counter += 1
        # Completion rate of operations
        if n_jobs == 0:
            CRO = 0
            CRJ_avg = 0
            CRJ_std = 0
        else:
            CRO = sum(OP) / sum(opera_num)
            # Average job completion rate
            CRJ_avg = sum(CRJ) / n_jobs
            # Standard deviation of job completion rate
            sum_CRJ = 0
            for i in range(n_jobs):
                sum_CRJ += pow(CRJ[i] - CRJ_avg, 2)
            CRJ_std = math.sqrt(sum_CRJ / n_jobs)

        # Estimated Tardiness Rate Tard_e and Actual Tardiness Rate Tard_a
        CKs = []
        ETO = []
        for m in self.machines:
            CKs.append(m.currentTime)
        T_cure = np.mean(CKs)
        min_CK = min(CKs)
        num_unfinished_jobs = 0
        sum_unOpera = []
        ATJ = []
        UC = []
        ATJ_num_inOpera = []
        for j in jobs_list:
            C_last = 0
            if j.completed:
                C_last = j.endTime
                uncompleted_num = 0
            if not j.completed:
                UC.append(j)
                num_unfinished_jobs += 1
                sum_t_avg = 0
                uncompleted_num = 0
                for o in j.operation_list:
                    if o.completed:
                        C_last = o.endTime
                    if not o.completed:
                        uncompleted_num += 1
                        cMachines = o.cMachines
                        t_avg = np.mean(list(cMachines.values()))
                        sum_t_avg += t_avg
                        if max(T_cure, C_last) + sum_t_avg > j.DT:
                            ETO.append(o)
                sum_unOpera.append(uncompleted_num)
                if max(C_last, min_CK) > j.DT:
                    ATJ.append(j)
                    ATJ_num_inOpera.append(uncompleted_num)
            # UC_num_inOpera.append(num_unfinished_jobs)
        if sum(sum_unOpera) == 0:
            Tard_e = 0
        else:
            Tard_e = len(ETO) / sum(sum_unOpera)  # Estimated Tardiness Rate Tard_e
        if sum(sum_unOpera) == 0 or len(sum_unOpera) or len(ATJ_num_inOpera) == 0:
            Tard_a = 0
        else:
            Tard_a = sum(ATJ_num_inOpera) / sum(sum_unOpera)  # Actual Tardiness Rate Tard_a
        # jobs_Infor = [n_jobs, CRO, CRJ_avg, CRJ_std, Tard_e, Tard_a]
        jobs_Infor = [CRO, CRJ_avg, CRJ_std, Tard_e, Tard_a]
        return jobs_Infor

    def get_local_state(self, jobs_list, machines_list, currentTime):
        # total number of machines for the task
        states = []
        idel_machines = []
        n_jobs = len(jobs_list)
        num_machines = len(machines_list)
        WKs = []
        UKs = []
        for m in machines_list:
            if m.currentTime <= currentTime:
                idel_machines.append(m)
            wk = 0
            uk = 0
            operation_list = m.assignedOpera
            if len(operation_list) != 0:
                for op in operation_list:
                    wk += op.duration
                uk = wk / m.currentTime
            WKs.append(wk)
            UKs.append(uk)
        idel_rate = len(idel_machines) / num_machines

        # Average normalized machine workload
        max_WK = max(WKs)
        WKs = [wk / max_WK for wk in WKs]
        WK_avg = sum(WKs) / num_machines

        wk_sum = 0
        for w in WKs:
            wk_sum += pow(w - WK_avg, 2)
        WK_std = math.sqrt(wk_sum / num_machines)

        # Standard deviation of machine utilization rate
        UK_avg = sum(UKs) / num_machines
        sum_U = 0
        for u in UKs:
            sum_U += pow(u - UK_avg, 2)
        # Standard deviation of normalized machine workload
        UK_std = math.sqrt(sum_U / num_machines)

        OP = np.zeros(len(jobs_list))
        CRJ = np.zeros(len(jobs_list))
        opera_num = []
        for job in jobs_list:
            counter = 0
            n = len(job.operation_list)
            opera_num.append(n)
            for opera in job.operation_list:
                if opera.completed:
                    OP[counter] += 1
                else:
                    break
            CRJ[counter] = OP[counter] / len(job.operation_list)
            counter += 1
        # Completion rate of operations
        if n_jobs == 0:
            CRO = 0
            CRJ_avg = 0
            CRJ_std = 0
        else:
            CRO = sum(OP) / sum(opera_num)
            # Average job completion rate
            CRJ_avg = sum(CRJ) / n_jobs
            # Standard deviation of job completion rate
            sum_CRJ = 0
            for i in range(n_jobs):
                sum_CRJ += pow(CRJ[i] - CRJ_avg, 2)
            CRJ_std = math.sqrt(sum_CRJ / n_jobs)

        # Estimated Tardiness Rate Tard_e and Actual Tardiness Rate Tard_a
        CKs = []
        ETO = []
        for m in self.machines:
            CKs.append(m.currentTime)
        T_cure = np.mean(CKs)
        min_CK = min(CKs)
        num_unfinished_jobs = 0
        sum_unOpera = []
        ATJ = []
        UC = []
        ATJ_num_inOpera = []
        UC_num_inOpera = []
        for j in jobs_list:
            C_last = 0
            if j.completed:
                C_last = j.endTime
                uncompleted_num = 0
            if not j.completed:
                UC.append(j)
                num_unfinished_jobs += 1
                all_t_avg = []
                sum_t_avg = 0
                uncompleted_num = 0
                for o in j.operation_list:
                    if o.completed:
                        C_last = o.endTime
                    if not o.completed:
                        uncompleted_num += 1
                        cMachines = o.cMachines
                        t_avg = np.mean(list(cMachines.values()))
                        sum_t_avg += t_avg
                        if max(T_cure, C_last) + sum_t_avg > j.DT:
                            ETO.append(o)
                sum_unOpera.append(uncompleted_num)
                if max(C_last, min_CK) > j.DT:
                    ATJ.append(j)
                    ATJ_num_inOpera.append(uncompleted_num)
            # UC_num_inOpera.append(num_unfinished_jobs)
        if sum(sum_unOpera) == 0:
            Tard_e = 0
        else:
            Tard_e = len(ETO) / sum(sum_unOpera)  # Estimated Tardiness Rate Tard_e
        if sum(sum_unOpera) == 0 or len(sum_unOpera) or len(ATJ_num_inOpera) == 0:
            Tard_a = 0
        else:
            Tard_a = sum(ATJ_num_inOpera) / sum(sum_unOpera)  # Actual Tardiness Rate Tard_a
        # jobs_Infor = [n_jobs, CRO, CRJ_avg, CRJ_std, Tard_e, Tard_a]
        jobs_Infor = [CRO, CRJ_avg, CRJ_std, Tard_e, Tard_a]
        states.append(jobs_Infor)
        # machines_Infor = [num_machines, idel_time_avg, U_avg, U_std, W_avg, W_std]
        machines_Infor = [idel_rate, UK_avg, UK_std, WK_avg, WK_std]
        states.append(machines_Infor)
        return states


    def get_state_feature(self, jobs_list, machines_list):
        # total number of machines for the task
        num_machines = len(machines_list)
        UK = []
        # total number of jobs for the task in current system
        n_jobs = len(jobs_list)
        # Average utilization rate of machines
        workloads = []
        idle_times = []
        for m in machines_list:
            CT= m.currentTime
            sum_pt = 0
            for j in jobs_list:
                for o in j.operation_list:
                    if o.assignedMachine == m.name:
                        sum_pt += o.cMachines[m.name]
            idle_times.append(self.env.now - sum_pt)
            workloads.append(sum_pt)
            if CT == 0:
                u = 0
            else:
                u = sum_pt / CT
            UK.append(u)
        U_avg = sum(UK) / num_machines
        idel_time_avg = sum(idle_times) / num_machines

        # the normalized workload of machines
        normal_workloads = []
        max_w = max(workloads)
        for item in workloads:
            if max_w == 0:
                n_w = 0
            else:
                n_w = item / max_w
            normal_workloads.append(n_w)

        # Average normalized machine workload
        W_avg = sum(normal_workloads) / num_machines

        # Standard deviation of normalized machine workload
        sum_w = 0
        for item in workloads:
            sum_w += pow(item - W_avg, 2)
        W_std = sum_w / num_machines

        # Standard deviation of machine utilization rate
        sum_U = 0
        for u in UK:
            sum_U += pow(u - U_avg, 2)
        U_std = math.sqrt(sum_U / num_machines)

        OP = np.zeros(len(jobs_list))
        CRJ = np.zeros(len(jobs_list))
        opera_num = []
        for job in jobs_list:
            counter = 0
            n = len(job.operation_list)
            opera_num.append(n)
            for opera in job.operation_list:
                if opera.completed:
                    OP[counter] += 1
                else:
                    break
            CRJ[counter] = OP[counter] / len(job.operation_list)
            counter += 1
        # Completion rate of operations
        if n_jobs == 0:
            CRO = 0
            CRJ_avg = 0
            CRJ_std = 0
        else:
            CRO = sum(OP) / sum(opera_num)
            # Average job completion rate
            CRJ_avg = sum(CRJ) / n_jobs
            # Standard deviation of job completion rate
            sum_CRJ = 0
            for i in range(n_jobs):
                sum_CRJ += pow(CRJ[i] - CRJ_avg, 2)
            CRJ_std = math.sqrt(sum_CRJ / n_jobs)

        # Estimated Tardiness Rate Tard_e and Actual Tardiness Rate Tard_a
        CKs = []
        ETO = []
        for m in self.machines:
            CKs.append(m.currentTime)
        T_cure = np.mean(CKs)
        min_CK = min(CKs)
        num_unfinished_jobs = 0
        sum_unOpera = []
        ATJ = []
        UC = []
        ATJ_num_inOpera = []
        UC_num_inOpera = []
        for j in jobs_list:
            C_last = 0
            if j.completed:
                C_last = j.endTime
                uncompleted_num = 0
            if not j.completed:
                UC.append(j)
                num_unfinished_jobs += 1
                all_t_avg = []
                sum_t_avg = 0
                uncompleted_num = 0
                for o in j.operation_list:
                    if o.completed:
                        C_last = o.endTime
                    if not o.completed:
                        uncompleted_num += 1
                        cMachines = o.cMachines
                        t_avg = np.mean(list(cMachines.values()))
                        sum_t_avg += t_avg
                        if max(T_cure, C_last) + sum_t_avg > j.DT:
                            ETO.append(o)
                sum_unOpera.append(uncompleted_num)
                if max(C_last, min_CK) > j.DT:
                    ATJ.append(j)
                    ATJ_num_inOpera.append(uncompleted_num)
            # UC_num_inOpera.append(num_unfinished_jobs)
        if sum(sum_unOpera) == 0:
            Tard_e = 0
        else:
            Tard_e = len(ETO) / sum(sum_unOpera)  # Estimated Tardiness Rate Tard_e
        if sum(sum_unOpera) == 0 or len(sum_unOpera) or len(ATJ_num_inOpera) == 0:
            Tard_a = 0
        else:
            Tard_a = sum(ATJ_num_inOpera) / sum(sum_unOpera) # Actual Tardiness Rate Tard_a

        return [num_machines, n_jobs, U_avg, U_std, CRO, CRJ_avg, CRJ_std, idel_time_avg, W_avg, W_std, Tard_e, Tard_a]


    def get_state(self, jobs_list, machines_list, previous_feature):
        current_feature = self.get_state_feature(jobs_list, machines_list)
        D = [current_feature[i] - previous_feature[i] for i in range(len(current_feature))]
        current_feature.extend(D)
        return current_feature


    def get_global_features(self):
        # 1. about machines
        # 1.1 Average utilization rate of machines
        CKs = []
        n_machine = len(self.machines)
        idel_machines = []
        WKs = []
        UKs = []
        for m in self.machines:
            if m.currentTime <= self.env.now:
                idel_machines.append(m)
            CKs.append(m.currentTime)

            wk = 0
            uk = 0
            operation_list = m.assignedOpera
            if len(operation_list) != 0:
                for op in operation_list:
                    wk += op.duration
                uk = wk / m.currentTime
            WKs.append(wk)
            UKs.append(uk)
        idel_rate = len(idel_machines) / n_machine
        T_cure = np.mean(CKs)
        min_CK = min(CKs)

        # Average normalized machine workload
        max_WK = max(WKs)
        # 1.3 Average normalized machine workload
        WKs = [wk / max_WK for wk in WKs]
        WK_avg = sum(WKs) / n_machine
        wk_sum = 0
        for w in WKs:
            wk_sum += pow(w - WK_avg, 2)
        # 1.4 the standard deviation of normalized machine workload
        WK_std = math.sqrt(wk_sum / n_machine)

        # Standard deviation of machine utilization rate
        UK_avg = sum(UKs) / n_machine
        sum_U = 0
        for u in UKs:
            sum_U += pow(u - UK_avg, 2)
        # Standard deviation of normalized machine workload
        # 1.2 Standard deviation of machine utilization rate
        UK_std = math.sqrt(sum_U / n_machine)

        # about tasks
        task_states = []
        for ta in range(self.num_tasks):
            CRJ = []
            UC = []
            ETO = []
            ATJ = []
            ATJ_num_inOpera = []
            sum_unOpera = []
            num_unfinished_jobs = 0
            job_list = self.tasks_list[ta].jobsList
            opera_num = []

            if len(job_list):
                OPs = []
                ATJ_NUM = 0
                UC_NUM = 0
                all_oper_num = 0
                for job in job_list:
                    op = 0
                    C_last = 0
                    n = len(job.operation_list)
                    opera_num.append(n)
                    if job.completed:
                        C_last = job.endTime
                        op = len(job.operation_list)
                    if not job.completed:
                        UC.append(job)
                        num_unfinished_jobs += 1
                        sum_t_avg = 0
                        uncompleted_num = 0
                        for opera in job.operation_list:
                            if opera.completed:
                                C_last = opera.endTime
                                op += 1
                            if not opera.completed:
                                uncompleted_num += 1
                                cMachines = opera.cMachines
                                t_avg = np.mean(list(cMachines.values()))
                                sum_t_avg += t_avg
                                if max(T_cure, C_last) + sum_t_avg > job.DT:
                                    ETO.append(opera)
                        UC_NUM += uncompleted_num
                        sum_unOpera.append(uncompleted_num)
                        if max(C_last, min_CK) > job.DT:
                            ATJ_NUM += len(job.operation_list) - op
                            ATJ.append(job)
                            ATJ_num_inOpera.append(uncompleted_num)
                    # UC_num_inOpera.append(num_unfinished_jobs)
                    OPs.append(op)
                    Crj_i = op / len(job.operation_list)
                    CRJ.append(Crj_i)
                # Completion rate of operations
                CRO = sum(OPs) / sum(opera_num)
                # Average job completion rate
                CRJ_avg = sum(CRJ) / len(CRJ)
                # Standard deviation of job completion rate
                sum_CRJ = 0
                for i in range(len(job_list)):
                    sum_CRJ += pow(CRJ[i] - CRJ_avg, 2)
                CRJ_std = math.sqrt(sum_CRJ / len(job_list))
                if len(ETO) == 0:
                    Tard_e = 0
                else:
                    Tard_e = len(ETO) / sum(sum_unOpera)  # Estimated Tardiness Rate Tard_e
                if UC_NUM ==0 or ATJ_NUM == 0:
                    Tard_a = 0
                else:
                    Tard_a = ATJ_NUM / UC_NUM    # Actual Tardiness Rate Tard_a
            else:
                CRO = 0
                CRJ_avg = 0
                CRJ_std = 0
                Tard_e = 0
                Tard_a = 0
            task_states.append([CRO, CRJ_avg, CRJ_std, Tard_e, Tard_a])
        task_states.append([idel_rate, UK_avg, UK_std, WK_avg, WK_std])
        return task_states

    def get_global_states(self, previous_feature):
        current_feature = self.get_global_features()
        for ta in range(len(current_feature)):
            task_features = current_feature[ta]
            pre_task_features = previous_feature[ta]
            D = [task_features[i] - pre_task_features[i] for i in range(len(task_features))]
            current_feature[ta].extend(D)
        return current_feature

    def machine_failure(self):
        if self.env.now == self.starTime:
            machine = self.random_state.choice(self.machines)
            machine.available = False
            if machine.currentTime > self.env.now:
                processingOpera = machine.assignedOpera[-1]
                self.putBactOpera(processingOpera)
                del machine.assignedOpera[-1]
            self.failure_machine = copy.deepcopy(machine)
            self.machines.remove(machine)
            repairTime = random.randint(1, 99)
            self.finishTime = self.starTime + repairTime
            if self.finishTime not in self.decision_points:
                self.decision_points.append(self.finishTime)
                self.decision_points = sorted(self.decision_points)
    def putBactOpera(self, opera):
        opera.assignedMachine = ""
        opera.assigned = False
        opera.completed = False
        opera.startTime = 0
        opera.duration = 0
        opera.endTime = 0
        opera.endTime = 0
        J = self.tasks_list[opera.taskID].jobsList[opera.jobID]
        if opera.idOpertion == 0:
            J.RT = 0
        if opera.idOpertion == len(J.operation_list) - 1:
            J.completed = False
            self.num_finished -= 1
            J.endTime = 0
            del self.completed_jobs[J.idTask][-1]

    def machine_repair(self):
        if self.env.now == self.finishTime:
            self.failure_machine.available = True
            self.failure_machine.currentTime = self.env.now
            self.machines.append(self.failure_machine)
            if self.env.now not in self.decision_points:
                self.decision_points.append(self.env.now)
                self.decision_points = sorted(self.decision_points)

    def dynamic_event(self):
        if self.index_job < self.num_new_job:
            # if self.env.now == self.arrival_interval[self.index_job]:
            if self.env.now == self.next_new_job_AT:

                task_id = self.random_state.randint(0, self.num_tasks - 1)
                job = self.Jobs_Initializationr(task_id, self.env.now)
                self.jobs.append(job)
                self.index_job += 1
                self.record_jobs_arrival()
                self.tasks_list[task_id].jobsList.append(job)

                self.dynamic_contor += 1
                if self.index_job < self.num_new_job:
                    self.next_new_job_AT += self.arrival_interval[self.index_job]
                    if self.next_new_job_AT not in self.decision_points:
                        self.decision_points.append(self.next_new_job_AT)
                    if self.next_new_job_AT == self.env.now:
                        task_id = self.random_state.randint(0, self.num_tasks - 1)
                        job = self.Jobs_Initializationr(task_id, self.env.now)
                        self.jobs.append(job)
                        self.index_job += 1
                        self.record_jobs_arrival()
                        self.tasks_list[task_id].jobsList.append(job)
                        self.dynamic_contor += 1
                        if self.index_job < self.num_new_job:
                            self.next_new_job_AT += self.arrival_interval[self.index_job]
                            if self.next_new_job_AT not in self.decision_points:
                                self.decision_points.append(self.next_new_job_AT)
                    self.decision_points = sorted(self.decision_points)

                if self.env.now not in self.decision_points:
                    self.decision_points.append(self.env.now)
                    self.decision_points = sorted(self.decision_points)

    def record_jobs_arrival(self):
        #Record the time and the number of new job arrivals
        self.in_system_job_num += 1
        self.in_system_job_dic[self.env.now] = self.in_system_job_num


    def step(self, jobs, machines, action):


        assigned_opera_mac = eval(action)(jobs, machines, self.env.now)
        all_values_not_none = all(value is None for value in
                                  assigned_opera_mac.values())
        PTs = []

        if not all_values_not_none:
            for mac, opera in assigned_opera_mac.items():
                if opera is not None:
                    machine = None
                    for m in machines:
                        if m.name == mac:
                            machine = m
                            break
                    opera.assignedMachine = mac
                    opera.assigned = True
                    opera.startTime = self.env.now
                    opera.duration = opera.cMachines[mac]
                    # opera.endTime = opera.getEndTime()
                    opera.endTime = opera.startTime + opera.duration
                    opera.completed = True
                    machine.currentTime = opera.endTime
                    machine.assignedOpera.append(opera)
                    # machine.state = 'busy'
                    J = self.tasks_list[opera.taskID].jobsList[opera.jobID]
                    # J = jobs[opera.jobID]
                    # J.span = J.getSpan()

                    if opera.idOpertion == 0:
                        J.RT = opera.startTime

                    if opera.idOpertion == len(J.operation_list) - 1:
                        J.completed = True
                        self.num_finished += 1
                        J.endTime = J.getEndTime()
                        self.completed_jobs[J.idTask].append(J)


                    if machine.currentTime not in self.decision_points:
                        self.decision_points.append(machine.currentTime)
                        self.decision_points = sorted(self.decision_points)

                    if self.num_finished == self.num_jobs:
                        self.done = True

                        for i in range(self.num_tasks):
                            self.tasks_list[i].completed = True
                            if len(self.completed_jobs[i]) > 0:

                                last_J = self.completed_jobs[i][-1]
                                self.tasks_list[i].endTime = last_J.endTime
                            else:
                                self.tasks_list[i].endTime = 0
                    # machine.state = 'idle'
                    PTs.append(opera.endTime)
        return PTs

    def act(self, jobs, machines, action, machine_queue, selected_task):

        queue = {}
        idel_machine = []
        for m in machines:
            queue[m.name] = machine_queue[m.name]
            if m.currentTime <= self.env.now:
                idel_machine.append(m)
        assigned_opera_mac, queue = eval(action)(self.tasks_list, jobs, machines, selected_task, self.env.now, queue)
        for m in machines:
            machine_queue[m.name] = queue[m.name]
        # assigned_opera_mac = eval(action)(jobs, machines, self.env.now)
        all_values_not_none = all(value is None for value in assigned_opera_mac.values())
        PTs = []

        if not all_values_not_none:
            for mac, opera in assigned_opera_mac.items():
                if opera is not None:
                    machine = None
                    for m in machines:
                        if m.name == mac:
                            machine = m
                            break
                    opera.assignedMachine = mac
                    opera.assigned = True
                    opera.startTime = self.env.now
                    opera.duration = opera.cMachines[mac]
                    # opera.endTime = opera.getEndTime()
                    opera.endTime = opera.startTime + opera.duration
                    opera.completed = True
                    machine.currentTime = opera.endTime
                    machine.assignedOpera.append(opera)
                    # machine.state = 'busy'
                    J = self.tasks_list[opera.taskID].jobsList[opera.jobID]
                    # J = jobs[opera.jobID]
                    # J.span = J.getSpan()

                    if opera.idOpertion == 0:
                        J.RT = opera.startTime

                    if opera.idOpertion == len(J.operation_list) - 1:
                        J.completed = True
                        self.num_finished += 1
                        J.endTime = J.getEndTime()
                        self.completed_jobs[J.idTask].append(J)


                    if machine.currentTime not in self.decision_points:
                        self.decision_points.append(machine.currentTime)
                        self.decision_points = sorted(self.decision_points)


                    if self.num_finished == self.num_jobs:
                        self.done = True

                        for i in range(self.num_tasks):
                            self.tasks_list[i].completed = True
                            if len(self.completed_jobs[i]) > 0:

                                last_J = self.completed_jobs[i][-1]
                                self.tasks_list[i].endTime = last_J.endTime
                            else:
                                self.tasks_list[i].endTime = 0
                    # machine.state = 'idle'
                    PTs.append(opera.endTime)
        return PTs, machine_queue



def Exponential_arrival(E_ave, total_jobs_num):
    '''
    ref{Real-Time Scheduling for Dynamic Partial-No-Wait Multiobjective Flexible Job Shop by Deep Reinforcement Learning}
    In training and testing environments, the new jobs are assumed to arrive following Poisson distribution, that is, the interarrival time between two successive new job insertions is subjected to exponential distribution exp(1/λnew) with the mean value of λnew.
    :param E_ave:
    :param total_jobs_num:
    :return:
    '''

    A = np.random.exponential(E_ave, total_jobs_num).round()
    A = [A[i] + 1 for i in range(len(A))]
    A = sorted(A)
    A = np.cumsum(A)
    return A

def select_task(machines, decision_point):
    # step 1: find the idle machines
    idle_machines = []
    queue = {}
    for m in machines:
        if m.currentTime <= decision_point:
            idle_machines.append(m)
    selected_task = random.randint(0,2)
    return selected_task, idle_machines