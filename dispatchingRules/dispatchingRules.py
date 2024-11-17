import os
import torch
import params
from PPO.rewards import *
import simpy
import random
import numpy as np
import datetime
from Environment.Job import Job
from Environment.Operation import Operation

# The assigned machine for the opration is the one with the least process time, and the next assigned operation is with the least processing time
def SPT_SPTM(jobs, machines, current_time):
    processed_opera = {}
    # Get ready operation
    ready_opera = ready_operations(jobs, current_time)
    machine_queue = {}
    for mac in machines:
        machine_queue[mac.name] = []
        processed_opera[mac.name] = None
    # routing

    for index, opera in ready_opera.items():
        if opera is not None:
            cMachines = opera.cMachines
            if len(cMachines):
                maXPT = max(cMachines.values())
                selected_mac = max(cMachines.keys())
                for name, PT in cMachines.items():
                    if PT <= maXPT and name in machine_queue.keys():
                        selected_mac = name
                        maXPT = PT
                if selected_mac in machine_queue.keys():
                    machine_queue[selected_mac].append(opera)

    # sequencing
    for mac in machines:
        operas = machine_queue[mac.name]
        if len(operas):
            PT = []
            for o in operas:
                PT.append(o.cMachines[mac.name])
            index = np.argmin(PT)
            op = operas[index]
            processed_opera[mac.name] = op
            op.assigned = True
            machine_queue[mac.name].remove(op)

    return processed_opera

def SPT_LWT(jobs, machines, current_time):
    processed_opera = {}
    ready_opera = ready_operations(jobs, current_time)
    less_PTs = {}
    machine_queue = {}
    for m in machines:
        machine_queue[m.name] = []
        less_PTs[m.name] = 0
        que = m.assignedOpera
        for op in que:
            less_PTs[m.name] += op.cMachines[m.name]

    # routing
    for index, opera in ready_opera.items():
        if opera is not None:
            cMachines = opera.cMachines
            if len(cMachines):
                TPT = np.Inf
                selected_mac = None
                for name in cMachines.keys():
                    if name in machine_queue.keys():
                        if less_PTs[name] < TPT:
                            TPT = less_PTs[name]
                            selected_mac = name
                if selected_mac is not None:
                    machine_queue[selected_mac].append(opera)

    # sequencing
    for mac in machines:
        operas = machine_queue[mac.name]
        if len(operas):
            PT = []
            for o in operas:
                PT.append(o.cMachines[mac.name])
            index = np.argmin(PT)
            op = operas[index]
            processed_opera[mac.name] = op
            op.assigned = True
            machine_queue[mac.name].remove(op)

    return processed_opera

def LPT_SPTM(jobs, machines, current_time):
    processed_opera = {}
    # Get ready operation
    ready_opera = ready_operations(jobs, current_time)
    machine_queue = {}
    for mac in machines:
        machine_queue[mac.name] = []
        processed_opera[mac.name] = None
    # routing
    # for index, opera in ready_opera.items():
    #     if opera is not None:
    #         cMachines = opera.cMachines
    #         if len(cMachines):
    #             selected_mac = min(cMachines, key=cMachines.get)
    #             if selected_mac in machine_queue.keys():
    #                 machine_queue[selected_mac].append(opera)
    for index, opera in ready_opera.items():
        if opera is not None:
            cMachines = opera.cMachines
            if len(cMachines):
                maXPT = max(cMachines.values())
                selected_mac = max(cMachines.keys())
                for name, PT in cMachines.items():
                    if PT <= maXPT and name in machine_queue.keys():
                        selected_mac = name
                        maXPT = PT
                if selected_mac in machine_queue.keys():
                    machine_queue[selected_mac].append(opera)

    # for index, opera in ready_opera.items():
    #     if opera is not None:
    #         PTs = {}
    #         for m in opera.cMachines.keys():
    #             if m in machine_queue.keys():
    #                 PTs[m] = opera.cMachines[m]
    #         if len(PTs) != 0:
    #             selected_key = min(PTs, key=PTs.get)
    #             machine_queue[selected_key].append(opera)

    # sequencing
    for mac in machines:
        operas = machine_queue[mac.name]
        if len(operas):
            PT = []
            for o in operas:
                PT.append(o.cMachines[mac.name])
            index = np.argmax(PT)
            op = operas[index]
            processed_opera[mac.name] = op
            op.assigned = True
            machine_queue[mac.name].remove(op)

    return processed_opera


def LPT_LWT(jobs, machines, current_time):
    processed_opera = {}
    ready_opera = ready_operations(jobs, current_time)
    less_PTs = {}
    machine_queue = {}
    for m in machines:
        machine_queue[m.name] = []
        less_PTs[m.name] = 0
        que = m.assignedOpera
        for op in que:
            less_PTs[m.name] += op.cMachines[m.name]

    # routing
    for index, opera in ready_opera.items():
        if opera is not None:
            cMachines = opera.cMachines
            if len(cMachines):
                TPT = np.Inf
                selected_mac = None
                for name in cMachines.keys():
                    if name in machine_queue.keys():
                        if less_PTs[name] < TPT:
                            TPT = less_PTs[name]
                            selected_mac = name
                if selected_mac is not None:
                    machine_queue[selected_mac].append(opera)


    # sequencing
    for mac in machines:
        operas = machine_queue[mac.name]
        if len(operas):
            PT = []
            for o in operas:
                PT.append(o.cMachines[mac.name])
            index = np.argmax(PT)
            op = operas[index]
            processed_opera[mac.name] = op
            op.assigned = True
            machine_queue[mac.name].remove(op)

    return processed_opera

def MOPNR_SPTM(jobs, machines, current_time):
    processed_opera = {}
    # Get ready operation
    ready_opera = ready_operations(jobs, current_time)
    machine_queue = {}
    for mac in machines:
        machine_queue[mac.name] = []
        processed_opera[mac.name] = None
    # routing
    # for index, opera in ready_opera.items():
    #     if opera is not None:
    #         cMachines = opera.cMachines
    #         if len(cMachines):
    #             selected_mac = min(cMachines, key=cMachines.get)
    #             if selected_mac in machine_queue.keys():
    #                 machine_queue[selected_mac].append(opera)
    for index, opera in ready_opera.items():
        if opera is not None:
            cMachines = opera.cMachines
            if len(cMachines):
                maXPT = max(cMachines.values())
                selected_mac = max(cMachines.keys())
                for name, PT in cMachines.items():
                    if PT <= maXPT and name in machine_queue.keys():
                        selected_mac = name
                        maXPT = PT
                if selected_mac in machine_queue.keys():
                    machine_queue[selected_mac].append(opera)
    # for index, opera in ready_opera.items():
    #     if opera is not None:
    #         PTs = {}
    #         for m in opera.cMachines.keys():
    #             if m in machine_queue.keys():
    #                 PTs[m] = opera.cMachines[m]
    #         if len(PTs) != 0:
    #             selected_key = min(PTs, key=PTs.get)
    #             machine_queue[selected_key].append(opera)

    # sequencing
    for m in machines:
        opra_list = machine_queue[m.name]
        if len(opra_list):
            remain_opera_num = []
            for key, op in enumerate(opra_list):
                job = jobs[op.jobID]
                finished_opera_num = 0
                for o in job.operation_list:
                    if o.completed:
                        finished_opera_num += 1
                    else:
                        break
                unfinished_opera_num = len(job.operation_list) - finished_opera_num
                remain_opera_num.append(unfinished_opera_num)
            index = np.argmax(remain_opera_num)
            selected_op = opra_list[index]
            processed_opera[m.name] = selected_op
            selected_op.assigned = True
            machine_queue[m.name].remove(selected_op)

    return processed_opera

def MOPNR_LWT(jobs, machines, current_time):
    processed_opera = {}
    ready_opera = ready_operations(jobs, current_time)
    less_PTs = {}
    machine_queue = {}
    for m in machines:
        machine_queue[m.name] = []
        less_PTs[m.name] = 0
        que = m.assignedOpera
        for op in que:
            less_PTs[m.name] += op.cMachines[m.name]

    # routing
    for index, opera in ready_opera.items():
        if opera is not None:
            cMachines = opera.cMachines
            if len(cMachines):
                TPT = np.Inf
                selected_mac = None
                for name in cMachines.keys():
                    if name in machine_queue.keys():
                        if less_PTs[name] < TPT:
                            TPT = less_PTs[name]
                            selected_mac = name
                if selected_mac is not None:
                    machine_queue[selected_mac].append(opera)
    # sequencing
    for m in machines:
        opra_list = machine_queue[m.name]
        if len(opra_list):
            remain_opera_num = []
            for key, op in enumerate(opra_list):
                job = jobs[op.jobID]
                finished_opera_num = 0
                for o in job.operation_list:
                    if o.completed:
                        finished_opera_num += 1
                    else:
                        break
                unfinished_opera_num = len(job.operation_list) - finished_opera_num
                remain_opera_num.append(unfinished_opera_num)
            index = np.argmax(remain_opera_num)
            selected_op = opra_list[index]
            processed_opera[m.name] = selected_op
            selected_op.assigned = True
            machine_queue[m.name].remove(selected_op)

    return processed_opera

def MWKR_SPTM(jobs, machines, current_time):
    processed_opera = {}
    # Get ready operation
    ready_opera = ready_operations(jobs, current_time)
    machine_queue = {}
    for mac in machines:
        machine_queue[mac.name] = []
        processed_opera[mac.name] = None
    # routing
    # for index, opera in ready_opera.items():
    #     if opera is not None:
    #         cMachines = opera.cMachines
    #         if len(cMachines):
    #             selected_mac = min(cMachines, key=cMachines.get)
    #             if selected_mac in machine_queue.keys():
    #                 machine_queue[selected_mac].append(opera)
    for index, opera in ready_opera.items():
        if opera is not None:
            cMachines = opera.cMachines
            if len(cMachines):
                maXPT = max(cMachines.values())
                selected_mac = max(cMachines.keys())
                for name, PT in cMachines.items():
                    if PT <= maXPT and name in machine_queue.keys():
                        selected_mac = name
                        maXPT = PT
                if selected_mac in machine_queue.keys():
                    machine_queue[selected_mac].append(opera)
    # sequencing
    for m in machines:
        operas = machine_queue[m.name]
        all_avg_PT = []
        if len(operas):
            for o in operas:
                job = jobs[o.jobID]
                sum_PT = 0
                for o in job.operation_list:
                    if not o.completed:
                        avg_PT = sum(o.cMachines.values()) / len(o.cMachines)
                        sum_PT += avg_PT
                all_avg_PT.append(sum_PT)
            index = np.argmax(all_avg_PT)
            selected_op = operas[index]
            processed_opera[m.name] = selected_op
            selected_op.assigned = True
            machine_queue[m.name].remove(selected_op)
    return processed_opera

def MWKR_LWT(jobs, machines, current_time):
    processed_opera = {}
    ready_opera = ready_operations(jobs, current_time)
    less_PTs = {}
    machine_queue = {}
    for m in machines:
        machine_queue[m.name] = []
        less_PTs[m.name] = 0
        que = m.assignedOpera
        for op in que:
            less_PTs[m.name] += op.cMachines[m.name]

    # routing
    for index, opera in ready_opera.items():
        if opera is not None:
            cMachines = opera.cMachines
            if len(cMachines):
                TPT = np.Inf
                selected_mac = None
                for name in cMachines.keys():
                    if name in machine_queue.keys():
                        if less_PTs[name] < TPT:
                            TPT = less_PTs[name]
                            selected_mac = name
                if selected_mac is not None:
                    machine_queue[selected_mac].append(opera)
    # sequencing
    for m in machines:
        operas = machine_queue[m.name]
        all_avg_PT = []
        if len(operas):
            for o in operas:
                job = jobs[o.jobID]
                sum_PT = 0
                for o in job.operation_list:
                    if not o.completed:
                        avg_PT = sum(o.cMachines.values()) / len(o.cMachines)
                        sum_PT += avg_PT
                all_avg_PT.append(sum_PT)
            index = np.argmax(all_avg_PT)
            selected_op = operas[index]
            processed_opera[m.name] = selected_op
            selected_op.assigned = True
            machine_queue[m.name].remove(selected_op)
    return processed_opera

# The assigned machine for the opration is the one with the most process time, and the next assigned operation with shortest processing time is selected.
def LPT(jobs, machines, current_time, machine_queue):
    processed_opera = {}
    # Get ready operation
    ready_opera = get_ready_operations(jobs, current_time)
    for mac in machines:
        processed_opera[mac.name] = None
    # routing
    for index, opera in ready_opera.items():
        if opera is not None:
            min_PT = min(opera.cMachines.values())
            key = [k for k, v in opera.cMachines.items() if v == min_PT][0]
            if key in machine_queue.keys():
                machine_queue[key].append(opera)

    # sequencing
    for m in machines:
        operas = machine_queue[m.name]
        if len(operas):
            PT = []
            for o in operas:
                PT.append(o.cMachines[m.name])
            index = np.argmax(PT)
            op = operas[index]
            processed_opera[m.name] = op
            machine_queue[m.name].remove(op)

    return processed_opera, machine_queue

# The assigned machine for the opration is the one with the least process time, and the next assigned operation is from the job with with earlist arrival time.
def FIFO(jobs, machines, current_time, machine_queue):

    processed_opera = {}
    # Get ready operation
    ready_opera = get_ready_operations(jobs, current_time)
    for mac in machines:
        processed_opera[mac.name] = None

    # routing
    for index, opera in ready_opera.items():
        if opera is not None:
            min_PT = min(opera.cMachines.values())
            key = [k for k, v in opera.cMachines.items() if v == min_PT][0]
            if key in machine_queue.keys():
                machine_queue[key].append(opera)

    #sequencing
    for m in machines:
        oper_list = machine_queue[m.name]
        if len(oper_list):
            minAT = oper_list[0].AT
            min_opera = oper_list[0]
            for op in oper_list:
                AT = op.AT
                if AT < minAT:
                    minAT = AT
                    min_opera = op
            min_opera.assignedMachine = m.name
            processed_opera[m.name] = min_opera
            machine_queue[m.name].remove(min_opera)

    return processed_opera, machine_queue

# A Reinforcement Learning Approach for Flexible Job Shop Scheduling Problem With Crane Transportation and Setup Times
# The assigned machine for a operation is decided by argmin(max(CTk, CTi,j-1)), and the next assigned operation is from the job with the most uncompleted operations
def MOPNR(jobs, machines, current_time, machine_queue):
    processed_opera = {}
    # Get ready operation
    ready_opera = get_ready_operations(jobs, current_time)
    record_jobs = {}
    CT = {}
    for mac in machines:
        record_jobs[mac.name] = []
        processed_opera[mac.name] = None
        CT[mac.name] = mac.currentTime   # completed time of the last operation in the machine

    # routing argmin(max(CTk, CTi,j-1))
    for index, opera in ready_opera.items():
        if opera is not None:
            opera_id = opera.idOpertion
            if opera_id == 0:
                CT_previous = 0
            if opera_id > 0:
                job = jobs[index]
                previus_opera = preovious_operation(job, opera)
                CT_previous = previus_opera.endTime
            tmp = []
            tmp_mac = []
            for mac, currentTime in CT.items():
                if mac in opera.cMachines:
                    tmp.append(max(currentTime, CT_previous))
                    tmp_mac.append(mac)
            if len(tmp):
                min_index = np.argmin(tmp)
                selected_mac = tmp_mac[min_index]
                machine_queue[selected_mac].append(opera)

    # sequencing
        for m in machines:
            opra_list = machine_queue[m.name]
            if len(opra_list):
                remain_opera_num = []
                for key, op in enumerate(opra_list):
                    job = jobs[op.jobID]
                    finished_opera_num = 0
                    for o in job.operation_list:
                        if o.completed:
                            finished_opera_num += 1
                        else:
                            break
                    unfinished_opera_num = len(job.operation_list) - finished_opera_num
                    remain_opera_num.append(unfinished_opera_num)
                index = np.argmax(remain_opera_num)
                selected_op = opra_list[index]
                processed_opera[m.name] = selected_op
                machine_queue[m.name].remove(selected_op)
    return processed_opera, machine_queue

# The assigned machine for a operation is decided by argmin(max(CTk, CTi,j-1)), and the next assigned operation is the one with the least average processing time of all available machines.
def LAPT(jobs, machines, current_time, machine_queue):
    processed_opera = {}
    # Get ready operation
    ready_opera = get_ready_operations(jobs, current_time)
    CT = {}
    for mac in machines:
        processed_opera[mac.name] = None
        CT[mac.name] = mac.currentTime  # completed time of the last operation in the machine

    # routing argmin(max(CTk, CTi,j-1))
    for key, opera in ready_opera.items():
        if opera is not None:
            opera_id = opera.idOpertion
            if opera_id == 0:
                CT_previous = 0
            if opera_id > 0:
                job = jobs[key]
                previus_opera = preovious_operation(job, opera)
                CT_previous = previus_opera.endTime
            tmp = []
            tmp_mac = []
            for mac, currentTime in CT.items():
                if mac in opera.cMachines:
                    tmp.append(max(currentTime, CT_previous))
                    tmp_mac.append(mac)
            if len(tmp):
                index = np.argmin(tmp)
                selected_mac = tmp_mac[index]
                machine_queue[selected_mac].append(opera)

    # sequencing
    for mac, opera_list in machine_queue.items():
        PT_avg = []
        if len(opera_list):
            for op in opera_list:
                available_machines = op.cMachines
                avg_pt = sum(available_machines.values()) / len(available_machines)
                PT_avg.append(avg_pt)
            index = np.argmin(PT_avg)
            opera = opera_list[index]
            processed_opera[mac] = opera
            machine_queue[mac].remove(opera)
    return processed_opera, machine_queue


# The assigned machine for the opration is the one with the most process time, and the next assigned operation is the one with the most average processing time of all available machines.
def MAPT(jobs, machines, current_time, machine_queue):
    processed_opera = {}
    # Get ready operation
    ready_opera = get_ready_operations(jobs, current_time)
    for mac in machines:
        processed_opera[mac.name] = None
    # routing
    for index, opera in ready_opera.items():
        if opera is not None:
            min_PT = max(opera.cMachines.values())
            key = [k for k, v in opera.cMachines.items() if v == min_PT][0]
            if key in machine_queue.keys():
                machine_queue[key].append(opera)

    # sequencing
    for mac, opera_list in machine_queue.items():
        PT_avg = []
        if len(opera_list):
            for op in opera_list:
                available_machines = op.cMachines
                avg_pt = sum(available_machines.values()) / len(available_machines)
                PT_avg.append(avg_pt)
            index = np.argmax(PT_avg)
            opera = opera_list[index]
            processed_opera[mac] = opera
            machine_queue[mac].remove(opera)
    return processed_opera, machine_queue

# The assigned machine for a operation is decided by argmin(max(CTk, CTi,j-1, Ai)), and the next assigned operation is selected randomly from uncompleted operations.
def Random(jobs, machines, current_time, machine_queue):
    processed_opera = {}
    # Get ready operation
    ready_opera = get_ready_operations(jobs, current_time)
    CT = {}
    for mac in machines:
        processed_opera[mac.name] = None
        CT[mac.name] = mac.currentTime  # completed time of the last operation in the machine

    # routing argmin(max(CTk, CTi,j-1))
    for index, opera in ready_opera.items():
        if opera is not None:
            AT = opera.AT
            opera_id = opera.idOpertion
            if opera_id == 0:
                CT_previous = 0
            if opera_id > 0:
                job = jobs[index]
                previus_opera = preovious_operation(job, opera)
                CT_previous = previus_opera.endTime
            tmp = []
            tmp_mac = []
            for mac, currentTime in CT.items():
                if mac in opera.cMachines:
                    tmp.append(max(currentTime, CT_previous, AT))
                    tmp_mac.append(mac)
            if len(tmp):
                index = np.argmin(tmp)
                selected_mac = tmp_mac[index]
                machine_queue[selected_mac].append(opera)

    # sequencing
    for mac, opera_list in machine_queue.items():
        if len(opera_list):
            index = np.random.randint(0, len(opera_list))
            opera = opera_list[index]
            processed_opera[mac] = opera
            machine_queue[mac].remove(opera)
    return processed_opera, machine_queue

def ready_operations(jobs, currentTime):
    ready_operations = {}
    for index, job in enumerate(jobs):
        opera_list = job.operation_list
        if len(opera_list) != 0:
            if not opera_list[0].completed:
                ready_operations[index] = opera_list[0]
            else:
                pre_opera = opera_list[0]
                for opera in opera_list[1:]:
                    if not opera.completed:
                        if pre_opera.completed and pre_opera.endTime <= currentTime:
                            ready_operations[index] = opera
                            break
                    else:
                        pre_opera = opera
    return ready_operations

def get_unassigned_operation(jobs, currentTime):
    unassigned_opera = {}
    for index, job in enumerate(jobs):
        unassigned_opera[index] = None
        # if not job.assigned:   # Meeting this condition indicates that this operation has not been assigned yet
        op_list = job.operation_list
        op_num = len(op_list)
        if not job.completed:
            if op_num > 0:
                if not op_list[0].assigned:
                    unassigned_opera[index] = op_list[0]
                    op_list[0].assigned = True
                    continue

                pre_opera = op_list[0]
                for op in op_list[1:]:
                    if pre_opera.endTime <= currentTime and pre_opera.completed:
                        if not op.assigned:
                            unassigned_opera[index] = op
                            op.assigned = True
                            break
                        else:
                            pre_opera = op

    return unassigned_opera

def get_ready_operations(jobs, current_time):
    ready_opera = {}
    for index, job in enumerate(jobs):
        ready_opera[index] = None
        if not job.completed:   # Meeting this condition indicates that this operation has not been assigned yet
            op_list = job.operation_list
            op_num = len(op_list)
            if len(op_list) != 0:
                if not op_list[0].assigned:
                    ready_opera[index] = op_list[0]
                else:
                    if len(op_list) > 1:
                        for key in range(1, len(op_list)):
                            if not op_list[key].assigned:
                                if op_list[key-1].completed and op_list[key-1].endTime <= current_time:
                                    ready_opera[index] = op_list[key]
                                    break
    return ready_opera
def preovious_operation(job, opera):
    opera_id = opera.idOpertion
    if opera_id == 0:
        return None
    else:
        previous_id = opera_id - 1
        opera_list = job.operation_list
        for op in opera_list:
            if op.idOpertion == previous_id:
                previous_opera = op
                break
        return previous_opera

def Exponential_arrival(E_ave, total_jobs_num):
    '''
    Generate the time intervel of arrival jobs
    :param E_ave:
    :param total_jobs_num:
    :return:
    '''
    A = np.random.exponential(E_ave, total_jobs_num)
    A = [int(A[i]) for i in range(len(A))]
    A = sorted(A)
    return A

def Job_Generator(id_job, weight, arrival_time, max_operation_num, num_machines, processing_time_range):
    # generate the operation list
    operations_list = []
    operations_num = np.random.randint(1, max_operation_num)
    for i in range(operations_num):
        operation = Operation_Generator(i, id_job, num_machines, processing_time_range, arrival_time)
        operations_list.append(operation)
    task_id = np.random.randint(3)
    job = Job(task_id, id_job, weight, arrival_time, operations_list)
    return job

def Operation_Generator(id_operation, id_job, num_machines, processing_time_range, arrival_time):

    candidate_machines = {}
    # generate the information of candidate machine
    candidate_machine_num = np.random.randint(1, num_machines+1)
    sample_index = np.random.choice(range(num_machines), candidate_machine_num, replace=False)
    process_time = np.random.randint(processing_time_range[0], processing_time_range[1] + 1, candidate_machine_num)
    count = 0
    for m in sample_index:
        machine_name = 'M' + str(m)
        candidate_machines[machine_name] = process_time[count]
        count += 1
    operation = Operation(id_operation, id_job, candidate_machines, arrival_time)
    return operation

def step(sim, jobs, machines, action, queue):

    assigned_opera_mac, queue = eval(action)(jobs, machines, sim.env.now, queue)
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
                opera.startTime = sim.env.now
                opera.duration = opera.cMachines[mac]
                # opera.endTime = opera.getEndTime()
                opera.endTime = opera.startTime + opera.duration
                opera.completed = True  # 这句话不可以放在yield sim.env.timeout(opera.duration)之前。因为，这样可能导致实际系统的时间还没有到达该Opera完成的时间，但是在接下来的重调度时，由于该Opera的completed被标为了True,所以后驱Opera会被选择处理，但事实上，该Opera还没有执行完成
                machine.currentTime = opera.endTime
                machine.assignedOpera.append(opera)
                machine.state = 'busy'
                J = jobs[opera.jobID]
                # J.span = J.getSpan()

                if opera.idOpertion == 0:
                    J.RT = opera.startTime
                if opera.idOpertion == len(J.operation_list) - 1:
                    J.completed = True
                    sim.num_finished += 1
                    J.endTime = J.getEndTime()
                    sim.completed_jobs[J.idTask].append(J)

                if machine.currentTime not in sim.decision_points:
                    sim.decision_points.append(machine.currentTime)
                    sim.decision_points = sorted(sim.decision_points)
                if sim.num_finished == sim.num_jobs:
                    sim.done = True
                    # 值得注意的是，在所有新工作均到达车间之前，属于task i的全部工作被执行完成时，不可以将其completed标为True，因为后续可能会有属于task i的新工作到达
                    for i in range(sim.num_tasks):
                        sim.tasks_list[i].completed = True
                        if len(sim.completed_jobs[i]) > 0:
                            last_J = sim.completed_jobs[i][-1]
                            sim.tasks_list[i].endTime = last_J.endTime
                        else:
                            sim.tasks_list[i].endTime = 0
                machine.state = 'idle'
                PTs.append(opera.duration)
    return PTs, queue
