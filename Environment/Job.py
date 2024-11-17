#=========================================================================================
import random
import numpy as np

class Job():
    """Represent job to-do in schedule"""

    # def __init__(self, aItineraryName, aItineraryColor, aTaskNumber, aItineraryNumber, aMachineName):
    def __init__(self, task_id, id_job, weight, arrival_time, operations_list):
        self.idTask = task_id
        self.idJob = id_job
        self.JobName = 'J'+ str(self.idJob)
        self.weight = weight
        self.operation_list = operations_list
        self.endTime = 0
        self.assigned = False
        self.RT = self.getReleaseTime()  # release time of a job, i.e. the earliest time at which a job can start processing
        # self.span = self.getSpan()  # the process time of a job
        self.span = self.endTime - self.RT  # the process time of a job
        self.AT = arrival_time   # arrival time of a job
        self.DDT = 1.5   # random.uniform(0.5, 1.5)  due date tightness, ref{Real-Time Scheduling for Dynamic Partial-No-Wait Multiobjective Flexible Job Shop by Deep Reinforcement Learning}
        self.DT = self.getDueTime()       # due time of a job
        self.completed = False
        self.wait_time = self.RT - self.AT

    def __hash__(self):
        return hash(str(self))

    def getTupleStartAndDuration(self):
        return (self.RT, self.span)

    def getEndTime(self):
        endTime = 0
        if len(self.operation_list) != 0:
            for index in range(len(self.operation_list)):
                if self.operation_list[index].completed:
                    endTime = self.operation_list[index].endTime
        return endTime

    def getDueTime(self):
        # di = Ai + (sum(mean ti,j,k))*DDT ref. {Dynamic scheduling for flexible job shop with new job insertions by deep reinforcement learning}
        self.dueTime = 0
        if len(self.operation_list) != 0:
            t_i_j = []
            for o in self.operation_list:
                cmachines = o.cMachines
                t_j = sum(cmachines.values()) / len(cmachines)
                t_i_j.append(t_j)
            self.dueTime = np.round(self.AT + self.DDT * sum(t_i_j))
        return self.dueTime

    def getReleaseTime(self):
        if len(self.operation_list) != 0:
            self.RT = self.operation_list[0].startTime
        return self.RT