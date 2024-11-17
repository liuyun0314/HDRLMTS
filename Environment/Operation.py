
class Operation():
    def __init__(self, id_operation, jobID, taskID, cMachineNames, arrival_time):
        self.taskID = taskID
        self.jobID = jobID
        self.idOpertion = id_operation
        self.OperationName = 'O' + str(self.idOpertion)
        self.AT = arrival_time
        self.weight = 1
        self.startTime = 0
        self.duration = 0
        self.endTime = 0
        self.cMachines = cMachineNames   # candidate machines{'M1':89,,,,}
        self.assignedMachine = ""
        self.completed = False
        self.assigned = False

    def getEndTime(self):
        self.endTime = self.startTime + self.duration
        return self.endTime

