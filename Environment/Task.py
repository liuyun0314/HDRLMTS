class Task():
    def __init__(self, id, objective):
        self.idTask = id
        self.job_counter = 0
        self.TaskName = 'T' + str(self.idTask)
        self.objective = objective     # 'WTmean' / 'WFmean' / 'WTmax'
        self.priority = 1
        self.startTime = 0
        self.jobsList = []
        self.duration = self.getDuration()
        self.endTime = 0
        # self.finished_counter = 0
        self.completed = False

    def getDuration(self):
        if len(self.jobsList) == 0:
            self.duration = 0
        else:
            for i in range(len(self.jobsList)):
                self.duration += self.jobsList[i].PT
        return self.duration
