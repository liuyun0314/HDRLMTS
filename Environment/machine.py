
class Machine():
    def __init__(self, id_mahicne, current_time):
        self.idMachine = id_mahicne
        self.name = 'M' + str(self.idMachine)
        self.currentTime = current_time
        self.assignedOpera = []
        self.state = 'idle'
        self.available = True

    def exportToDict(self):
        """Serialize information about Machine into dictionary"""
        exData = {}
        exData['machineName'] = self.name
        exData['currentTime'] = self.currentTime
        exData['assignedOper'] = self.assignedOpera
        return exData
