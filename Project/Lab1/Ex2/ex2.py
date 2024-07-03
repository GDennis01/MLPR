import sys

class BusRecord:
    def __init__(self,id,route,x,y,time):
        self.id = id
        self.route = route
        self.x=x
        self.y = y
        self.time = time

    def __str__(self):
        return self.name+" "+self.surname+" "+self.country+" "+str(self.avg_score)        
    def __repr__(self):
        return str(self)
    

def euclidean_distance(r1, r2):
    return ((r1[0]-r2[0])**2 + (r1[1]-r2[1])**2)**0.5

def total_distance(records,busId):
    distance=0
    prev_point = None
    filtered_records = [i for i in records if i.id==busId]
    for record in filtered_records:
        if prev_point != None:
            distance+=euclidean_distance(prev_point,(record.x,record.y)) 
        prev_point=(record.x,record.y)
    return distance

def avg_speed(records,lineId):
    filtered_records = [i for i in records if i.route == lineId]
    distance=0
    time=0;
    prev_buses = dict()

    for record in filtered_records:
        if record.id in prev_buses:
            distance += euclidean_distance((record.x,record.y),(prev_buses[record.id][0],prev_buses[record.id][1]))
            time += record.time-prev_buses[record.id][2]
        prev_buses[record.id]=(record.x,record.y,record.time)
    return distance/time

def loadAllRecords(file):
    try:
        records=[]
        with open(file) as f:
            for line in f:
                id,route,x,y,time=line.split(" ")
                records.append(BusRecord(id,route,int(x),int(y),int(time)))
            return records
    except:
        raise

if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise
    records = loadAllRecords(sys.argv[1])
    if sys.argv[2] == "-b":
        print(total_distance(records,sys.argv[3]))
    elif sys.argv[2] == "-l":
        print(avg_speed(records,sys.argv[3]))
    else:
        raise 