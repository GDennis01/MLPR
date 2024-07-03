import sys

class Competitor:
    def __init__(self,name,surname,country,avg_score):
        self.name = name
        self.surname = surname
        self.country=country
        self.avg_score = avg_score

    def __str__(self):
        return self.name+" "+self.surname+" "+self.country+" "+str(self.avg_score)        
    def __repr__(self):
        return str(self)


if __name__ == "__main__":
    competitors = []
    with open("ex_data.txt","r") as f:
        for line in f:
            fields = line.split(" ")
            scores = [fields[3],fields[4],fields[5],fields[6],fields[7]]
            scores = [float(i) for i in scores]
            scores.sort()
            avg = scores[1]+scores[2]+scores[3]
            competitors.append(Competitor(fields[0],fields[1],fields[2],avg))

    country_score=dict()

    for comp in competitors:
        if comp.country not in country_score:
            country_score[comp.country] = comp.avg_score
        else:
            country_score[comp.country]+=comp.avg_score
    country_score=sorted(country_score.items(),key=lambda x:x[1],reverse=True)
    competitors=sorted(competitors,key=lambda x:x.avg_score,reverse=True)
    print(f'final ranking:')
    print(f'1: {competitors[0].name} - Score: {competitors[0].avg_score}')
    print(f'2: {competitors[1].name} - Score: {competitors[1].avg_score}')
    print(f'3: {competitors[2].name} - Score: {competitors[2].avg_score}\n')
    print("Best Country:\n"+country_score[0][0]+" - Total Score: "+str(country_score[0][1]))
       
      
            
         
        

    pass