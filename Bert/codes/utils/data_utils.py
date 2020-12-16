import sys
import pandas as  pd

def write_csv(args,result):#record the prediction result in csv files
    result = [-1 if each==2 else each for each in result]
    id = pd.read_csv(args.submision_sample_path)[["id"]]
    result = pd.DataFrame(result,columns = ["y"])
    result = id.join(result)
    result.to_csv(args.sub_path+"result.csv",index = False)

def detailed_predictions(args,predictions):
    f = open("../subs/"+str(args.times)+"/predictions.txt","w",encoding = "utf8")
    for each in predictions:
        tem = [str(i) for i in each]
        f.write(" ".join(tem))
        f.write("\n")
    f.close()

class MyLogger(object):
    def __init__(self, filename='default.log', add_flag=True, stream=sys.stdout):
        self.terminal = stream
        print("filename:", filename)
        self.filename = filename
        self.add_flag = add_flag
        # self.log = open(filename, 'a+')

    def write(self, message):
        if self.add_flag:
            with open(self.filename, 'a+') as log:
                self.terminal.write(message)
                log.write(message)
        else:
            with open(self.filename, 'w') as log:
                self.terminal.write(message)
                log.write(message)

    def flush(self):
        pass



