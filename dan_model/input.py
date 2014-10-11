import re

class InputPair:
    def __init__(self, sentence, meaning):
        self.sentence = sentence.strip('\n')
        self.sentence = self.sentence + " "
        self.meaning  = meaning.strip('\n')
        self.meaning  = self.meaning + ","
        self.meaning  = self.meaning.replace(",,",",")

    def getWords(self):
        List = re.findall("([^ ]+)\s", self.sentence) 
        del List[0]
        #AFSI-2010: remove duplicate words
        wordsD = {}
        for w in List:
           wordsD[w] = 1
        return wordsD.keys()

    def getPrims(self):
        List = re.findall("([^,]+),", self.meaning) 
        del List[0]
        #AFSI-2010: remove duplicate prims
        primsD = {}
        for f in List:
           primsD[f] = 1
        return primsD.keys()


class InputFile:
    def __init__(self, name):
        self.name = name
        self.handle = open(self.name)

    def getNextPair(self):
        line = self.handle.readline()		# read and throw away delimiter line
        if line == "":
           return([],[])
        else:
           pair = InputPair(self.handle.readline(), self.handle.readline())
           return (pair.getWords(),pair.getPrims()) 

    def close(self):
        self.handle.close()
