import re

class Frequency:

    def __init__(self, corpus):
        self.freqs = dict()
        self.total = 0
        self.processCorpus(corpus)

    def processCorpus(self, path):
        infile = open(path,'r')
        for line in infile:
            if not line.startswith('SENTENCE'):
                continue
            else:
                pat = re.compile(" ([^:]+):")
                wds = re.findall(pat, line)
                wds = wds[1:]
                print wds
                for wd in wds:
                    self.count(wd)

    def count(self, wd):
        if wd in self.freqs:
            self.freqs[wd] = self.freqs[wd] + 1
        else:
            self.freqs[wd] = 1.0
        self.total += 1

    def getFreq(self,wd):
        if wd in self.freqs:
            return self.freqs[wd]
        else:
            return 0

    def getTotal(self):
        return self.total
        
