import re

#=========================================================================#
class Meaning:    # added on Dec. 29
#=========================================================================#
    def __init__(self, Beta):
        self.prob = {}
        self.unseen = 1.0/float(Beta)

    def __str__(self):
        return self.prob.items()

    def __repr__(self):
        return str(self.prob.items())

    def setValue(self, prim, value):
        self.prob[prim] = value

    def getValue(self, prim):
        if not self.prob.has_key(prim):
            return self.getUnseen()
        else:
            return self.prob[prim]

    def setUnseen(self, value):
        self.unseen = value

    def getUnseen(self):
        return self.unseen

    def update(self, probD, unseen):
        for f in probD.keys():
            self.prob[f] = probD[f]
        self.unseen = unseen

    def getSeenPrims(self):
        return self.prob.keys()

    def getSortedPrims(self):
        items = self.prob.items()
        ranked = [ [v[1],v[0]] for v in items ]
        ranked.sort(reverse=True)
        return ranked

    def Print(self):
        ranked = [ [v[1], v[0]]  for v in self.prob.items() ]
        ranked.sort(reverse=True)
        print ranked, '<', self.unseen, '>',

    def Write(self, handle):
        kL = self.prob.keys()
        kL.sort()
        for k in kL:
            handle.write(k+":"+str(self.getValue(k))+",")
        handle.write("unseen:"+str(self.getUnseen())+",")
        handle.write("\n")

    def Copy(self, meaning):
        for f in meaning.getSeenPrims():
            self.setValue(f, meaning.getValue(f))
        self.setUnseen(meaning.getUnseen())


#=========================================================================#
class TransDist:
#=========================================================================#
    def __init__(self, Beta):
        self.wordsD = {}
        self.Beta = Beta

    def reset(self, word):
        self.wordsD[word] = Meaning(self.Beta)

    def initialize(self, wordsL):
        for w in wordsL:
            self.reset(w)

    def Write(self, handle):
        kL = self.wordsD.keys()
        kL.sort()
        for k in kL:
            handle.write(k+"=")
            self.wordsD[k].Write(handle)
        handle.write("\n")

    def Read(self, handle):
        while 1:
            line = handle.readline().strip("\n")
            if line=="":
                break
            w, l = line.split("=")
            self.reset(w)
            itemsL = re.findall("([^,]+),", l)
            for item in itemsL:
                k,v = item.split(":")
                if k=="unseen":
                    self.wordsD[w].setUnseen(float(v))
                else:
                    self.wordsD[w].setValue(k,float(v))

    def getWords(self):
        wordsL = self.wordsD.keys()
        return wordsL

    def setValue(self, word, prim, value):
        if not self.wordsD.has_key(word):
            self.reset(word)
        self.wordsD[word].setValue(prim, value)

    def getValue(self, word, prim):
        if self.wordsD.has_key(word):
            return self.wordsD[word].getValue(prim)
        return 0.0

    def getUnseen(self, word):
        if self.wordsD.has_key(word):
            return self.wordsD[word].getUnseen()
        return 0.0 # ERROR
#        if self.unseen.has_key(word):
#            return self.unseen[word]
#        return self.default

    def setUnseen(self, word, unseen_value):
         self.wordsD[word].setUnseen(unseen_value)
#        self.unseen[word] = unseen_value

    def getMeaning(self, word):
        meaning = Meaning(self.Beta)
        if self.wordsD.has_key(word):
            meaning.Copy(self.wordsD[word])
        return meaning
#        meaning = Meaning(self.Beta)
#        for f in self.getSeenPrims(word):
#            meaning.setValue(f, self.getValue(word, f))
#        meaning.setUnseen(self.getUnseen(word))
#        return meaning

    # ------------------------------------------------------------- #
    # returns a sorted list of pairs [ [v_j, f_j], ... ] where v_j is the probability
    # associated with the primitive f_j
    def getSortedPrims(self, word):
        if self.wordsD.has_key(word):
            return self.wordsD[word].getSortedPrims()
        return []
#        items = self.prob[word].items()
#        ranked = [ [v[1],v[0]] for v in items ]
#        ranked.sort(reverse=True)
#        return ranked

    # ------------------------------------------------------------- #
    # returns the meaning of a word, sorted and pruned, and prepared for printing
    def getMeaning2Print(self, word, wfreq, minprob):
        meaning = "%s:%d [" % (word, wfreq)
        items = self.getSortedPrims(word)
        for v,f in items:
            if v > minprob:
                meaning = meaning + "%s:%6.5f," % (f, v)
        meaning = meaning + " ]\n"
        return meaning

    # ------------------------------------------------------------- #
    # returns a list of primitives seen with word: used mostly for the original lexicon
    def getSeenPrims(self, word):
        if self.wordsD.has_key(word):
            return self.wordsD[word].getSeenPrims()
        return []
#        if not self.prob.has_key(word):
#            return []
#        return self.prob[word].keys()

    def printAll(self, wordL, primL, output):
        line = "      "
        for f in primL:
            line = line + "%s  " % f[0:5]
        line = line + "\n"
        output.write(line)

        for w in wordL:
            line = "%s  " % w
            for f in primL:
                line = line + "%.3f  " % self.getValue(w,f)
            line = line + "\n"
            output.write(line)


#=========================================================================#
class AlignDist:
#=========================================================================#
    def __init__(self, Alpha):
        self.prob = {}
        self.sumA = {}
        self.unseen = 0 #1.0/Alpha

    def reset(self):
        self.prob = {}
        self.sumA = {}
        self.unseen = 0 #1.0/Alpha

    def Write(self, handle):
        wL = self.sumA.keys()
        wL.sort()
        for w in wL:
            fL = self.sumA[w].keys()
            fL.sort()
            for f in fL:
                handle.write(w+"="+f+"="+str(self.getAssoc(w,f))+",")
        handle.write("\n\n")

    def Read(self, handle):
        while 1:
            line = handle.readline().strip("\n")
            if line=="":
                break
            itemsL = re.findall("([^,]+),", line)
            for item in itemsL:
                w,f,v = item.split("=")
                self.setAssoc(w,f,float(v))

    def setValue(self, prim, word, value):
        if not self.prob.has_key(prim):
            self.prob[prim] = {}
        self.prob[prim][word] = value

    def getValue(self, prim, word):
        if not self.prob.has_key(prim):
            return 0.0
        elif not self.prob[prim].has_key(word):
            return self.getUnseen()
        else:
            return self.prob[prim][word]

    def getUnseen(self):
        return self.unseen

    def clear(self):
        self.prob = {}

    def setAssoc(self, word, prim, value):
        if not self.sumA.has_key(word):
            self.sumA[word] = {}
        if not self.sumA[word].has_key(prim):
            self.sumA[word][prim] = value
        else:
            self.sumA[word][prim] = self.sumA[word][prim] + value

    def getAssoc(self, word, prim):    # sum_over_sentences => sum_r a(w,f; I_r)
        if not self.sumA.has_key(word) or not self.sumA[word].has_key(prim):
            return 0.0
        else:
            return self.sumA[word][prim]

    def printAll(self, wordL, primL, output):
        line = "      "
        for f in primL:
            line = line + "%s  " % f[0:5]
        line = line + "\n"
        output.write(line)

        for w in wordL:
            line = "%s  " % w
            for f in primL:
                line = line + "%.3f  " % self.getValue(w,f)
            line = line + "\n"
            output.write(line)
