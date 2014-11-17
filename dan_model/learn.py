import sys
import input
import lexicon
import wmmapping
import evaluate
import statistics
import re
import math

class Learner:
    def __init__(self, Beta, Lambda, alpha, epsilon, simtype, theta, lexname, outdir, add_dummy, traceword, minfreq=0):
        # parameters for smoothing meaning probabilities
        self.Beta = Beta
        self.Lambda = Lambda
        # parameters for smoothing alignment probabilities
        self.alpha = 0 #alpha
        self.epsilon = 0 #epsilon
        # threshold for determining whether a word is learned by the model
        self.simthreshold = theta
        # the similarity measure used for evaluation
        self.simtype = simtype

        self.minfreq = minfreq
        self.outdir = outdir
        self.traceword = traceword

        # translation and alignment distribution tables
        self.transp = wmmapping.TransDist(self.Beta)
        self.alignp = wmmapping.AlignDist(self.alpha)

        # word and time properties for plots
        self.wordsp = evaluate.WordPropsTable("word_props")
        self.timesp = evaluate.TimePropsTable("time_props")

        self.time = 0
        self.vocabD = {}       # hash of words whose meaning isLearned at some point in time
#        self.novelD = {}       # hash of novel words and their sim_score at time of first exposure
        self.allprimsD = {}    # hash of primitives seen so far

        self.lastcompscoreD = {}

        # the original `gold-standard' lexicon
        self.original_lex = lexicon.readAll(lexname, self.Beta)

        # initialize translation table for all words in the lexicon
        allwordsL = self.original_lex.getWords()
        if add_dummy==1: dummy = "dummy"
        elif add_dummy==2: dummy = "dummy" + str(self.time)
        if add_dummy > 0: allwordsL.append(dummy)
        self.transp.initialize(allwordsL)

        # initialize list of PoS tags to consider for counting vocabulary
        self.postagL = ['V', 'N']

    # ------------------------------------------------------------- #
    def reset(self):

        # reset translation and alignment distribution tables
        self.transp.reset()
        self.alignp.reset()

        # reset word and time properties for plots
        self.wordsp.reset()
        self.timesp.reset()

        self.time = 0
        self.vocabD = {}       # hash of words whose meaning isLearned at some point in time
        self.allprimsD = {}    # hash of primitives seen so far

    # ------------------------------------------------------------- #
    def Write(self, filename):
        handle = open(self.outdir+filename, 'w')
        handle.write(str(self.Beta)+" "+str(self.Lambda)+" "+str(self.alpha)+" "+str(self.epsilon)+" "+str(self.simthreshold)+" "+self.simtype+" "+str(self.minfreq)+" "+self.outdir+" "+str(self.time)+" \n")

        kL = self.vocabD.keys()
        kL.sort()
        for k in kL:
            handle.write(k+"="+str(self.vocabD[k])+",")
        handle.write("\n")

        kL = self.allprimsD.keys()
        kL.sort()
        for k in kL:
            handle.write(k+"="+str(self.allprimsD[k])+",")
        handle.write("\n")

        kL = self.lastcompscoreD.keys()
        kL.sort()
        for k in kL:
            handle.write(k+"="+str(self.lastcompscoreD[k])+",")
        handle.write("\n")

        self.transp.Write(handle)
        self.alignp.Write(handle)

        self.wordsp.Write(handle)
#        self.timesp.Write(handle)

        handle.close()

    # ------------------------------------------------------------- #
    def Read(self, filename):
        handle = open(self.outdir+filename, 'r')

        line = handle.readline().strip("\n")
        line += " "
        paramsL = re.findall("([^ ]+) ", line)
        self.Beta = float(paramsL[0])
        self.Lambda = float(paramsL[1])
        self.alpha = float(paramsL[2])
        self.epsilon = float(paramsL[3])
        self.simthreshold = float(paramsL[4])
        self.simtype = paramsL[5]
        self.minfreq = int(paramsL[6])
        self.outdir = paramsL[7]
        self.time = int(paramsL[8])

        line = handle.readline().strip("\n")
        itemsL = re.findall("([^,]+),", line)
        for item in itemsL:
            k,v = item.split("=")
            self.vocabD[k] = int(v)

        line = handle.readline().strip("\n")
        itemsL = re.findall("([^,]+),", line)
        for item in itemsL:
            k,v = item.split("=")
            self.allprimsD[k] = int(v)

        line = handle.readline().strip("\n")
        itemsL = re.findall("([^,]+),", line)
        for item in itemsL:
            k,v = item.split("=")
            self.lastcompscoreD[k] = float(v)

        self.transp.Read(handle)
        self.alignp.Read(handle)

        self.wordsp.Read(handle)
#        self.timesp.Read(handle)

        handle.close()

    # ------------------------------------------------------------- #
    def getLambdaFunction(self):
        #wtype_count = self.wordsp.getWCount(0)
        if self.Lambda == 1:
            return (0.5 / (math.pow(2.72, 0.005 * (self.time - 1000)) + 1.0))
        elif self.Lambda == 2:
            return (0.5 / math.pow( math.pow(2.72, 0.005 * (self.time - 1000)) + 1.0, 0.75))
        elif self.Lambda == 4:
            return (0.5 / math.sqrt(math.pow(2.72, 0.005 * (self.time - 1000)) + 1.0))
        elif self.Lambda == 5:
            return (0.5 / math.pow( math.pow(2.72, 0.005 * (self.time - 1000)) + 1.0, 0.33))
        elif self.Lambda == 6:
            return (0.5 / math.pow( math.pow(2.72, 0.005 * (self.time - 1000)) + 1.0, 0.25))
        elif self.Lambda == 7:
            return (0.5 / math.pow( math.pow(2.72, 0.005 * (self.time - 1000)) + 1.0, 0.1))

        elif self.Lambda == 8:
            return (0.5 / (math.pow(2.72, 0.005 * self.time) + 1.0))
        elif self.Lambda == 9:
            return (0.5 / (math.pow(2.72, 0.005 * (self.time - 100)) + 1.0))
        elif self.Lambda == 10:
            return (0.5 / (math.pow(2.72, 0.005 * (self.time - 1000)) + 1.0))

        if self.Lambda == 11:
            return (1.0 / (self.time + 1.0))
        elif self.Lambda == 12:
            return (1.0 / (math.pow(self.time,0.75) + 1.0))
        elif self.Lambda == 13:
            return (1.0 / (math.sqrt(self.time) + 1.0))
        elif self.Lambda == 14:
            return (1.0 / (math.pow(self.time,0.33) + 1.0))
        elif self.Lambda == 15:
            return (1.0 / (math.pow(self.time,0.25) + 1.0))
        elif self.Lambda == 16:
            return (1.0 / (math.pow(self.time,0.1) + 1.0))

        elif self.Lambda == 17:
            return (1.0 / (math.pow(self.time,2) + 1.0))

    # ------------------------------------------------------------- #
    def produceRankedList(self, targetMeaning, tasktype):
        simD = {}
        for word in self.wordsp.getAllWords(0):
            if tasktype in ['priming', 'production']:
                # get learned meaning of word
                wmeaning = self.transp.getMeaning(word)
            else:
                # task is 'comprehension'
                # get true meaning of word
                wmeaning = self.original_lex.getMeaning(word)

            simD[word] = evaluate.calculateSimilarity(self.Beta, targetMeaning, wmeaning, self.simtype)

        rankedLL = [ [v[1],v[0]] for v in simD.items() ]
        rankedLL.sort(reverse=True)
        return rankedLL

    # ------------------------------------------------------------- #
    def getPoSTag(self, w):
        word_pos = w + ":"
        wpL = re.findall("([^:]+):", word_pos)
        postag = wpL[1]
        return postag

    # ------------------------------------------------------------- #
    def calculateAvgComp(self, W, key):
        if len(W)==0:
            return 0.0
        sum = 0.0
        vsize = 0
        for w in W:
            # get postag of word w
            postag = self.getPoSTag(w)
            # check if postag matches key
            if self.isSeen(w):
                if key=='ALL' or (key in ['N','V'] and postag==key) or (key=='OTH' and not postag in ['V','N']):
                    sum = sum + self.getComprehensionScore(w)
                    vsize += 1
        if sum==0.0:
            avg = 0.0
        else:
            avg = sum / float(vsize)
        return avg

    # ------------------------------------------------------------- #
    def getComprehensionScore(self, word):
        if not self.isSeen(word):
            uniform = wmmapping.Meaning(self.Beta)
            trueMeaning = self.original_lex.getMeaning(word)
            comp = evaluate.calculateSimilarity(self.Beta, uniform, trueMeaning, self.simtype)
            return comp
        return self.lastcompscoreD[word]

    # ------------------------------------------------------------- #
    def calculateComprehensionScore(self, word):
        trueMeaning = self.original_lex.getMeaning(word)
        lrndMeaning = self.transp.getMeaning(word)
        self.lastcompscoreD[word] = evaluate.calculateSimilarity(self.Beta, lrndMeaning, trueMeaning, self.simtype)

    # ------------------------------------------------------------- #
    def isSeen(self, word):
        if self.wordsp.hasWord(word):
            return True
        return False

    # ------------------------------------------------------------- #
    def isLearned(self, word):
        if not self.isSeen(word):
            return False
        if self.getComprehensionScore(word) >= self.simthreshold:
            return True
        return False

    # ------------------------------------------------------------- #
    def updateMappingTables(self, wordsL, primsL):

        # reset alignment probabilities
        self.alignp.clear()

        # calculate alignment probs for (w,f) pairs in current sentence,
        # and update sum_align probs for (w,f) over the entire corpus
        for f in primsL:
            sumT = 0
            for w in wordsL:
                sumT = sumT + self.transp.getValue(w,f)
            denom = sumT + (self.alpha * self.epsilon)
            for w in wordsL:
                a = (self.transp.getValue(w,f) + self.epsilon) / denom
                self.alignp.setValue(w,f,a)
                self.alignp.setAssoc(w,f,a)

        # update translation probs for words in current sentence,
        # taking into account all semantic primitives seen so far
        if self.Lambda < 1.0:
            #if self.time < 2: Lambda = 0.5
            #elif self.time < 8: Lambda = 0.01
            #else: Lambda = self.Lambda
            Lambda = self.Lambda
        else:
            Lambda = self.getLambdaFunction()
            #sys.stderr.write("HERE:"+str(self.Lambda)+"-->"+str(Lambda)+"\n")
        minprob = 1.0/float(self.Beta)
        for w in wordsL:
            if w == 'dummy': Lambda = 0.5
            else: Lambda = self.Lambda
            #sys.stderr.write("HERE: <"+w+"> : "+str(Lambda)+"\n")
            sumA = 0
            for f in self.allprimsD:
                sumA = sumA + self.alignp.getAssoc(w,f)
            denom = sumA + (self.Beta * Lambda)
            for f in self.allprimsD:
                t = (self.alignp.getAssoc(w,f) + Lambda) / denom
                self.transp.setValue(w,f,t)

            t_unseen = Lambda / denom
            self.transp.setUnseen(w, t_unseen)

    # ------------------------------------------------------------- #
    def processPair(self, wordsL, primsL, add_dummy):

        # increase time by 1
        self.time = self.time + 1

        # add current primitives to the list of all primitives
        for f in primsL:
            if not self.allprimsD.has_key(f):
                self.allprimsD[f] = 1
            else:
                self.allprimsD[f] = self.allprimsD[f] + 1

        for w in wordsL:
            # if w is seen for the first time, add it to wordsp as a new word,
            # otherwise increase its frequency of occurrence by 1
            if not self.wordsp.hasWord(w):
                 # self.wordsp.addWord(w, self.time, len(self.vocabD.keys()))
                 self.wordsp.addWord(w, self.time, self.getLearnedCount(self.postagL))
                 # postag = self.getPoSTag(w)
            else:
                 self.wordsp.increaseWFreq(w)
            self.wordsp.updateLastTime(w, self.time)

        if add_dummy==1:
            dummy = "dummy"
            wordsL.append(dummy)
            print wordsL
            print primsL
        elif add_dummy==2:
            dummy = "dummy" + str(self.time)
            wordsL.append(dummy)

        # update the alignment and translation probabilities
        self.updateMappingTables(wordsL, primsL)

        # print alignments and meaning probs for all (w,f) pairs in current input
        #for w in wordsL:
        #    for f in primsL:
        #        print "a(%s|%s) -> %6.4f" %(w,f,self.alignp.getValue(w,f))
        #    for f in primsL:
        #        if not w==dummy:
        #            rfp = self.calculateRFProb(w,f)
        #            print "rf(%s|%s) -> %6.4f" %(w,f,rfp)
        #    for f in self.allprimsD.keys():
        #        print "p(%s|%s) => %4.3f" %(f,w,self.transp.getValue(w,f))

        if add_dummy!=0:
            wordsL.remove(dummy)

        # update some information for each current word, except for the DUMMY word
        for w in wordsL:

            # calculate compscore for w
            self.calculateComprehensionScore(w)

            # store the sim_score of all novel words in the current input
#            if (self.novelD.has_key(w) and self.novelD[w][0] < 0):
#                self.novelD[w][0] = self.time
#                self.novelD[w][1] = self.getComprehensionScore(w)

            # if the meaning of the word has been learned, update its properties;
            # note this is done only the first time sim(L_w,T_w) exceeds the threshold
            if not self.vocabD.has_key(w) and self.isLearned(w):
                #self.wordsp.updateLrndProps(w, self.time, self.wordsp.getWFreq(w), len(self.vocabD.keys()))
                self.wordsp.updateLrndProps(w, self.time, self.wordsp.getWFreq(w), self.getLearnedCount(self.postagL))
                if self.wordsp.getWFreq(w) > self.minfreq:
                    self.vocabD[w] = 1

    # ------------------------------------------------------------- #
    def getHeardCount(self, minfreq, postagL):
        if 'ALL' in postagL:
           return self.wordsp.getWCount(minfreq)

        allwordsL = self.wordsp.getAllWords(minfreq)
        count = 0
        for w in allwordsL:
            t = self.getPoSTag(w)
            if t in postagL:
                count += 1
        return count

    # ------------------------------------------------------------- #
    def getLearnedCount(self, postagL):
        if 'ALL' in postagL:
            return len(self.vocabD.keys())

        vocabL = self.vocabD.keys()
        count = 0
        for w in vocabL:
            t = self.getPoSTag(w)
            if t in postagL:
                count += 1
        return count

    # ------------------------------------------------------------- #
    def calculateRFProb(self, word, prim):
        allwordsL = self.wordsp.getAllWords(0)
        if not word in allwordsL: print "ERROR %s" %(word)
        num = self.transp.getValue(word, prim) * self.wordsp.getWFreq(word)
        denom = 0.0
        #num = 1.0 / self.Beta
        #denom = num
        for w in allwordsL:
            denom += self.transp.getValue(w, prim) * self.wordsp.getWFreq(w)
            #if not w==word: denom += self.transp.getValue(w, prim) * self.wordsp.getWFreq(w)
        rfp = num / denom
        return rfp

    # ------------------------------------------------------------- #
    def processCorpus(self, corpus, add_dummy, maxsents, maxlearned=0):

        # indata is a file object containing sentences and their meaning
        indata = input.InputFile(corpus)

        # each pair has two lists: the word list, and the semantic primitive list,
        # extracted from the current sentence/meaning pair

        (current_wordsL, current_primsL) = indata.getNextPair()
        sent_count = 1
        while current_wordsL != []:

            #if (maxtime > 0) and (self.time > maxtime):
            if (maxsents > 0) and (sent_count > maxsents):
                break

            print "---------- trial ", self.time, " ----------"
            avgcompD = {}
            self.processPair(current_wordsL, current_primsL, add_dummy)

            # calculate the proportion of unlearned words to received words
            received_count = self.getHeardCount(self.minfreq, self.postagL) # self.wordsp.getWCount(self.minfreq)
            learned_count  = self.getLearnedCount(self.postagL) # len(self.vocabD.keys())

            #print sent_count, self.time, received_count, learned_count

            if maxsents < 0 and learned_count > maxlearned:
                break

            # calculate average comprehension and similarity score of the all words, and of learned words
            #avgcompD['ALL']  = self.calculateAvgComp(self.wordsp.getAllWords(self.minfreq), 'ALL')
            #avgcompD['LRN'] = self.calculateAvgComp(self.vocabD.keys(), 'ALL')
            #avgcompD['N'] = self.calculateAvgComp(self.wordsp.getAllWords(self.minfreq), 'N')
            #avgcompD['V'] = self.calculateAvgComp(self.wordsp.getAllWords(self.minfreq), 'V')
            #avgcompD['OTH'] = self.calculateAvgComp(self.wordsp.getAllWords(self.minfreq), 'OTH')

            self.timesp.addTime(self.time, received_count, learned_count, avgcompD)

            (current_wordsL, current_primsL) = indata.getNextPair()
            sent_count += 1

        # calculate RF probs: rf(wug|A), rf(wug|B), rf(wug|C)
        #wL = ['wug:N']
        #oL = ['A', 'B', 'C']
        rfD = {}
        #for w in wL:
        #    rfD[w] = {}
        #    for o in oL:
        #        rfD[w][o] = self.calculateRFProb(w,o)

        indata.close()
        return learned_count, self.time, rfD
