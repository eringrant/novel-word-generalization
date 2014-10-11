import math
import sys
import re
#import matplotlib.pyplot as plt

def log2(x):
    l2 = math.log10(x) / math.log10(2)
    return (l2)


# calculates similarity using Average Precision
def getAP(trueD, lrndLL):

    # lrndLL: ranked list [ [v_j,f_j] ], where f_j is a primitive and v_j is its probability value
    # trueD:  dictionary of original primitives, in the form { f_i: v_i }

    # calculate precision at each cut point,
    # separating primitives with different probabilities
    i = 0
    tp = 0
    cut_count = 0

    precD = {}
    while i < len(lrndLL) and tp < len(trueD.keys()):
        j = i
        while j < len(lrndLL) and lrndLL[j][0]==lrndLL[i][0]:
            if trueD.has_key(lrndLL[j][1]):
                tp = tp + 1
            j = j + 1
        if not precD.has_key(tp):
            precD[tp] = float(tp) / float(j)
        i = j

    sumprec = 0.0
    for tp in precD.keys():
        sumprec = sumprec + precD[tp]
    avgprec = sumprec / len(precD.keys())

    return avgprec


# calcualtes Euclidean length of a meaning vector
def getVLen(Beta, meaning):
    seen_count = 0
    length = 0.0
    for f in meaning.getSeenPrims():
        v = meaning.getValue(f)
        length += v * v
        seen_count += 1

    v = meaning.getUnseen()
    length += (Beta - seen_count) * v * v
    return math.sqrt(length)

# measuring the difference between two prob. distributions,
# using the Jensen--Shannon divergence
def getJSD(Beta, meaning1, meaning2):

    primsL = meaning1.getSeenPrims()
    for f in meaning2.getSeenPrims():
        if f not in primsL:
            primsL.append(f)

    sum1 = 0.0
    sum2 = 0.0
    for f in primsL: 
        p = meaning1.getValue(f)
        q = meaning2.getValue(f)
        m = .5 * (p + q)
        if m > 0.0:
            if p > 0.0:
                sum1 = sum1 + p * log2(p / m) 
            if q > 0.0:
                sum2 = sum2 + q * log2(q / m) 

    seen_count = len(primsL)
    p = meaning1.getUnseen()
    q = meaning2.getUnseen()
    m = .5 * (p + q)
    if m > 0.0:
        if p > 0.0:
            sum1 += (Beta - seen_count) * p * log2(p / m)
        if q > 0.0:
            sum2 += (Beta - seen_count) * q * log2(q / m)

    jsd = 0.5 * (sum1 + sum2)
    return jsd


# calculates similarity using Cosine
def getCos(Beta, meaning1, meaning2):

    primsL = meaning1.getSeenPrims()
    for f in meaning2.getSeenPrims():
        if f not in primsL:
            primsL.append(f)

    cos = 0.0
    for f in primsL: 
        p = meaning1.getValue(f)
        q = meaning2.getValue(f)
        cos = cos + (p * q)

    seen_count = len(primsL)
    cos = cos + ((Beta - seen_count) * meaning1.getUnseen() * meaning2.getUnseen())

    v1 = getVLen(Beta, meaning1)
    v2 = getVLen(Beta, meaning2)

    cos = cos / (v1 * v2)

    return cos


def calculateSimilarity(Beta, meaning1, meaning2, simtype):
    sim = 0.0
#    if simtype == 'sum':
#        sim = 
    if simtype == 'cos':
        sim = getCos(Beta, meaning1, meaning2)
    elif simtype == 'jsd':
        sim = 1.0 - getJSD(Beta, meaning1, meaning2)
    return sim


class WordProps:
    def __init__(self, time, vsize):
        self.wfreq = 1				# word frequency, set to 1 after first occurrence
        self.first_time = time		# time of first occurrence
        self.first_vsize = vsize    # vocab_size at first occurrence
        self.lrnd_time = -1         # time at which the word is learned
        self.lrnd_freq = -1         # occurrences needed to learn word, set to -1 at the beginning, since word not learned yet
        self.lrnd_vsize = -1        # vocab_size when word is learned, set to -1 at the beginning, since word not learned yet
        self.last_time = -1			# last time the word has been observed

    def Write(self, handle):
        handle.write(str(self.wfreq)+","+str(self.first_time)+","+str(self.first_vsize)+","+str(self.lrnd_time)+","+str(self.lrnd_freq)+","+str(self.lrnd_vsize)+","+str(self.last_time)+",\n")

    def getWFreq(self):
        return self.wfreq

    def getFTime(self):
        return self.first_time
 
    def getFVSize(self):
        return self.first_vsize

    def getLrndFreq(self): 
        return self.lrnd_freq

    def getLrndVSize(self):
        return self.lrnd_vsize

    def getLrndTime(self): 
        return self.lrnd_time

    def getLastTime(self):
        return self.last_time

    def updateLastTime(self, time):
        self.last_time = time

    def setWFreq(self, wfreq):
        self.wfreq = wfreq

    def setLrndTime(self, lrnd_time):
        if self.lrnd_time == -1:
            self.lrnd_time = lrnd_time

    def setLrndFreq(self, freq):
        if self.lrnd_freq == -1:
            self.lrnd_freq = freq
            return True
        return False

    def setLrndVsize(self, vsize):
        if self.lrnd_vsize == -1:
            self.lrnd_vsize = vsize
            return True
        return False

    def incWFreq(self):
        self.wfreq = self.wfreq + 1


class WordPropsTable:
    def __init__(self, name):
        self.name = name
        self.points = {}

    def reset(self):
        self.points = {}

    def Write(self, handle):
        wL = self.points.keys()
        wL.sort()
        for w in wL:
            handle.write(w+"=")
            self.points[w].Write(handle)
        handle.write("\n")

    def Read(self, handle):
        while 1:
            line = handle.readline().strip("\n")
            if line=="":
                break
            w, l = line.split("=")
            propsL = re.findall("([^,]+),", l)
            self.addWord(w, int(propsL[1]), int(propsL[2]))
            self.updateWFreq(w, int(propsL[0]))
            self.updateLrndProps(w, int(propsL[3]), int(propsL[4]), int(propsL[5]))
            self.updateLastTime(w, int(propsL[6]))

    def hasWord(self, word):
        if self.points.has_key(word):
            return True
        return False

    def addWord(self, word, time, vsize):
        self.points[word] = WordProps(time, vsize)

    def updateWFreq(self, word, wfreq):
        self.points[word].setWFreq(wfreq)

    def increaseWFreq(self, word):
        self.points[word].incWFreq()

    def updateLastTime(self, word, time):
        self.points[word].updateLastTime(time)

    def updateLrndProps(self, word, lrnd_time, lrnd_freq, lrnd_vsize):
        self.points[word].setLrndTime(lrnd_time)
        self.points[word].setLrndFreq(lrnd_freq)
        return self.points[word].setLrndVsize(lrnd_vsize)

    def getWFreq(self, word):
        if self.points.has_key(word):
            return self.points[word].getWFreq()
        return -1

    def getFTime(self, word):
        if self.points.has_key(word):
            return self.points[word].getFTime()
        return -1

    def getFVSize(self, word):
        if self.points.has_key(word):
            return self.points[word].getFVSize()
        return -1

    def getLrndFreq(self, word): 
        if self.points.has_key(word):
            return self.points[word].getLrndFreq()
        return -1

    def getLrndVSize(self, word):
        if self.points.has_key(word):
            return self.points[word].getLrndVSize()
        return -1

    def getLrndTime(self, word): 
        if self.points.has_key(word):
            return self.points[word].getLrndTime()
        return -1

    def getLastTime(self, word):
        if self.points.has_key(word):
            return self.points[word].getLastTime()
        return -1

    def getAllWords(self, minfreq):
        if minfreq==0:
            allwordsL = self.points.keys()
        else:
            allwordsL = []
            for w in self.points.keys():
                if self.points[w].getWFreq() > minfreq:
                    allwordsL.append(w)
        return allwordsL

    def getWCount(self, minfreq):   # returns number of words whose frequency is greater than minfreq
        count = 0
        for w in self.points.keys():
            if self.points[w].getWFreq() > minfreq:
                count = count + 1
        return count

    def getProps(self):
        maxfreq = 0
        timeL = []
        exposuresL = []
        for item in self.points.values():
            lrnd_freq = item.getLrndFreq()
            if not lrnd_freq == -1:    # word is learned
                timeL.append(item.getFTime())
                exposuresL.append(lrnd_freq)
                if lrnd_freq > maxfreq:
                    maxfreq = lrnd_freq
        return timeL, exposuresL, maxfreq


    def plotProps(self, outdir, maxsents, Lambda):

        (xL, yL, maxfreq) = self.getProps()
        ## plot fast mapping effects
        plt.clf()
        plt.axis([0,maxsents,0,maxfreq])
        plt.plot(xL, yL, 'bo')
        plt.xlabel('time of first exposure')
        plt.ylabel('number of usages needed to learn')
        title = "Overall Pattern of Learning Novel Words"
        plt.title(title)
        figname = "%sfm_%d_%d.pdf" % (outdir, Lambda, maxsents)
        plt.savefig(figname)


class TimeProps:
    def __init__(self, heard, learned, avgcompD):
        self.heard   = heard      # no. of word types heard at some point before t
        self.learned = learned    # no. of word types learned at some point before t
        self.avgcompD = avgcompD.copy()

    def getHeard(self): 
        return self.heard

    def getLearned(self):
        return self.learned

#    def getLRate(self):
#        return self.lrate

    def getAvgcomp(self, key):
        return self.avgcompD[key]

#    def getAvgcompLrnd(self):
#        return self.avgcomp_lrnd
#
#    def getAvgcompNovel(self):
#        return self.avgcomp_novel

    def getAvgsimAll(self):
        return self.avgsim_all

    def getAvgsimLrnd(self):
         return self.avgsim_lrnd


class TimePropsTable:
    def __init__(self, name):
        self.name = name
        self.points = {}

    def reset(self):
        self.points = {}

    def hasTime(self, time):
        if self.points.has_key(time):
            return True
        return False

    def addTime(self, time, heard, learned, avgcompD):
        self.points[time] = TimeProps(heard, learned, avgcompD)

    def getVocabProps(self):

        # data for plotting vocabulary growth and overall learning curves
        timeL = []
        ratioL = []

        heardL = []
        learnedL = []

        hunit_lrndD = {}

        prev_count = 0
        for time in self.points.keys():
            item = self.points[time]
            learned_no = item.getLearned()
            heard_no   = item.getHeard()
            if learned_no == 0:
                ratio = 0.0
            else:
                ratio = float(learned_no) / float(heard_no)
            timeL.append(time)
            ratioL.append(ratio)
            heardL.append(heard_no)
            learnedL.append(learned_no)
            count = learned_no
            if heard_no % 50 == 0 and count > prev_count:
                diff = count - prev_count
                if not hunit_lrndD.has_key(heard_no):
                    hunit_lrndD[heard_no] = diff
                else:
                    hunit_lrndD[heard_no] += diff
                prev_count = count

        heardunitL = []
        difflrndL  = []
        hL = hunit_lrndD.keys()
        hL.sort()
        for h in hL:
            heardunitL.append(h)
            difflrndL.append(hunit_lrndD[h])

        return timeL, heardL, learnedL, ratioL, heardunitL, difflrndL


    def getCompProps(self):
        timeL = []
        compA = []
        compL = []
        compN = []
        compV = []
        compO = []
        for time in self.points.keys():
            item = self.points[time]
            timeL.append(time)
            compA.append(item.getAvgcomp('ALL'))
            compL.append(item.getAvgcomp('LRN'))
            compN.append(item.getAvgcomp('N'))
            compV.append(item.getAvgcomp('V'))
            compO.append(item.getAvgcomp('OTH'))

        return timeL, compA, compL, compN, compV, compO


    def plotProps(self, outdir, maxsents, Lambda): #, alpha, epsilon):

        (timeL, heardL, learnedL, ratioL, heardunitL, difflrndL) = self.getVocabProps()
        (timeL, compA, compL, compN, compV, compO) = self.getCompProps()

        plt.clf()
        plt.axis([0,maxsents,0,1])
        plt.plot(timeL, ratioL, 'b-')
        plt.xlabel('time')
        plt.ylabel('proportion of words learned')
        title = "Learning Curve"
        plt.title(title)
        figname = "%slcurves_%d_%d.pdf" % (outdir, Lambda, maxsents)
        plt.savefig(figname)

        maxheard = self.points[maxsents].getHeard()
        plt.clf()
        plt.axis([0,maxheard,0,1])
        plt.plot(heardL, ratioL, 'b-')
        plt.xlabel('word types received')
        plt.ylabel('proportion of words learned')
        title = "Vocabulary Growth"
        plt.title(title)
        figname = "%svgrowth_%d_%d.pdf" % (outdir, Lambda, maxsents)
        plt.savefig(figname)

        plt.clf()
        yminL = []
        for i in heardunitL:
            yminL.append(0)
        plt.axis([0,maxheard,0,1])
        plt.vlines(heardunitL, yminL, difflrndL)
        plt.xlabel('word types received')
        plt.ylabel('number of words learned per 10 words received')
        title = "Vocabulary Spurt"
        plt.title(title)
        figname = "%svspurt_%d_%d.pdf" % (outdir, Lambda, maxsents)
        plt.savefig(figname)

        plt.clf()
        plt.axis([0,maxsents,0,1])
        plt.plot(timeL, compA, 'k-', timeL, compL, 'r-', timeL, compN, 'g--', timeL, compV, 'g-.', timeL, compO, 'g:')
        plt.xlabel('time')
        plt.ylabel('Compscore for All, Lrnd, N, V, & O')
        title = "Comprehension Score"
        plt.title(title)
        figname = "%scompscores_%d_%d.pdf" % (outdir, Lambda, maxsents)
        plt.savefig(figname)
