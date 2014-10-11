#!/usr/bin/python

import sys
import getopt
import learn
import evaluate
import wmmapping
#import numpy
#import psyco
#psyco.full()

# --------------------------------------------- #
#						usage					#
# --------------------------------------------- #
def usage():
   print "usage:"
   print "  main.py -c (--corpus) -B (--Beta) -L (--Lambda) -a (--alpha) -e (--epsilon) -l (--lexicon) -t (--simthreshold) -y (--testtype) -o (--output) [ -v (--evalmode) -f (--minfreq) -x (--maxsents) -r (--traceword) -h (--help) -d (--dummy) -s (--simmeasure)]"
   print ""
   print "  --corpus:   input corpus"
   print "  --Beta:     estimated number of semantic primitives"
   print "  --Lambda:   for smoothing of meaning probabilities"
   print "              ** could be a fixed value (smaller than 1), or an integer in range [1,n] corresponding to a Lambda function"
   print "  --alpha:    estimated number of words per utterance--scene pair"
   print "  --epsilon:  for smoothing of alignment probabilities"
   print "  --lexicon:  original lexicon"
   print "  --testtype:   red+fblock| block | fblocking | highlight | bblocking | attenuation | baserate | mini | reduced"
   print "  --output:   output directory"
   print " OPTIONAL Args: [default value is given in brackets]"
   print "  --minfreq: minimum frequency -- only words with freq. higher than this value are considered in the evaluation [0]" 
   print "  --maxsents: maximum number of input pairs to process [-1: all input pairs are processed]"
   print "  --evalmode: vocab, verbose, tracecomp, traceprob"
   print "  --dummy:    0 (no dummy), 1 & 2 (adds a dummy to each sentence) [no dummy]"
   print "  --help:     prints this usage"
   print ""


# --------------------------------------------- #
# --------------------------------------------- #
def getSumTrans(learner, w, sL):
    p = 0.0
    for s in sL:
        p += learner.transp.getValue(w,s)
    return p

# --------------------------------------------- #
# --------------------------------------------- #
def printTestInfo(learner, symptomsLL):
    #wordsL = [ '1:N', '2:N', '3:N', '4:N', '5:N', '6:N', 'dummy:N' ]
    wordsL = [ '1:N', '2:N', 'dummy:N' ]
    print 
    print "             ",
    for w in wordsL:
        print "%s     " % (w),
    print
    for sL in symptomsLL:
        print sL, "    ",
        for w in wordsL:
            p = getSumTrans(learner, w, sL) 
            print "%f    " % (p),
            #p = learner.getUnseen(w)
            #print "%f    " % (p),
        print

# --------------------------------------------- #
# print learned lexicon
# --------------------------------------------- #
def printLearnedLexicon(learner):
    #filename = "%s%s%s%s%s%s%s" % (learner.outdir, "/lexicon_lm_", learner.Lambda, "_a", learner.alpha, "_b", learner.epsilon)
    filename = "%s%s%s%s%s" % (learner.outdir, "/lexicon_B-", learner.Beta, "_lm-", learner.Lambda)
    output = open(filename, 'w')
    minprob = 0.0001
    for word in learner.wordsp.getAllWords(0):
        #tmprimsLL = learner.original_lex.getSortedPrims(word)
        lrnd_meaning = learner.transp.getMeaning(word)
        line = "%s:%d [" % (word, learner.wordsp.getWFreq(word))
        #for true_p,f in tmprimsLL:
            #if true_p > minprob:
        for f in lrnd_meaning.getSeenPrims():
                lrnd_p = lrnd_meaning.getValue(f)
                line += "%s:(%6.5f), " % (f, lrnd_p)
        line += " ]\n\n"
        output.write(line)

        comp = "   << %f >>\n\n" % (learner.getComprehensionScore(word))
        output.write(comp)
    output.close

# --------------------------------------------- #
#						main					#
# --------------------------------------------- #
def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hd:c:B:L:a:e:l:y:o:v:t:s:f:x:r:", ["help", "dummy=", "corpus=", "Beta=", "Lambda=", "alpha=", "epsilon=", "lexicon=", "testtype=", "output=",  "evalmode=", "simthreshold=", "simmeasure=", "minfreq=", "maxsents=", "traceword="])
    except getopt.error, msg:
        print msg
        usage()
        sys.exit(2)

    if len(opts) < 6:
        usage()
        sys.exit(0)

    ## set default values for the optional arguments
    add_dummy = 0
    eval_mode = ""
    maxsents = -1
    minfreq = 0
    traceword = ""
    simtype = 'cos'
    theta = 0.7
    testtype = ""

    for o, a in opts: 
        if o in ("-h", "--help"):
            usage()
            sys.exit(0)
        if o in ("-d", "--dummy"):
            add_dummy = int(a) 
        if o in ("-c", "--corpus"):
            corpus = a
        if o in ("-B", "--Beta"):
            Beta = int(a)
        if o in ("-L", "--Lambda"):
            value = float(a)
            if value < 1.0:
                Lambda = value
            else:
                Lambda = int(a)
        if o in ("-a", "--alpha"):
            alpha = int(a)
        if o in ("-e", "--epsilon"):
            epsilon = float(a)
        if o in ("-l", "--lexicon"):
            lexname = a
        if o in ("-y", "--testtype"):
            testtype = a
        if o in ("-o", "--output"):
            outdir = a
        if o in ("-v", "--evalmode"):
            eval_mode = a 
            if not eval_mode in ("vocab", "verbose", "tracecomp", "traceprob"):
                usage()
                sys.exit(0)
        if o in ("-t", "--simthreshold"):
            theta = float(a)
        if o in ("-s", "--simmeasure"):
            simtype = a
        if o in ("-f", "--minfreq"):
            minfreq = int(a)
        if o in ("-x", "--maxsents"):
            maxsents = int(a)
        if o in ("-r", "--traceword"):
            traceword = a

    # learn the meanings of words from input corpus, and update the learning curves
    learner = learn.Learner(Beta, Lambda, alpha, epsilon, simtype, theta, lexname, outdir, add_dummy, traceword, minfreq)
    (j1, j2, rfD) = learner.processCorpus(corpus, add_dummy, maxsents, 10000)

    # print props 
#    learner.wordsp.plotProps(outdir, maxsents, Lambda)
#    learner.timesp.plotProps(outdir, maxsents, Lambda)
#    learner.printNovelWordInfo()
#    printLearnedLexicon(learner)

if __name__ == "__main__":
    main()
