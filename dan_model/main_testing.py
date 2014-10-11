#!/usr/bin/python

import learn
import evaluate
import frequency
import wmmapping
import pickle
import copy
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter

## parameter values
Beta = 400
Lambda = 0.004
alpha = 0
epsilon = 0
lexname = "lexicon-XT.all"
corpus = "input-XT.dev"
outdir = "."

## set default values for the optional arguments
add_dummy = 1
eval_mode = "vocab"
maxsents = -1
minfreq = 0
traceword = ""
simtype = 'cos'
theta = 0.7
testtype = ""

# load the file with individual test objects from the ontology
f = open("individuals.pkl",'r')
items = pickle.load(f)
f.close()

f = open("testitems.pkl",'r')
test_set = pickle.load(f)
f.close()
for (k,v) in test_set.items():
    print k,"\t",v
raw_input()

# TEACHING SET NUMBER
tsn = 0

# convert all items to meaning objects with their semantic representations
reps = dict()
for (k,v) in items.items():
    rep = wmmapping.Meaning(Beta)
    for feature in v:
       rep.setValue(feature, 1.0/float(len(v)))
    reps[k] = rep

for (k,v) in test_set.items():
    rep = wmmapping.Meaning(Beta)
    for feature in v:
       rep.setValue(feature, 1.0/float(len(v)))
    reps[k] = rep


# construct teaching sets for each of the four teaching conditions
teach_1X1 = [reps['dalmatian0']]
teach_1X3 = [reps['dalmatian0']] * 3
teach_3Xsub = [reps['dalmatian0'], reps['dalmatian1'], reps['dalmatian2']]
teach_3Xbas = [reps['dalmatian0'], reps['poodle0'], reps['pug0']]
teach_3Xsup = [reps['dalmatian0'],reps['tabby0'],reps['flounder0']]

teaching_sets = [teach_1X3, teach_3Xsub, teach_3Xbas, teach_3Xsup]
teaching_names = ["3x1 dal", "3 dal", "3 dog", "3 anim"]


relevant_features = ["dalmatian_f0", "dalmatian_f1", "dalmatian_f2", "dog_f0", "dog_f1", "dog_f2", "animal_f0", "animal_f1", "animal_f2", "poodle0", "pug0", "poodle_f0", "poodle_f1", "poodle_f2", "pug_f0", "pug_f1", "pug_f2"]

def relevant(ft):
    return ft in relevant_features or ft.startswith("size") \
        or ft.startswith("color") or ft.startswith("pose")

# construct test set: 2 subordinate matches, 2 basic-level matches,
# 4 superordinate matches, 2 distractors
#test_names = ['dalmatian3','dalmatian4','poodle1','pug1','tabby1','manx1','flounder1','shark1','fire-truck1','swivel-chair1']
#test_reps = [reps[n] for n in test_names]

# teaching phase -- learner trains on teaching set
def teach(learner, teach_set, log):
    log.write("TEACHING SET: " + teaching_names[tsn])
    i = 0
    for meaning in teach_set:
        learner.processPair(['fep:N'],meaning.getSeenPrims(), add_dummy)
       	m = learner.transp.getMeaning("fep:N").getSortedPrims()

def bar_chart(scores, test_items):
    l0 = [scores[0][i] for i in test_items]
    l1 = [scores[1][i] for i in test_items]
    l2 = [scores[2][i] for i in test_items]
    l3 = [scores[3][i] for i in test_items]

    ind = np.array([2.5*n for n in range(len(test_items))])
    width = 0.5

    fig = plt.figure()
    ax = fig.add_subplot(111)

    p0 = ax.bar(ind,l0,width,color='r')
    p1 = ax.bar(ind+width,l1,width,color='g')
    p2 = ax.bar(ind+2*width,l2,width,color='b')	
    p3 = ax.bar(ind+3*width,l3,width,color='y')	

    ax.set_ylabel("total score")
    ax.set_xlabel("test item")
    #ax.set_xticks(ind + 2.5 * width)
    #ax.set_xticklabels(test_items,rotation='vertical')

    title = "Generalization scores\n Beta=" + str(Beta) + " Lambda=" + str(Lambda)
    ax.set_title(title)

    plt.show()

def test(learner, log):
    fep_meaning = learner.transp.getMeaning("fep:N")
    probs = []
    totals = dict()
   
    for (item, features) in test_set.items():
        item_rep = reps[item]
        prob = 0        

        total_sim_to_obj = 0
        total_sim_to_fep = 0
        total_word_freq = 0
        for word in learner.transp.getWords():
            word_meaning = learner.transp.getMeaning(word)
            total_sim_to_obj += evaluate.getCos(Beta, word_meaning, item_rep)
            total_sim_to_fep += evaluate.getCos(Beta, fep_meaning, word_meaning)
            total_word_freq += float(learner.wordsp.getWFreq(word))

        #print item
        #print item_rep.getSortedPrims()
        #print learner.transp.getWords()
        #raw_input()

        for word in learner.transp.getWords():
            word_meaning = learner.transp.getMeaning(word)
            #print word, learner.wordsp.getWFreq(word)
            #print evaluate.getCos(Beta, fep_meaning, word_meaning)/total_sim_to_fep
            #print evaluate.getCos(Beta, word_meaning, item_rep)/total_sim_to_obj
            #print learner.wordsp.getWFreq(word)/total_word_freq
            #raw_input()

            prob += evaluate.getCos(Beta, fep_meaning, word_meaning)/total_sim_to_fep \
                    * evaluate.getCos(Beta, word_meaning, item_rep)/total_sim_to_obj  \
                    * learner.wordsp.getWFreq(word)/total_word_freq

        probs.append((item, prob))
        

    for key, value in sorted(probs, key=lambda x: -1*x[1]):
        totals[key] = value
        log.write(str(key) + ": " + str(value) + "\n")

    return totals
# --------------------------------------------- #
#      main					                    #
# --------------------------------------------- #
def main(repkl=True, rescore=False):
    global tsn
    if repkl:      
        # learn the meanings of words from input corpus, and update the learning curves
        learner = learn.Learner(Beta, Lambda, alpha, epsilon, simtype, theta, lexname, outdir, add_dummy, traceword, minfreq)
        (j1, j2, rfD) = learner.processCorpus(corpus, add_dummy, maxsents, 10000)
        f = open("learner.pkl",'w')        
        pickle.dump(learner,f)
        f.close()

    #outfile = open("RESULT.TXT",'w')
    all_prims = []
    probs = {}
    all_scores = []

    log = open("out.log",'w')
    i = 0

    if rescore:
    # run experiments
        for t in teaching_sets:
        
            f = open("learner.pkl",'r')
            exp_learner = pickle.load(f)
            f.close()

            teach(exp_learner, t, log)
            all_scores.append(test(exp_learner, log))
            tsn += 1

        scorefile = open("scores.pkl",'w')
        pickle.dump(all_scores, scorefile)
        scorefile.close()
    
    scorefile = open("scores.pkl",'r')
    all_scores = pickle.load(scorefile)
    scorefile.close()

    log.close()
    bar_chart(all_scores,test_set)
                                 
if __name__ == "__main__":
    main(repkl=False, rescore=False)
