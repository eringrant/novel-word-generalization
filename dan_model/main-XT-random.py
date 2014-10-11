#!/usr/bin/python

import learn
import evaluate
import frequency
import wmmapping
import pickle
import copy
import matplotlib.pyplot as plt
import numpy as np

## parameter values
Beta = 200
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
for (k,v) in items.items():
    print k,"\t",v
raw_input()

tsn = 0

# need two more dalmatians for fair testing
# items['dalmatian3'] = items['dalmatian0'][:-1] + ['dalmatian3']
# items['dalmatian4'] = items['dalmatian0'][:-1] + ['dalmatian4']

# convert all items to semantic representations
reps = dict()
for (k,v) in items.items():
    rep = wmmapping.Meaning(Beta)
    for feature in v:
       rep.setValue(feature, 1.0/float(len(v)))
    reps[k] = rep

# construct teaching sets for each of the four teaching conditions
teach_1X1 = [reps['dalmatian0']]
teach_1X3 = [reps['dalmatian0']] * 3
teach_3Xsub = [reps['dalmatian0'], reps['dalmatian1'], reps['dalmatian2']]
teach_3Xbas = [reps['dalmatian0'], reps['poodle0'], reps['pug0']]
#teach_3Xsup = [reps['dalmatian0'],reps['tabby0'],reps['flounder0']]

teaching_sets = [teach_1X3, teach_3Xsub, teach_3Xbas]
teaching_names = ["3x1 dal", "3 dal", "3 dog"]
#teaching_sets = [teach_1X1,teach_1X3, teach_3Xsub, teach_3Xbas] #, teach_3Xsup]

relevant_features = ["dalmatian_f0", "dalmatian_f1", "dalmatian_f2", "dog_f0", "dog_f1", "dog_f2", "animal_f0", "animal_f1", "animal_f2", "poodle0", "pug0", "poodle_f0", "poodle_f1", "poodle_f2", "pug_f0", "pug_f1", "pug_f2"]

def relevant(ft):
    return ft in relevant_features or ft.startswith("size") \
        or ft.startswith("color") or ft.startswith("pose")

# construct test set: 2 subordinate matches, 2 basic-level matches,
# 4 superordinate matches, 2 distractors
#test_names = ['dalmatian3','dalmatian4','poodle1','pug1','tabby1','manx1','flounder1','shark1','fire-truck1','swivel-chair1']
#test_reps = [reps[n] for n in test_names]

# teaching phase -- learner trains on teaching set
def teach(learner, teach_set):
    outfile = open("RESULT_" + teaching_names[tsn] , 'w')
    i = 0
    for meaning in teach_set:
        outfile.write("TEACHING TRIAL " + str(i) + "\n")
        learner.processPair(['fep:N'],meaning.getSeenPrims(), add_dummy)

       	m = learner.transp.getMeaning("fep:N").getSortedPrims()
	m_trimmed = []
	for prim in m:
		if relevant(prim[1]):
			outfile.write(str(prim[1]))
			outfile.write(" ")
			outfile.write(str(prim[0]))
			outfile.write("\n")

	outfile.write("----------------------------------------------------\n")
    	i += 1	
 
def bar_chart(all_prims,probs):
    l0 = [probs[p][0] for p in all_prims]
    l1 = [probs[p][1] for p in all_prims]
    l2 = [probs[p][2] for p in all_prims]
    # l3 = [probs[p][3] for p in all_prims]

    ind = np.array([2.5*n for n in range(len(all_prims))])
    width = 0.5

    fig = plt.figure()
    ax = fig.add_subplot(111)

    p0 = ax.bar(ind,l0,width,color='r')
    p1 = ax.bar(ind+width,l1,width,color='g')
    p2 = ax.bar(ind+2*width,l2,width,color='b')	
    # p3 = ax.bar(ind+3*width,l3,width,color='y')	

    ax.set_ylabel("probability")
    ax.set_xticks(ind + 2 * width)
    ax.set_xticklabels(all_prims,rotation='vertical')

    title = "Learned meaning of fep\n Beta=" + str(Beta) + " Lambda=" + str(Lambda)
    ax.set_title(title)

    plt.show()
    
# --------------------------------------------- #
#      main					                    #
# --------------------------------------------- #
def main(repkl=True):
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

    i = 0
    # run experiments
    for t in teaching_sets:
        
        #outfile.write("####################### TRIAL " + str(i) + " #############################\n")
        f = open("learner.pkl",'r')
        exp_learner = pickle.load(f)
        f.close()

        teach(exp_learner, t)
	tsn += 1
    
        m = exp_learner.transp.getMeaning("fep:N")
        #outfile.write(str(m.getSortedPrims()) + "\n")                 
        for p in m.getSortedPrims():
            if(p[0] > 0.015):
                if p[1] not in all_prims:   
                    all_prims.append(p[1])
                    probs[p[1]] = [0,0,0]
                probs[p[1]][i] = p[0]
        #outfile.write("\n")
        i += 1
    #outfile.close()
    bar_chart(all_prims,probs)

       
if __name__ == "__main__":
    main()
