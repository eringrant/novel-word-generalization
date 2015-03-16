import sys
import os
core_path = os.path.abspath('../core')
sys.path.append(core_path)


import pickle
import learn
import learnconfig
import input
import constants as CONST
import evaluate
from wmmapping import Meaning
import itertools

"""
The Fast Mapping Experiment of Vlach & Sandhofer 2012 (Fast Mapping Across Time)
"""

#===============================================================================
#     Input Parameters
#===============================================================================
corpus_path = sys.argv[1]
lexicon_path = sys.argv[2]

config_path = sys.argv[3]
# May be empty string to avoid
stopwords_path = sys.argv[4]
outdir  = sys.argv[5]

if outdir[-1] != '/':
    outdir += '/'
if not os.path.exists(outdir):
    os.makedirs(outdir)

stopwords = []
if len(stopwords_path) > 2: # At least a.b as name 
    stopwords_file = open(stopwords_path, 'r')
    for line in stopwords_file:
        stopwords.append(line.strip()+ ":N") 



#===============================================================================
#    Experiment
#===============================================================================

def read_test_objects(test_path):
    test_objects = {}
    test_file = open(test_path, 'r')
    condition = ""
    for line in test_file:
        if line.startswith("#"):
            condition = line.strip().split()[1]
            test_objects[condition] = []
        else:
            y = line.strip().split()
            
        #    test_objects[condition].append(y)
             
            tmp = []
            for f in y:
                
                if f.startswith("basic"):
                    for i in range(0,4):
                        tmp.append(f+str(i))
            
                elif f.startswith("super"):
                    for i in range(0,6):
                        tmp.append(f+str(i))
                else:
                    tmp.append(f)

            test_objects[condition].append(tmp)
            
    return test_objects

novelw = "FEP"
training_path ="xt_training.txt"
test_path ="xt_test.txt"
test_objects = read_test_objects(test_path)
learner_config = learnconfig.LearnerConfig(config_path)

training_file = file(training_path,'r')
line = training_file.readline()
while line:
    cond = line.strip().split()[1] + " "+ line.strip().split()[-1]  
 
    learner = learn.Learner(lexicon_path, learner_config, stopwords)
    
    while True:
        line = training_file.readline()
        if line.startswith("#") or line == "":
            break 
        features = line.strip().split()
        
        tmp = []
        for f in features:
            if f.startswith("basic"):
                for i in range(0,4):
                    tmp.append(f+str(i))
            
            if f.startswith("super"):
                for i in range(0,6):
                    tmp.append(f+str(i))
        features += tmp
        

        learner.process_pair([novelw], features, outdir, None)
    
    
    
    novelw_meaning =  learner._learned_lexicon.meaning(novelw)        
    print cond, novelw_meaning
    
    
    for condition in ['Sub', 'Basic', 'Super']:
        for y in test_objects[condition]:
            
            learner_dump = open(outdir + "/tmpfile_learner.pkl", "wb")
            pickle.dump(learner, learner_dump)
            learner_dump.close()
 
            
            novelw_features = novelw_meaning.seen_features()
            
            feature_subset = list(itertools.combinations(novelw_features,1))
            feature_subset += list(itertools.combinations(novelw_features,2))
            feature_subset += list(itertools.combinations(novelw_features,3))

            new_prob = 0.
            for feats in feature_subset:
                tmpp = 0.
                for y_i in y:
                    if y_i in feats:
                        tmpp +=  novelw_meaning.prob(y_i)
                    else:
                        tmpp += 1. /learner._beta
                new_prob += (tmpp/len(feats))
            
            new_prob /= len(feature_subset)
            print "test1---", condition, y, "subsets %.2f" % new_prob       

            #print "novel word", novelw_features, "features", feature_subset
            print 

            #Y|Fep
            mulpyfep = 1.
            sumpyfep = 0.
            
             
            p_yi = []
            for y_i in y:
                pyifep =  novelw_meaning.prob(y_i)
#                if not y_i in novelw_meaning.seen_features(): pyifep = 0.4 #TODO 
                mulpyfep *=  pyifep
                sumpyfep  += pyifep
                p_yi.append(pyifep)
            #print "test", condition, y, "mul p(Y|FEP) %.2f" % mulpyfep, "sum p(Y|FEP) %.2f" % (sumpyfep/len(p_yi)), p_yi
            '''
            learner.process_pair([novelw], y, outdir, None)
            novelw_meaning_2 =  learner._learned_lexicon.meaning(novelw)        
            
            
            allfeatures = novelw_meaning.seen_features() | novelw_meaning_2.seen_features()
            
            nwm_str = ""
            nwm2_str = ""
            for f in allfeatures:
                nwm_str += (f + ":%.2f " % novelw_meaning.prob(f))
                nwm2_str+= (f + ":%.2f " % novelw_meaning_2.prob(f))


            print "meaning at time t  ", nwm_str, "\nmeaning at time t+1", nwm2_str,\
            "\ncosine %.2f" % evaluate.calculate_similarity(len(allfeatures), novelw_meaning, novelw_meaning_2, CONST.COS), "\n"

            '''
            '''
            mulpyfep = 1.
            p_yi = []
            for y_i in y:
                pyifep =  novelw_meaning.prob(y_i)
                #if not y_i in novelw_meaning.seen_features(): pyifep = 0.1 #TODO 
                mulpyfep *=  pyifep
                p_yi.append(pyifep)
            '''
           # print novelw_meaning, "test", condition, y, "p(Y|FEP) %.2f" % mulpyfep, p_yi
           # print "------------------------"
 
            '''
            learner_dump = open(outdir + "/tmpfile_learner.pkl", "rb")
            learner = pickle.load(learner_dump)
            learner_dump.close()
            '''
            break
    
    if line == "": break


    
    


