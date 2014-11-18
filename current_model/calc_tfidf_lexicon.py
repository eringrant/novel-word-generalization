# This code assign a probability to each feature in the meaning representation
# of a word. The probability is calculated by computing tfidf like scores for features.
# Each wordnet category considered as a documnet, and all the features in that category are 
# considered as the words in that document. 


# Older version
# The prob. is calculated by computing tfidf  score of that feature.
# where the frequency of each feature is wither one or zero, and the doc frequecny
# is the number of times it occurr with other words.

import sys
import math
import os
exppath = os.path.abspath('../../experiments')
sys.path.append(exppath)
from find_sense import *


inlexname = sys.argv[1]
outlexname = sys.argv[2]

word_feat = {} #mapping a word to its features
feat_count= {} #counting number of times a feature occured in all words

noun_categ = {} # label of nouns
categ_feat = {} # a map with number of features in each categ
feat_categ_count = {} # counting the number of times a feature occured in all categs

file = open(inlexname, 'r')
for line in file:
    index = line.find(" ")
    word = line[0:index]
    features = list(set(line[index + 1:].split(",")[:-1]))
    if word == "":
        continue
    word_feat[word] = []
    for f in features:
        f = f.split(":")[0]
        word_feat[word].append(f)
        if not feat_count.has_key(f):
            feat_count[f] = 0
        feat_count[f] += 1

    # finding the category of a word

    pos = word[word.find(":")+1:]
    w = word[:word.find(":")]
    senses = wn.synsets(w, wn.NOUN) #TODO NOUN 
    if len(senses) < 1: 
        continue
    right_sense = get_right_sense(w)
    label = senses[right_sense].lexname
    noun_categ[word] = label
    
    if not categ_feat.has_key(label):
        categ_feat[label] = {}
    for f in features:
        f = f.split(":")[0]

        if not categ_feat[label].has_key(f):
            categ_feat[label][f] = 0
        categ_feat[label][f] += 1

        if not feat_categ_count.has_key(f):
            feat_categ_count[f] = []
        if not label in feat_categ_count[f]:
            feat_categ_count[f].append(label)
    
outfile = open(outlexname, 'w')
# The total number of words
words_num = len(word_feat.keys())
# the total number of categories
categs_num = len(categ_feat.keys())

print "categ num", categs_num
for word in word_feat:
    outstr = word + " "
    
    #number of features in the word
    iscateg = False
    catlabel = ""
    feat_num = len(word_feat[word])
    # if it is a noun -- look at the categ
    if noun_categ.has_key(word):
        catlabel = noun_categ[word]
        iscateg = True
        feat_num = 0
        for f in categ_feat[catlabel]:
            feat_num += categ_feat[catlabel][f]
        print "feat num", feat_num, len(categ_feat[catlabel].keys())

    # a map to store the features and their scores
    feat_score = {}
    max_score = 0
    for f in word_feat[word]:
        
        if iscateg == True:
            tf = categ_feat[catlabel][f] / float(feat_num)
            idf = math.log( categs_num/ float(len(feat_categ_count[f])))
#            print f, tf, idf
        else:
            tf = 1.0 / feat_num
            idf = math.log(words_num / float(feat_count[f]))
        
        #outstr += (f + ":" + str(tf * idf) + ",")
        feat_score[f] = tf * idf
        if feat_score[f] > max_score:
            max_score = feat_score[f]
    
#    print outstr
    scale_factor = 1.0 / max_score
    for f in feat_score:
        scaled_score = scale_factor * feat_score[f]
        outstr += (f + ":" + str(scaled_score) + ",")

    outfile.write(outstr + "\n" + "\n")
    

