#!/usr/bin/python

import os
import re
import random
import itertools
import pprint
import pickle

import input
import learn
import learnconfig
import numpy as np

verbose = True
check_probs = True

# input corpus
corpus_path = 'input_wn_fu_cs_scaled_categ.dev'

# gold-standard lexicon
lexname = 'all_catf_prob_lexicon_cs.all'

# output directory
outdir = 'referent_probability_experiments_output'
if not os.path.exists(outdir):
    os.makedirs(outdir)

# number of iterations for each condition
num_iterations = 2

# number of random novel words to generate experimental conditions for
num_novel_word_conditions = 2

# manually specify novel objects?
fix_novel_words = True
fixed_novel_words = [
        'nose:N'
]

# manually specify familiar objects?
fix_familiar_objects = True
fixed_familiar_objects = [
        'apple:N'
        #'piece:N'
]

parameter_values = {
# config file parameters
    'dummy' :                           [False],
    'forget' :                          [False, 0.03],
    'novelty' :                         [False, 0.03],
    'lambda' :                          [-1],
    #'lambda' :                          [-1, 0.03],
    #'power' :                           [1, 0.5, 0.25], #c = 1 is ND; c = 0.5 is LT_.5 (late talker less severe),; c = 0.25 is LT_.25 (late talker more severe)
    'power' :                           [1],
    'maxtime' :                         [10000],
    'category-learner' :                [False],
# training parameters
    #'permute-input-sentences' :         [True, False],
    'probabilistic-features' :           [True], # if True, generate a subset of the gold-standard features for the test scene
# test parameters
    #'target-type' :                     ['familiar', 'novel'],
    #'number-familiar-objects' :         [n for n in range(3)], # number of familiar objects in test array
    'number-familiar-objects' :         [1],
    #'number-novel-objects' :            [n for n in range(3)] # number of novel objects in test array
    'number-novel-objects' :            [1]
}

def get_random_sample_words(corpus_path, n, maxtime=None):
    word_list = []

    count = 0

    corpus = input.Corpus(corpus_path)
    (words, features) = corpus.next_pair()
    count += 1

    for wd in words:
        word_list.append(wd)

    while words != [] and ((count < maxtime) if maxtime is not None else True):

        if len(words) == 1:
            # Skip singleton utterances
            (words, features) = corpus.next_pair()
            continue

        for wd in words:
            if wd[-1] == 'N': # get only the nouns
                word_list.append(wd)

        (words, features) = corpus.next_pair()
        count += 1

    return random.sample(set(word_list), n)

def create_corpus_without_word(word, corpus_path):
    corpus_output_filename = corpus_path + '_without_' + word

    found_word = False
    at_SEM_REP = False

    fin = open( corpus_path )
    fout = open( corpus_output_filename , "w" )

    for line in fin:
        if word in line:
            found_word = True
            sent = line
            continue

        if found_word == True:
            found_word = False
            at_SEM_REP = True
            continue

        if at_SEM_REP == True:
            at_SEM_REP = False
            continue

        fout.write(line)

    fin.close()
    fout.close()

    return corpus_output_filename


# generate the novel word possibilities
if fix_novel_words is False:
    parameter_values['novel-word'] =  get_random_sample_words(corpus_path, num_novel_word_conditions)
else:
    parameter_values['novel-word'] = fixed_novel_words

# create and hash the paths to the modified corpora
corpora = {}
for word in parameter_values['novel-word']:
    corpora[word] = create_corpus_without_word(word, corpus_path)

# Cartesian product of all parameter settings
experiment_conditions = [dict(zip(parameter_values, x)) for x in itertools.product(*parameter_values.values())]

# read the input lexicon file and store lexemes in a probabilistic lexicon in memory
M = 10000 # based on script generate_dev_data.sh
problex = input.read_gold_lexicon(lexname, M)

def generate_scene(objects, experiment_condition, probabilistic=False):
    # ensure we have a list of objects
    if isinstance(objects, str):
        objects = [ objects ]

    scene = []

    for obj in objects:
        for v, f in problex.meaning(obj).sorted_features():
            prob = float(v)

            if probabilistic:
                r = random.random()
                if prob > r:
                    scene.append(f)

            else:
                scene.append(f)

    return scene

def calculate_referent_probability(learner, utterance, scene_generator, scene_representation, forget=False):

    # ensure we have a list of words and a list of features
    if isinstance(utterance, str):
        utterance = [ utterance ]
    if isinstance(scene_representation, str):
        scene_representation = [ scene_representation ]

    feature_prob = {} # dictionary of { feature : p (f) = \sum_{w' \in W} p ( f | w' ) * p( w' ) }
    joint_prob = {} # dictionary of { (word, feature) : p( f | w ) * p( w ) = p( f, w ) }

    for feature in scene_representation:

        feature_prob[feature] = 0.0

        for word in learner._wordsp.all_words(0): # get all words the learner has seen (with min frequency of 0)

            # p (f) = \sum_{w'} [ p ( f | w' ) * p( w' ) ] = \sum_{w'} [ p ( f, w' ) ]
            feature_prob[feature] += learner._learned_lexicon.prob(word, feature) * learner._wordsp.frequency(word)

            if word in utterance and feature in scene_representation:
                # hack to ensure meaning probability is updated for encountered words (when forget is True)
                learner.acquisition_score(word)

                joint_prob[(word, feature)] = learner._learned_lexicon.prob(word, feature) * learner._wordsp.frequency(word)

    # calculate the referent probabilities
    referent_prob = {} # dictionary of { word : p ( w | F ) }

    for seen_word in scene_generator:
        for spoken_word in utterance:
            referent_prob[(spoken_word, seen_word)] = 1

            # get the features indicating this seen word from the gold-standard lexicon
            for value, feature in problex.meaning(seen_word).sorted_features():
                if feature in scene_representation:
                    # p ( w | F ) = \prod_f [ p ( w | f ) ] = \prod_f [ p ( f, w ) / p ( f ) ]
                    referent_prob[(spoken_word, seen_word)] *= (joint_prob[(spoken_word, feature)] / feature_prob[feature])

    if verbose:
        print '------------------------------------------------------------------'
        print ''
        print 'Joint probability of word and feature:'
        pprint.pprint(joint_prob)
        print ''
        print 'Marginal probability of feature:'
        pprint.pprint(feature_prob)
        print ''
        print 'Referent probability:'
        pprint.pprint(referent_prob)
        print ''

    return referent_prob

def write_config_file(
    dummy,
    forget,
    forget_decay,
    novelty,
    novelty_decay,
    L,
    power,
    maxtime
    ):

    config_filename = 'temp_config.ini'

    f = open(config_filename, 'w')

    f.write("""[Smoothing]
beta=10000
""")

    f.write('lambda=' + str(L) + '\n')
    f.write('power=' + str(power) + '\n')
    f.write("""epsilon=0.01
alpha=20

[Similarity]
simtype=COS
theta=0.7

[Features]
""")

    if dummy is False:
        f.write('dummy=false\n')
    else:
        f.write('dummy=true\n')

    if forget is False:
        f.write('forget=false\n')
        f.write('forget-decay=0\n')
    else:
        f.write('forget=true\n')
        f.write('forget-decay=' + str(forget_decay) + '\n')

    if novelty is False:
        f.write('novelty=false\n')
        f.write('novelty-decay=0\n')
    else:
        f.write('novelty=true\n')
        f.write('novelty-decay=' + str(novelty_decay) + '\n')

    f.write('assoc-type=SUM\n')
    f.write('category=false\n')

    f.write("""semantic-network=false
hub-type=hub-freq-degree
hub-num=75

[Statistics]
stats=true
context-stats=false
familiarity-smoothing=0.01
familiarity-measure=COUNT
age-of-exposure-norm=100
tag1=ALL
word-props-name=word_props_
time-props-name=time_props_
context-props-name=context_props

[Other]
minfreq=0
record-iterations=-1
""")

    f.write('maxtime=' + str(maxtime) + '\n')

    f.write("""maxlearned=-1\n""")

    f.write('remove-singleton-utterances=false\n')

    return config_filename

def delete_config_file():
    os.remove('temp_config.ini')

def setup_experiments(experiment_condition):

    # forgetting
    if experiment_condition['forget'] is not False:
        forget = True
        forget_decay = experiment_condition['forget']
    else:
        forget = False
        forget_decay = 0

    # novelty
    if experiment_condition['novelty'] is not False:
        novelty = True
        novelty_decay = experiment_condition['novelty']
    else:
        novelty = False
        novelty_decay = 0

    config_path = write_config_file(
        dummy=experiment_condition['dummy'],
        forget=forget,
        forget_decay=forget_decay,
        novelty=novelty,
        novelty_decay=novelty_decay,
        L=experiment_condition['lambda'],
        power=experiment_condition['power'],
        maxtime=experiment_condition['maxtime']
    )

    novel_word = experiment_condition['novel-word']
    corpus_without_word_path = corpora[novel_word]

    # create and teach the learner
    learner_config = learnconfig.LearnerConfig(config_path)
    stopwords = []
    learner = learn.Learner(lexname, learner_config, stopwords)
    learner.process_corpus(corpus_without_word_path, outdir)

    # generate the test utterance and scene
    if fix_familiar_objects is False:
        familiar_objects =  get_random_sample_words(corpus_without_word_path, experiment_condition['number-familiar-objects'], maxtime=experiment_condition['maxtime'])
    else:
        familiar_objects = fixed_familiar_objects

    # check that the learner knows all the familiar objects
    for obj in familiar_objects:
        assert obj in learner._wordsp.all_words(0)

    # generate the scene
    scene_generator = [novel_word] + familiar_objects
    print scene_generator
    scene_representation = generate_scene(scene_generator, experiment_condition, probabilistic=experiment_condition['probabilistic-features'])
    utterance = [novel_word]

    # check that the novel word has not yet been encountered
    for feature in scene_representation:
        assert learner._learned_lexicon.prob(novel_word, feature) < 0.001

    return learner, utterance, scene_generator, scene_representation, novel_word

# not yet implemented
    ## target type
    #if experiment_condition['target-type'] == 'familiar':
    #    get_random_sample_words(corpus_path, n):
    #elif experiment_condition['target-type'] == 'novel':
    #'number-familiar-objects' :         [n for n in range(3)], # number of familiar objects in test array
    #'number-novel-objects' :            [n for n in range(3)] # number of novel objects in test array


def run_experiments(learner, experiment_condition, utterance, scene_generator, scene_representation):

    # learn the test utterance and then calculate the referent probabilities

    if check_probs:

        print ''
        print '-------Before processing the scene--------'
        print ''

        for obj in scene_generator:
            print 'word:', obj
            for v, f in problex.meaning(obj).sorted_features():
                print 'Feature:', f, '\t\tProb:', learner._learned_lexicon.prob(obj, f)

    print utterance, scene_representation
    learner.process_pair(utterance, scene_representation, outdir, experiment_condition['category-learner'])

    if check_probs:

        print ''
        print '-------After processing the scene--------'
        print ''

        for word in utterance:
            print 'Word:', word
            for feature in scene_representation:
                print 'Feature:', feature, '\t\tProb:', learner._learned_lexicon.prob(word, feature)
        print '------------------------------------------------------------------'
        for obj in scene_generator:
            print 'word:', obj
            for v, f in problex.meaning(obj).sorted_features():
                print 'Feature:', f, '\t\tProb:', learner._learned_lexicon.prob(obj, f)

    return calculate_referent_probability(learner, utterance, scene_generator, scene_representation, forget=experiment_condition['forget'])

def clean_up():
    for word in parameter_values['novel-word']:
        os.remove(corpus_path + '_without_' + word)


if __name__ == '__main__':
    experimental_results = {}
    try:
        for experiment_condition in experiment_conditions:

            # create a label for this trial
            r = dict(experiment_condition)
            del r['number-familiar-objects']
            del r['number-novel-objects']
            del r['category-learner']
            del r['dummy']
            del r['novel-word']
            del r['maxtime']
            del r['power']
            del r['lambda']
            novel_word = experiment_condition['novel-word']
            power = experiment_condition['power']
            l = experiment_condition['lambda']

            d=""
            for i in r:
                a=i
                b=r[i]
                c=i+" : "+str(r[i])
                d=d+c+', '

            c='lambda'+" : "+str(l)
            d=d+c+', '
            c='power'+" : "+str(power)
            d=d+c+', '
            c='novel-word'+" : "+novel_word
            d=d+c

            experimental_results[d] = {}
            experimental_results[d]['referent probability of novel target'] = []
            experimental_results[d]['referent probability of familiar target'] = []

            for i in range(num_iterations):
                learner, utterance, scene_generator, scene_representation, novel_word = setup_experiments(experiment_condition)

                print '=================================================================='
                print 'Experiment condition:'
                pprint.pprint(experiment_condition)
                print ''
                print 'Utterance:', utterance
                print 'Scene:', scene_representation
                print ''

                referent_probabilities = run_experiments(learner, experiment_condition, utterance, scene_generator, scene_representation)

                #if len(utterance) > 0:
                    #print 'The familiar word(s) is /are:'
                    #for word in utterance:
                        #print '\t', word, '(with frequency', learner._wordsp.frequency(word), ')'

                print '\tReferent prob:'

                for referent in scene_generator:
                    for novel_word in utterance:
                        print '\t\tNovel word:\t\t', novel_word, '\t\tReferent:\t\t', referent, '\t\tReferent probability: ', referent_probabilities[(novel_word, referent)]

                #print '\t\tNovel word:\t\t', novel_word, '\t\tReferent probability: ', referent_probabilities[novel_word]
                #utterance.remove(novel_word)
                #for word in utterance:
                    #print '\t\tFamiliar word:\t\t', word, '\t\tReferent probability: ', referent_probabilities[word]


                experimental_results[d]['referent probability of novel target'].append(referent_probabilities[(utterance[0], scene_generator[0])])
                experimental_results[d]['referent probability of familiar target'].append(referent_probabilities[(utterance[0], scene_generator[1])])

            experimental_results[d]['mean referent probability of novel target across trials'] = \
                np.mean(experimental_results[d]['referent probability of novel target'])
            experimental_results[d]['mean referent probability of familiar target across trials'] = \
                np.mean(experimental_results[d]['referent probability of familiar target'])

    except KeyboardInterrupt:
        pass

    with open('results3.pkl', 'wb') as f:
        pickle.dump(experimental_results, f)

    clean_up()
