#!/usr/bin/python

import os
import random
import itertools

import input
import learnconfig

# input corpus
corpus_path = 'input_wn_fu_cs_scaled_categ.dev'

# output directory
outdir = 'referent_probability_experiments_output'
if not os.path.exists(outdir):
    os.makedirs(outdir)

# gold-staandard lexicon
lexname = norm_prob_lexicon_cs.all

# number of iterations for each condition
num_iterations = 20

# number of random novel words to generate experimental conditions for
num_novel_word_conditions = 10

parameter_values = {
# config file parameters
    'dummy' :                           [True, False],
    'forget' :                          [False, 0], #TODO: what are the possible values here
    'novelty' :                         [False, 0], #TODO: what are the possible values here
    'remove-singleton-utterances' :     [True, False],
    'maxtime' :                         [n for n in range(500, 1100, 500)], # max num words; 100 to 1000
    'category' :                        [True, False],
# training parameters
    #'permute-input-sentences' :         [True, False],
# test parameters
    'target-type' :                     ['familiar', 'novel'],
    'number-familiar-objects' :         [n for n in range(3)], # number of familiar objects in test array
    'number-novel-objects' :            [n for n in range(3)] # number of novel objects in test array
}

# generate the novel word possibilities
parameter_values['novel-word'] =  get_random_sample_words(corpus_path, num_novel_word_conditions):

# create and hash the paths to the modified corpora
corpora = {}
for word in parameter_values['novel-word']:
    corpora[word] = create_corpus_without_word(word, corpus_path):

# cartesian product of all parameter settings
experiment_conditions = [dict(zip(parameter_values, x)) for x in itertools.product(*parameter_values.values())]

def calculate_referent_probability(learner, utterance, scene):

    # ensure we have a list of words and a list of features
    if isinstance(utterance, str):
        utterance = [ utterance ]
    if isinstance(scene, str):
        scene = [ scene ]

    # TODO: determine if the referent prob should be a joint distribution of features, conditional upon a word

    feature_prob = {} # dictionary of { feature : p (f) = \sum_{w' \in W} p ( f | w' ) * p( w' ) }
    joint_prob = {} # dictionary of { (word, feature) : p( f | w ) * p( w ) = p(f, w) }

    # TODO: change loop order
    for feature in scene:

        sum_word_freq = 0.0
        sum_assoc = 0.0

        for word in learner._vocab:

            # p ( f | w' ) * p( w' )
            sum_assoc += learner.association(word, feature) * learner._wordsp.frequency(word))

            # TODO: for novel words, following returns 0

            if word in utterance and feature in scene: #TODO: check formatting aligns
                joint_prob[(word, feature)] = learner.association(word, feature) * learner._wordsp.frequency(word)

            sum_word_freq += learner._wordsp.frequency(word)

        feature_prob[feature] = sum_assoc

    # TODO: joint probability for novel words

    # normalise the joint probabililties over total word frequency
    for (w_f, prob) in joint_prob.values():
        joint_prob[w_f] = prob / float(sum_word_freq)

    # TODO: finish writing

def write_config_file(
    dummy,
    forget,
    forget_decay,
    novelty,
    novelty_decay,
    singletons,
    maxtime,
    category
    ):

    config_filename = 'temp_config.ini'

    f = open(config_filename, 'w')

    f.write("""[Smoothing]
beta=10000
lambda=-1
power=1
alpha=20
epsilon=0.01

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
    f.write('category' + str(category) + '\n')

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

    if singletons is True:
        f.write('remove-singleton-utterances=true\n')
    else:
        f.write('remove-singleton-utterances=false\n')

    return config_filename

def delete_config_file():
    os.remove('temp_config.ini')

def get_random_sample_words(corpus_path, n):
    word_list = []

    corpus = input.Corpus(corpus_path)
    (words, features) = corpus.next_pair()

    for wd in words:
        word_list.append(wd)

    while words != []:

        if len(words) == 1:
            # Skip singleton utterances
            (words, features) = corpus.next_pair()
            continue

        for wd in words:
            word_list.append(wd)

        (words, features) = corpus.next_pair()

    return random.sample(set(word_list), num_novel_word_conditions)

def create_corpus_without_word(word, corpus_path):
    corpus_output_filename = corpus_path + '_without_' + word

    found_word = False
    at_SEM_REP = False

    fin = open( corpus_path )
    fout = open( corpus_output_filename , "w")

    for line in fin:
        if word in line:
            found_word = True
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

def setup_experiments(experiment_condition):

    # forgetting
    if experiment_condition['forget'] is not False:
        forget = True
        forget_decay = experiment_condition['forget']
    else:
        forget = False
        forget_decay = 0

    # novelt
    if experiment_condition['novelty'] is not False:
        novelty = True
        novelty_decay = experiment_condition['novelty']
    else:
        novelty = False
        novelty_decay = 0

    config_path = write_config_file(
        experiment_condition['dummy'],
        forget,
        forget_decay,
        novelty,
        novelty_decay,
        experiment_condition['remove-singleton-utterances'],
        experiment_condition['maxtime'],
        experiment_condition['category']
    )

    corpus_without_word_path = create_corpus_without_word(experiment_condition['novel-word'], corpus_path)

    # create and teach the learner
    learner_config = learnconfig.LearnerConfig(config_path)
    stopwords = []
    learner = learn.Learner(lexname, learner_config, stopwords)
    learner.process_corpus(corpus_without_word_path, outdir)

    # generate the test array

    # target type
    if experiment_condition['target-type'] == 'familiar':
        get_random_sample_words(corpus_path, n):

    elif experiment_condition['target-type'] == 'novel':

    'number-familiar-objects' :         [n for n in range(3)], # number of familiar objects in test array
    'number-novel-objects' :            [n for n in range(3)] # number of novel objects in test array


#def run_experiments(experiment_condition):


def clean_up():
    for word in parameter_values['novel-word']:
        os.remove(corpus_path + '_without_' + word)


if __name__ == '__main__':
    #for experiment_condition in experiment_conditions:
        #setup_experiments(experiment_condition)
        #run_experiments(experiment_condition)

