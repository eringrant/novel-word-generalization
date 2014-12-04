"""
referent_probability_experiments.py

This module conducts the referent probability experiments as described in the
2010 CogSci paper.
Possible parameter values are expected to be defined in experiments.cfg in
Python's ConfigParser format.

"""
import os
import re

import random
import pprint
import numpy as np
from expsuite import PyExperimentSuite

import input
import learn
import learnconfig

verbose = True
check_probs = True

# read the input lexicon file and store lexemes in a probabilistic lexicon in memory
M = 10000 # based on script generate_dev_data.sh
lexname = 'norm_prob_lexicon_cs.all'
problex = input.read_gold_lexicon(lexname, M)

# hash the path of the modified corpora to avoid regenerating
corpora = {}

class NovelReferentExperiment(PyExperimentSuite):
    """
    A  condition (certain setting of parameter values) of the novel referent
    experiment.

    """

    def reset(self, params, rep):
        """ Setup the experiment. """

        # forgetting
        if params['forget'] is not False:
            self.forget = True
            self.forget_decay = params['forget']
        else:
            self.forget = False
            self.forget_decay = 0

        # novelty
        if params['novelty'] is not False:
            self.novelty = True
            self.novelty_decay = params['novelty']
        else:
            self.novelty = False
            self.novelty_decay = 0

        # create temporary config file
        config_path = write_config_file(
            dummy=params['dummy'],
            forget=self.forget,
            forget_decay=self.forget_decay,
            novelty=self.novelty,
            novelty_decay=self.novelty_decay,
            L=params['lambda'],
            power=params['power'],
            maxtime=params['maxtime']
        )

        if params['fix-novel-words'] is False:
            self.novel_word =  get_random_sample_words(params['corpus-path'], 1)
        else:
            self.novel_word = params['novel-word']

        # create the modified corpus if it does not already exist
        try:
            corpus_without_word_path = corpora[self.novel_word]
        except KeyError:
            corpora[self.novel_word] = create_corpus_without_word(self.novel_word, params['corpus-path'])
            corpus_without_word_path = corpora[self.novel_word]

        # create and teach the learner
        learner_config = learnconfig.LearnerConfig(config_path)
        stopwords = []
        self.learner = learn.Learner(lexname, learner_config, stopwords)
        self.learner.process_corpus(corpus_without_word_path, params['path'])

        # generate the test utterance and scene
        if params['fix-familiar-objects'] is False:
            #self.familiar_objects =  get_random_sample_words(corpus_without_word_path, params['n-familiar-objects'], maxtime=params['maxtime'])
            self.familiar_objects =  get_random_sample_words(corpus_without_word_path, 1, maxtime=params['maxtime'])
        else:
            self.familiar_objects = params['familiar-object']
        # TODO
        self.familiar_objects = [self.familiar_objects]
        print self.familiar_objects

        # check that the learner knows all the familiar objects
        for obj in self.familiar_objects:
            assert obj in self.learner._wordsp.all_words(0)

        self.utterance = [self.novel_word]

    def iterate(self, params, rep, n):
        """
        Conduct a trial of this experiment condition.

        """
        self.scene = []
        self.referent_to_features_map = {}

        # generate the scene
        # generate representations of familiar objects
        for obj in self.familiar_objects:

            values_and_features = problex.meaning(obj).sorted_features()

            # add features to scene with probability equal to gold-standard meaning
            features = np.random.choice(
                [vf[1] for vf in values_and_features],
                params['n-features'],
                p=[float(vf[0]) for vf in values_and_features]
            )


            self.referent_to_features_map[obj] = features
            self.scene += features

        # generate representation of the novel word
        self.referent_to_features_map[self.novel_word] = []

        number_novel_features_for_novel_word = int(params['n-features'] * params['prop-novel-features'])
        number_familiar_features_for_novel_word = params['n-features'] - number_novel_features_for_novel_word

        novel_features = ['novelft#' + str(i+1) for i in range(number_novel_features_for_novel_word)]
        self.scene += novel_features
        self.referent_to_features_map[self.novel_word] = novel_features

        values_and_features = problex.meaning(self.novel_word).sorted_features()

        # add features to scene with probability equal to gold-standard meaning
        features = np.random.choice(
            [vf[1] for vf in values_and_features],
            params['n-features'],
            p=[float(vf[0]) for vf in values_and_features]
        )

        self.referent_to_features_map[self.novel_word] += features
        self.scene += features

        if check_probs:

            print 'Utterance:', self.utterance, 'Scene:', self.scene

            print ''
            print '-------Before processing the scene--------'
            print ''

            for obj in self.familiar_objects + [self.novel_word]:
                print 'word:', obj
                f_check = [f for (v, f) in problex.meaning(obj).sorted_features()]
                f_check += self.scene
                f_check = set(f_check)
                for f in f_check:
                    print 'Feature:', f, '\t\tProb:', self.learner._learned_lexicon.prob(obj, f)
                print ''

        #self.learner.process_pair(self.utterance, self.scene, outdir, params['category-learner'])
        self.learner.process_pair(self.utterance, self.scene, params['path']), False

        if check_probs:

            print ''
            print '-------After processing the scene--------'
            print ''

            for obj in self.familiar_objects + [self.novel_word]:
                print 'word:', obj
                f_check = [f for (v, f) in problex.meaning(obj).sorted_features()]
                f_check += self.scene
                f_check = set(f_check)
                for f in f_check:
                    print 'Feature:', f, '\t\tProb:', self.learner._learned_lexicon.prob(obj, f)
                print ''

        return self.calculate_referent_probability(params['inference-type'])

    def calculate_referent_probability(self, inference_type):
        """
        Calculate the referent probability of each object in the scene and
        store it in this NovelReferentExperiment's referent probability
        dictionary.

        """
        self.feature_prob = {} # dictionary of { feature : p (f) = \sum_{w' \in W} p ( f | w' ) * p( w' ) }
        self.joint_prob = {} # dictionary of { (word, feature) : p( f | w ) * p( w ) = p( f, w ) }

        for feature in self.scene:

            self.feature_prob[feature] = 0.0

            for word in self.learner._wordsp.all_words(0): # get all words the learner has seen (with min frequency of 0)

                # p (f) = \sum_{w'} [ p ( f | w' ) * p( w' ) ] = \sum_{w'} [ p ( f, w' ) ]
                self.feature_prob[feature] += self.learner._learned_lexicon.prob(word, feature) * self.learner._wordsp.frequency(word)

                if word in self.utterance and feature in self.scene:
                    # hack to ensure meaning probability is updated for encountered words (when forget is True)
                    self.learner.acquisition_score(word)

                    self.joint_prob[(word, feature)] = self.learner._learned_lexicon.prob(word, feature) * self.learner._wordsp.frequency(word)

        # calculate the referent probabilities
        self.referent_prob = {} # dictionary of { word : p ( w | F ) }

        for spoken_word in self.utterance:
            for referent in self.referent_to_features_map:

                if inference_type == 'MUL':

                    self.referent_prob[(spoken_word, referent)] = 1

                    for feature in self.referent_to_features_map[referent]:
                        # p ( w | F ) = \prod_f [ p ( w | f ) ] = \prod_f [ p ( f, w ) / p ( f ) ]
                        self.referent_prob[(spoken_word, referent)] *= (self.joint_prob[(spoken_word, feature)] / self.feature_prob[feature])

                elif inference_type == 'SUM':

                    self.referent_prob[(spoken_word, referent)] = 0

                    for feature in self.referent_to_features_map[referent]:
                        # p ( w | F ) = \sum_f [ p ( w | f ) ] = \sum_f [ p ( f, w ) / p ( f ) ]
                        self.referent_prob[(spoken_word, referent)] *= (self.joint_prob[(spoken_word, feature)] / self.feature_prob[feature])

                else:
                    raise NotImplementedError

        if verbose:
            print '------------------------------------------------------------------'
            print ''
            print 'Joint probability of word and feature:'
            pprint.pprint(self.joint_prob)
            print ''
            print 'Marginal probability of feature:'
            pprint.pprint(self.feature_prob)
            print ''
            print 'Referent probability:'
            pprint.pprint(self.referent_prob)
            print ''

        return self.referent_prob

def get_random_sample_words(corpus_path, n, weighted=False, maxtime=None):
    """
    Return a random sample of n words from the corpus located in the filesystem
    at corpus_path.
    If weighted is True, weight the probability of sampling a word by its
    frequency in the corpus.
    If maxtime is an integer, read only maxtime sentences fro m the input
    corpus.

    """

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

    if weighted:
        return random.sample(word_list, n)
    else:
        return random.sample(set(word_list), n)

def create_corpus_without_word(word, corpus_path):
    """
    Create a corpus identical to the corpus at corpus_path, but with all
    sentences in which word occurs removed, and write it to file.
    Return the filename of the newly created corpus.

    """
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

def clean_up():
    for word in corpora:
        os.remove(corpora[word])
    os.remove('temp_config.ini')

def usage():
    print "usage:"
    print "  main.py -c (--corpus) -l (--lexicon) -o (--output) -v (--verbose)"
    print ""
    print "  --corpus:   input corpus"
    print "  --lexicon:  gold-standard lexicon"
    print "  --output:   output directory"
    print "  --verbose:  for detailed output"
    print "  --help:     prints this usage"
    print ""

def main():
    try:
        options_list = ["help", "corpus=", "lexicon=", "inputdir=", "output="]
        opts, args = getopt.getopt(sys.argv[1:], "hc:l:i:o:", options_list)
    except getopt.error, msg:
        print msg
        usage()
        sys.exit(2)

    if len(opts) < 4:
        usage()
        sys.exit(0)

    corpus_path = ""
    stop = ""
    for o, a in opts:
        if o in ("-h", "--help"):
            usage()
            sys.exit(0)
        if o in ("-c", "--corpus"):
            corpus_path = a
        if o in ("-l", "--lexicon"):
            lexname = a
        if o in ("-o", "--output"):
            outdir = a
        if o in ("-v", "--verbose"):
            verbose = True

    if not os.path.exists(outdir):
        os.makedirs(outdir)

if __name__ == '__main__':
    experiment = NovelReferentExperiment()
    experiment.start()
