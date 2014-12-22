"""
referent_probability_experiments.py

This module conducts the referent probability experiments as described in the
2010 CogSci paper.
Possible parameter values are expected to be defined in experiments.cfg in
Python's ConfigParser format.

"""
import os
import re

import pickle
import random
import pprint
import numpy as np
import itertools
from scipy.stats import rankdata
from collections import Counter

from experiment import Experiment

import input
import learn
import learnconfig

verbose = False
check_probs = False

# read the input lexicon file and store lexemes in a probabilistic lexicon in memory
M = 10000 # based on script generate_dev_data.sh
lexname = 'norm_prob_lexicon_cs.all'
problex = input.read_gold_lexicon(lexname, M)

# hash the path of the modified corpora to avoid regenerating
corpora = {}

class NovelReferentExperiment(Experiment):
    """
    A condition (certain setting of parameter values) of the novel referent
    experiment.

    """

    def setup(self, params, rep):
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
        config_filename = 'temp_config_'
        config_filename += '_'.join([str(param) + ':' + str(value) \
                for (param, value) in params.items() \
                if param not in ['path', 'name', 'fix-familiar-objects',
                    'fix-novel-words', 'corpus-path', 'inference-type',
                    'repetitions', 'familiar-object', 'iterations',
                    'maxtime', 'prop-novel-features', 'n-features',
                    'rep', 'novel-word']])
        config_filename += '_rep:' + str(rep)
        config_filename += '.ini'

        self.config_path = write_config_file(
            config_filename,
            dummy=params['dummy'],
            forget=self.forget,
            forget_decay=self.forget_decay,
            novelty=self.novelty,
            novelty_decay=self.novelty_decay,
            L=params['lambda'],
            power=params['power'],
            maxtime=params['maxtime']
        )

        searching_for_words = True
        while searching_for_words:

            if params['fix-words'] is False:
                #self.novel_word =  get_random_sample_words(params['corpus-path'], 1)
                #self.familiar_objects =  get_random_sample_words(corpus_without_word_path, 1, maxtime=params['maxtime'])

                # TODO: temporary hack
                if params['n-features'] == 5:
                    self.novel_word, self.familiar_objects = five_feature_condition.pop(random.randrange(len(five_feature_condition)))
                elif params['n-features'] == 10:
                    self.novel_word, self.familiar_objects = ten_feature_condition.pop(random.randrange(len(ten_feature_condition)))
                else:
                    raise NotImplementedError

            else:
                self.novel_word = params['novel-word']
                self.familiar_objects = params['familiar-object']
                searching_for_words = False

            # create the modified corpus if it does not already exist
            try:
                corpus_without_word_path = corpora[self.novel_word]
            except KeyError:
                corpora[self.novel_word] = create_corpus_without_word(self.novel_word, params['corpus-path'])
                corpus_without_word_path = corpora[self.novel_word]

            # create and teach the learner
            learner_config = learnconfig.LearnerConfig(self.config_path)
            stopwords = []
            self.learner = learn.Learner(lexname, learner_config, stopwords)
            self.learner.process_corpus(corpus_without_word_path, params['path'])

            self.familiar_features = set(
                list(itertools.chain.from_iterable(
                    [self.learner._learned_lexicon.seen_features(word) \
                    for word in self.learner._wordsp.all_words(0)])
                )
            )

            # make sure the learner has seen the familiar word at least three times
            if self.familiar_objects in self.learner._wordsp.all_words(3):
                searching_for_words = False
            else:
                self.learner.reset() # redo the learning with a word that exists in the corpus

        self.utterance = [self.novel_word]
        self.familiar_objects = [self.familiar_objects]

        return True

    def iterate(self, params, rep, n):
        """
        Conduct a trial of this experiment condition.

        """
        self.scene = []
        self.referent_to_features_map = {}

        # GENERATE THE SCENE
        # generate representations of familiar objects
        for obj in self.familiar_objects:

            values_and_features = problex.meaning(obj).sorted_features()

            if params['probabilistic'] is True:
                # add FAMILIAR features to scene with probability equal to gold-standard meaning
                probs = np.array([np.float128(value) for (value, feature) in values_and_features if feature in self.familiar_features])
                probs /= probs.sum()

                try:
                    features = list(np.random.choice(
                        a=[feature for (value, feature) in values_and_features if feature in self.familiar_features],
                        size=params['n-features'],
                        replace=False,  # sample features without replacement
                        p=probs
                    ))
                except ValueError: # there were not enough familiar features to sample from
                    return None

            else:
                # grab the top n familiar features
                features = [feature for value, feature in values_and_features \
                    if feature in self.familiar_features][:params['n-features']]
                if len(features) < params['n-features']:
                    return None

            self.referent_to_features_map[obj] = features
            self.scene += features

        # generate representation of the novel word
        self.referent_to_features_map[self.novel_word] = []

        number_novel_features_for_novel_word = int(params['n-features'] * params['prop-novel-features'])
        number_overlapping_features_for_novel_word = int(params['n-features'] * params['prop-overlapping-features'])
        number_familiar_features_for_novel_word = params['n-features'] - number_novel_features_for_novel_word

        values_and_features = problex.meaning(self.novel_word).sorted_features()

        if params['probabilistic'] is True:
            # add familiar features to scene with probability equal to gold-standard meaning
            probs = np.array([np.float128(value) for (value, feature) in values_and_features if feature in self.familiar_features])
            probs /= probs.sum()

            try:
                features = np.random.choice(
                    a=[feature for (value, feature) in values_and_features if feature in self.familiar_features],
                    size=number_familiar_features_for_novel_word,
                    replace=False,  # sample features without replacement
                    p=probs
                )
                features = list(features)
            except ValueError: # there were not enough familiar features to sample from
                return None

        else:
            features = [feature for value, feature in values_and_features \
                if feature in self.familiar_features][:number_familiar_features_for_novel_word]
            if len(features) < number_familiar_features_for_novel_word:
                return None

        if len(set(features).intersection(self.scene)) < number_overlapping_features_for_novel_word:
            candidates = list(set([f for v, f in problex.meaning(self.novel_word).sorted_features()]).intersection(
                [f for v, f in problex.meaning(self.familiar_objects[0]).sorted_features()]))
            while len(set(features).intersection(self.scene)) < number_overlapping_features_for_novel_word:
                try:
                    candidate = candidates.pop()
                    if candidate in self.learner._learned_lexicon.meaning(self.familiar_objects[0]).seen_features():
                        if candidate not in features:
                            features.pop(-1)
                            features = [candidate] + features
                        if candidate not in self.scene:
                            to_remove = self.scene.pop(-1) # assumption: if not probabilistic, remove weakest meaning from familiar object
                            self.scene += [candidate]
                            self.referent_to_features_map[self.familiar_objects[0]].remove(to_remove)
                            self.referent_to_features_map[self.familiar_objects[0]] += [candidate]
                except IndexError:
                    return None
        #elif len(set(features).intersection(self.scene)) > number_overlapping_features_for_novel_word:
        #    print 'Objects too similar.'
        #    return None

        self.referent_to_features_map[self.novel_word] += features
        self.scene += features

        novel_features = ['novelft#' + str(i+1) for i in range(number_novel_features_for_novel_word)]
        self.scene += novel_features
        self.referent_to_features_map[self.novel_word] += novel_features

        self.overlapping_features = [k for k,v in Counter(self.scene).items() if v>1]
        self.scene = list(set(self.scene)) # do not have duplicate features in the scene

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
        self.learner.process_pair(self.utterance, self.scene, params['path'], False)

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

        #TODO find better way to organise
        mul_referent_probs = self.calculate_referent_probability(inference_type='MUL')
        sum_referent_probs = self.calculate_referent_probability(inference_type='SUM')

        return_dict = { 'novel referent (SUM)' : sum_referent_probs[(self.novel_word, self.novel_word)],
                 'novel referent (MUL)' : mul_referent_probs[(self.novel_word, self.novel_word)],
                 'familiar referent (SUM)' : sum_referent_probs[(self.novel_word, self.familiar_objects[0])],
                 'familiar referent (MUL)' : mul_referent_probs[(self.novel_word, self.familiar_objects[0])],
                 'ratio (SUM)' : np.divide(sum_referent_probs[(self.novel_word, self.novel_word)],
                           sum_referent_probs[(self.novel_word, self.familiar_objects[0])]),
                 'ratio (PROD)' : np.divide(mul_referent_probs[(self.novel_word, self.novel_word)],
                           mul_referent_probs[(self.novel_word, self.familiar_objects[0])]),
                 'number of novel features' : int(params['n-features'] * params['prop-novel-features']),
                 'scene' : self.scene,
                 'overlapping feature(s)' : self.overlapping_features,
                 'novel word' : self.novel_word,
                 'familiar object' : self.familiar_objects[0]
            }

        return return_dict

    def finalize(self, params, rep):
        """
        Final tasks to perform after completing this experiment condition.

        """
        #os.remove(self.config_path)
        print('Finished experiment.')
        pass

    def calculate_referent_probability(self, inference_type):
        """
        Calculate the referent probability of each object in the scene and
        store it in this NovelReferentExperiment's referent probability
        dictionary.

        """
        self.feature_prob = {} # dictionary of { feature : p (f) = \sum_{w' \in W} p ( f | w' ) * p( w' ) }
        self.joint_prob = {} # dictionary of { (word, feature) : p( f | w ) * p( w ) = p( f, w ) }

        for feature in self.scene:

            self.feature_prob[feature] = np.float128(0)

            for word in self.learner._wordsp.all_words(0): # get all words the learner has seen (with min frequency of 0)

                # p (f) = \sum_{w'} [ p ( f | w' ) * p( w' ) ] = \sum_{w'} [ p ( f, w' ) ]
                self.feature_prob[feature] += np.multiply(
                    np.float128(self.learner._learned_lexicon.prob(word, feature)),
                    np.float128(self.learner._wordsp.frequency(word)))

                if word in self.utterance and feature in self.scene:
                    # hack to ensure meaning probability is updated for encountered words (when forget is True)
                    self.learner.acquisition_score(word)

                    self.joint_prob[(word, feature)] = \
                            np.multiply(
                                np.float128(self.learner._learned_lexicon.prob(word, feature)),
                                np.float128(self.learner._wordsp.frequency(word)))

        # calculate the referent probabilities
        self.referent_prob = {} # dictionary of { word : p ( w | F ) }

        for spoken_word in self.utterance:
            for referent in self.referent_to_features_map:

                if inference_type == 'MUL':

                    self.referent_prob[(spoken_word, referent)] = 1

                    for feature in self.referent_to_features_map[referent]:
                        # p ( w | F ) = \prod_f [ p ( w | f ) ] = \prod_f [ p ( f, w ) / p ( f ) ]
                        self.referent_prob[(spoken_word, referent)] *= \
                            np.divide(self.joint_prob[(spoken_word, feature)],
                            self.feature_prob[feature])

                elif inference_type == 'SUM':

                    self.referent_prob[(spoken_word, referent)] = np.divide(
                        # p ( w | F ) =  [ p ( w | f_1 + ... + f_n ) ] = [ \sum_f p ( f, w ) ] / \sum_f p ( f )
                        np.sum([self.joint_prob[(spoken_word, feature)] for feature in self.referent_to_features_map[referent]]),
                        np.sum([self.feature_prob[feature] for feature in self.referent_to_features_map[referent]]))

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

def get_random_sample_words(corpus_path, n=None, weighted=False, maxtime=None, min_freq=1):
    """
    Return a random sample of nouns from the corpus located in the filesystem
    at corpus_path, such the nouns have occcurred at least min_freq times.
    If n is an integer, return n words; otherwise return all nouns.
    If weighted is True, weight the probability of sampling a word by its
    frequency in the corpus.
    If maxtime is an integer, read only maxtime sentences from the input
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

    counter = Counter(word_list)
    word_list = [ word for word, count in counter.items() if count >= min_freq ]

    if n is None:
        return word_list

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
    config_filename,
    dummy,
    forget,
    forget_decay,
    novelty,
    novelty_decay,
    L,
    power,
    maxtime
    ):

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

def choose_words_by_features(lex, num_overlap_features, n=False, top=False, ranked=True, words=None):
    """
    Return a list of n (word1, word2) tuples which are such that word1 and
    word2 have exactly num_overlap features in common, according to the
    Lexicon lex.
    If top is an integer, the above check is restricted to first top
    features associated with each word, sorted by meaning probability in lex.
    If ranked is True, the overlapping features must appear in the same
    position in each words' sorted meaning probability list (accounting for
    the possibility of ties between features).
    If words is a list, use that list of words to choose from.

    """
    assert top is False or num_overlap_features <= top

    if top is not False:
        num = top
    else:
        num = -1

    if words is None:
        words = lex.words()

    feature_to_words_map = {}

    candidate_pool = [word for word in words if \
        (word[-1] == 'N') and
        ((top is not False and len(lex.meaning(word).sorted_features()) >= top) or \
        (top is False and len(lex.meaning(word).sorted_features()) >= num_overlap_features))]

    for word in candidate_pool:
        # hack to include the last item in the list
        v_f = lex.meaning(word).sorted_features()[:num]
        if num == -1:
            v_f.append(lex.meaning(word).sorted_features()[-1])

        for (value, feature) in v_f:
            try:
                feature_to_words_map[feature].append(word)
            except KeyError:
                feature_to_words_map[feature] = []
                feature_to_words_map[feature].append(word)

    words = []

    while len(candidate_pool) > 0:
        word = candidate_pool.pop()

        # hack to include the last item in the list
        v_f = lex.meaning(word).sorted_features()[:num]
        if num == -1:
            v_f.append(lex.meaning(word).sorted_features()[-1])

        # rank the features
        f_r = zip(
            [feature for value, feature in v_f],
            rankdata([value for value, feature in v_f], method='dense')
        )

        # find all words that have at least one feature in common (of top features)
        candidate_matches = list(itertools.chain.from_iterable(
            [feature_to_words_map[feature] for (feature, rank) in f_r]))

        # now restrict to the number of features we need
        candidate_matches = [cand for cand in set(candidate_matches) if \
            candidate_matches.count(cand) == num_overlap_features]

        # make sure feature rankings are aligned if applicable
        if ranked is True:
            checked = []
            for match in candidate_matches:

                cand_v_f = lex.meaning(match).sorted_features()[:num]
                if num == -1:
                    cand_v_f.append(lex.meaning(match).sorted_features()[-1])

                # rank the features
                cand_f_r = zip(
                    [feature for value, feature in cand_v_f],
                    rankdata([value for value, feature in cand_v_f], method='dense')
                )

                add = True
                for feature, rank in f_r:
                    if feature not in [f for f, r in cand_f_r] or \
                        rank == dict(cand_f_r)[feature]:
                        pass
                    else:
                        add = False
                if add is True:
                    checked.append(match)

            candidate_matches = checked

        words.extend([(word, cand) for cand in candidate_matches])

    choices = np.array(words)
    if n is not False:
        idx = np.random.choice(len(words),size=n)
    else:
        idx = np.random.choice(len(words),size=len(choices))

    return choices[idx]

def clean_up():
    for word in corpora:
        os.remove(corpora[word])

def main():

    # generate the familiar and novel targets
    words = get_random_sample_words('input_wn_fu_cs_scaled_categ.dev', maxtime=10000, min_freq=3)

    five_feature_condition = list(choose_words_by_features(problex, 1, words=words, top=5))
    #ten_feature_condition = list(choose_words_by_features(problex, 2, words=words, top=10))

    experiment = NovelReferentExperiment()
    experiment.start()

    #with open('results.pkl', 'wb') as f:
        #pickle.dump(experiment, f)

    #clean_up()

# generate the familiar and novel targets
words = get_random_sample_words('input_wn_fu_cs_scaled_categ.dev', maxtime=10000, min_freq=3)
five_feature_condition = list(choose_words_by_features(problex, 1, words=words, top=5))
ten_feature_condition = list(choose_words_by_features(problex, 2, words=words, top=10))

if __name__ == '__main__':
    main()
