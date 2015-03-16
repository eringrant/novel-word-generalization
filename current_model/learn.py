import copy
import math
import numpy
import re
import sys
import os
import constants as CONST
from category import *
import input
import wmmapping
import statistics
import evaluate
"""
learn.py


"""

class Learner:
    """
    Encapsulate all learning and model updating functionality.

    """

    def __init__(self, gamma, k):
        """

        """
        # Smoothing
        self._gamma = gamma
        self._k = k

        self._learned_lexicon = wmmapping.Lexicon([], self._gamma, self._k)
        self._aligns = wmmapping.Alignments(self._gamma)

    def gamma(self, word, feature):
        return self._learner_lexicon.gamma(word, feature)

    def k(self):
        return self._k

    def learned_lexicon(self):
        """ Return a copy of the learned Lexicon. """
        return copy.deepcopy(self._learned_lexicon)

    def association(self, word, feature):
        """
        Return the association score between word and feature.

        """
        return self._aligns.sum_alignments(word, feature)

    def calculate_alignments(self, words, features, outdir):
        """
        Update the alignments for each combination of word-feature pairs from
        the list words and set features.

        """
        # alignment(w|f) = p(f|w) / sum(w' in words)p(f|w')

        for feature in features:

            #import pdb; pdb.set_trace()

            # Normalization term, sum(w' in words) p(f|w')
            denom = 0.0

            # Calculate the normalization terms
            for word in words:
                denom += self._learned_lexicon.prob(word, feature)

            # Calculate alignment of each word
            for word in words:

                # alignment(w|f) = p(f|w) / normalization
                alignment = self._learned_lexicon.prob(word, feature) / denom

                # assoc_t(f, w) = assoc_{t-1}(f, w) + P(a | u, f)
                self._learned_lexicon.update_association(word, feature,
                        alignment)

            # End alignment calculation for each word

        # End alignment calculation for each feature

        for word in words:

            # Add features to the list of seen features for each word
            self._learned_lexicon.add_seen_features(word, features)

    def process_pair(self, words, features, outdir):
        """
        Process the pair words-features, two lists of words and features,
        respectively, to be learned from.

        Assume that features is in hierarchical order, from highest superordinate
        level to lowest subordinate level.

        """
        # Time calculated w.r.t words-features pairings being processed
        self._time += 1

        # Add current features to the set of all seen features
        for feature in features:
            self._features.add(feature)
            try:
                self._feature_frequencies[feature] += 1
            except KeyError:
                self._feature_frequencies[feature] = 0
                self._feature_frequencies[feature] += 1

        # add the features to the hierarchy in the correct order
        for word in words:
            self._learned_lexicon.add_features_to_hierarchy(word, features)

        # calculate the alignment probabilities and update the associations for
        # all word-feature pairs
        self.calculate_alignments(words, features, outdir)

    def generalisation_prob(self, word, scene):

        self._learned_lexicon.add_features_to_hierarchy(word, scene)

        gen_prob = 1.

        for feature in scene:

            prob = self._learned_lexicon.prob(word, feature)
            print "\t\tFeature:", feature, "\tProb:", prob
            gen_prob *= prob

        return gen_prob

    def process_corpus(self, corpus_path, outdir, corpus=None):
        """
        Process the corpus file located at corpus_path, saving any gathered
        statistics to the directory outdir. The file at corpus_path should
        contains sentences and their meanings. If a Corpus corpus is presented,
        the corpus_path is ignored and the corpus provided from is read instead.
        Return the number of words learned and the number of time steps it
        required.

        """
        close_corpus = False
        if corpus is None:
            if not os.path.exists(corpus_path):
                print "Error -- Corpus does not exist : " + corpus_path
                sys.exit(2)
            corpus = input.Corpus(corpus_path)
            close_corpus = True;

        (words, features) = corpus.next_pair()

        learned = 0 # Number of words learned
        while words != []:

            if self._maxtime > 0 and self._time >= self._maxtime:
                break

            self.process_pair(words, features, outdir)

            learned = len(list(self._vocab))

            if self._maxlearned > 0 and learned > self._maxlearned:
                break

            if self._time % 1000 == 0:
                print self._time

            (words, features) = corpus.next_pair()

        # End processing words-sentences pairs from corpus

        if close_corpus:
            corpus.close()

        return learned, self._time

    def record_statistics(self, corpus, words, novel_nouns, noun_count, outdir):
        """
        Record statistics in this learner's timesp and wordsp regarding word
        types learned based on the learner's postag list. Also records statistics
        on novelty based on the noun words in novel_nouns.

        """
        # Write statistics information ever record_itrs iterations
        if self._record_itrs > 0 and self._time % self._record_itrs == 0:
            self._wordsp.write(corpus, outdir, str(self._time))
            self._timesp.write(corpus, outdir, str(self._time))

        # Dictionary to store statistics
        avg_acq = {}

        #BM begin novelty ==============================================
        avg_acq_nn = self.avg_acquisition(novel_nouns, CONST.N)

        if noun_count >= 2 and len(novel_nouns) >= 1:
            avg_acq[CONST.NOV_N_MIN1] = avg_acq_nn

        if noun_count >= 2 and len(novel_nouns) >= 2:
            avg_acq[CONST.NOV_N_MIN2] = avg_acq_nn
        #BM end novelty ================================================

        all_words = self._wordsp.all_words(self._minfreq)
        all_learned = list(self._vocab)

        # Average acquisition score for all pos tags (words) that are learned
        avg_acq[CONST.LRN] = self.avg_acquisition(all_learned, CONST.ALL)

        # Record statistics for words based on the configured pos tags
        if CONST.ALL in self._postags:
            avg_acq[CONST.ALL]  = self.avg_acquisition(all_words, CONST.ALL)
        if CONST.N in self._postags or CONST.ALL in self._postags:
            avg_acq[CONST.N] = self.avg_acquisition(all_words, CONST.N)
        if CONST.V in self._postags or CONST.ALL in self._postags:
            avg_acq[CONST.V] = self.avg_acquisition(all_words, CONST.V)
        if CONST.OTH in self._postags or CONST.ALL in self._postags:
            avg_acq[CONST.OTH] = self.avg_acquisition(all_words, CONST.OTH)

        heard = self.heard_count(self._minfreq, self._postags)
        learned = self.learned_count(self._postags)

        # Record all information at this time step
        self._timesp.add_time(self._time, heard, learned, avg_acq)

