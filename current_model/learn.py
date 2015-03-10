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
import wgraph
"""
learn.py


"""

class Learner:
    """
    Encapsulate all learning and model updating functionality.

    """

    def __init__(self, config):
        """

        """
        if config is None:
            print "Initialization Error -- Config required"
            sys.exit(2)

        # Begin configuration of the learner based on config

        # Smoothing
        self._gamma = config.param_float("gamma")
        if self._gamma < 0:
            print "Config Error [gamma] Must be non-zero positive : "+str(self._gamma)
            sys.exit(2)

        self._k = config.param_float("k")
        if self._k < 0:
            print "Config Error [k] Must be non-zero positive : "+str(self._k)
            sys.exit(2)

        # End configuration based on config

        self._learned_lexicon = wmmapping.Lexicon([], self._gamma, self._k)
        self._aligns = wmmapping.Alignments(self._gamma)

        self._time = 0
        self._vocab = set()
        self._features = set()
        self._feature_frequencies = {}
        self._acquisition_scores = {}
        self._last_time = {}

    def gamma(self, word, feature):
        return self._learner_lexicon.gamma(word, feature)
        #return self.gamma_sub * t**2

    def k(self):
        return self._k
        #return self.gamma_sub * t**2

    def get_lambda(self):
        """ Return a lambda smoothing factor. """
        if self._lambda < 1 and self._lambda > 0:
            return self._lambda

        return 1.0 / (1 + self._time**self._power)

    def learned_lexicon(self):
        """ Return a copy of the learned Lexicon. """
        return copy.deepcopy(self._learned_lexicon)

    def update_meaning_prob(self, word):
        """
        Update the meaning probabilities of word in this learner's lexicon.
        This is done by calculating the association between this word and all
        encountered features - p(f|w) - then normalizing to produce a
        distribution.

        """
        for feature in self._features:

            sibling_features = self.sibling_features(word, feature)

            count = 0
            denom = 0

            for f in sibling_features:

                denom += self.association(word, f)
                count += 1

                denom += self.k() * self.gamma(word, feature)

                # TODO: recalculate unseen probability to be gamma / denom
                # for this feature group

                meaning_prob = (self.association(word, feature) +
                        self.gamma(word, feature)) / denom
                self._learned_lexicon.set_prob(word, feature, meaning_prob)

    def sibling_features(self, word, feature):
        return self._learned_lexicon.sibling_features(word, feature)

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
        self._learned_lexicon.add_features_to_hierarchy(features)

        # calculate the alignment probabilities and update the associations for
        # all word-feature pairs
        self.calculate_alignments(words, features)

        if self._stats_flag:
            t = self._time

            for word in words:
                # Update word statistics
                if not self._wordsp.has_word(word):
                    learned_c = self.learned_count(self._postags)
                    self._wordsp.add_word(word, t, learned_c)
                else:
                    self._wordsp.inc_frequency(word)
                self._wordsp.update_last_time(word, t)

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

            # Record statistics - average acquisition and similarity scores
            if self._stats_flag:
                self.record_statistics(corpus_path, words, novel_nouns,
                                       noun_count, outdir)
            # Record Context statistics
            if self._context_stats_flag:
                sims = self.calculate_similarity_scores(words)
                comps = self.calculate_comprehension_scores(words)
                self._contextsp.add_context(set(words), self._time, sims, comps)

            (words, features) = corpus.next_pair()

        # End processing words-sentences pairs from corpus

        if self._stats_flag:
            # Write statistics to files
            self._wordsp.write(corpus_path, outdir, str(self._time))
            self._timesp.write(corpus_path, outdir, str(self._time))
        if self._context_stats_flag:
            words = self._contextsp._words.keys()
            sims = self.calculate_similarity_scores(words)
            comps = self.calculate_comprehension_scores(words)
            for word in words:
                self._contextsp.add_similarity(word, sims[word])
                self._contextsp.add_comprehension(word, comps[word])
            # Write the statistics to files
            self._contextsp.write(corpus_path, outdir)

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

