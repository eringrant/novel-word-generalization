import copy
import logging
import math
import numpy
import os
import sys

#import input
import wmmapping
"""
learn.py


"""

class Learner:
    """
    Encapsulate all learning and model updating functionality.

    """

    def __init__(self,
        gamma_sup, gamma_basic, gamma_sub, gamma_instance,
        k_sup, k_basic, k_sub, k_instance,
                 modified_gamma=True, flat_hierarchy=False):
        """

        """
        self._gamma_sup = gamma_sup
        self._gamma_basic = gamma_basic
        self._gamma_sub = gamma_sub
        self._gamma_instance = gamma_instance
        self._k_sup = k_sup
        self._k_basic = k_basic
        self._k_sub = k_sub
        self._k_instance = k_instance
        self._modified_gamma = modified_gamma
        self._flat_hierarchy = flat_hierarchy

        self._learned_lexicon = wmmapping.Lexicon([],
                self._gamma_sup,
                self._gamma_basic,
                self._gamma_sub,
                self._gamma_instance,
                self._k_sup,
                self._k_basic,
                self._k_sub,
                self._k_instance,
            self._modified_gamma, self._flat_hierarchy)

    def gamma(self, word, feature):
        return self._learner_lexicon.gamma(word, feature)

    def k(self):
        return self._k

    def learned_lexicon(self):
        """ Return a copy of the learned Lexicon. """
        return copy.deepcopy(self._learned_lexicon)

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
                denom += self._learned_lexicon.prob(word, feature, p=False)

            # Calculate alignment of each word
            for word in words:

                # alignment(w|f) = p(f|w) / normalization
                alignment = self._learned_lexicon.prob(word, feature, p=False) / denom

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
        # add the features to the hierarchy in the correct order
        for word in words:
            self._learned_lexicon.add_features_to_hierarchy(word, features)

        # calculate the alignment probabilities and update the associations for
        # all word-feature pairs
        self.calculate_alignments(words, features, outdir)

    def generalisation_prob(self, word, scene, fixed_levels=False):
        """
        Calculate the probability of this learner to generalise word to scene.
        Assume that scene is a set of features, in hierarchical order, from
        highest superordinate level to lowest subordinate level.

        """

        # hack to equalise the number of levels in the feature hierarchy
        if fixed_levels:
            l = len(scene)
            for i in range(self._learned_lexicon._max_depth - l):
                scene.append('dummy')

        # add features to the hierarchy so that novel features are in the
        # correct position
        # doesn't change meaning probability
        self._learned_lexicon.add_features_to_hierarchy(word, scene)

        gen_prob = 1.

        for feature in scene:

            prob = self._learned_lexicon.prob(word, feature, p=True)
            gen_prob *= prob

        return gen_prob

    def process_corpus(self, corpus_path, outdir, corpus=None):
        """
        Process the corpus file located at corpus_path. The file at corpus_path
        should contains sentences and their meanings. If a Corpus corpus is
        presented, the corpus_path is ignored and the corpus provided from is
        read instead.

        """
        close_corpus = False
        if corpus is None:
            if not os.path.exists(corpus_path):
                logging.error("Error -- Corpus does not exist : " + corpus_path)
                sys.exit(2)
            corpus = input.Corpus(corpus_path)
            close_corpus = True;

        (words, features) = corpus.next_pair()

        while words != []:

            self.process_pair(words, features, outdir)

            (words, features) = corpus.next_pair()

        # End processing words-sentences pairs from corpus

        if close_corpus:
            corpus.close()

def cosine(k, meaning1, meaning2):
    """
    Calculate and return the similarity score using the Cosine method, comparing
    the probabilities within Meaning of first word and Meaning of second word as the vectors.
    beta is used as a smoothing factor.

    features contain all the features of the gold lexicon.
    """
    features = meaning1.seen_features() | meaning2.seen_features()

    meaning1_vec = numpy.zeros(len(features))
    meaning2_vec = numpy.zeros(len(features))

    # hash the level counts of the features
    level_counts = {}

    i = 0

    for feature in features:
        meaning1_vec[i] = meaning1.prob(feature)

        #meaning2_vec[i] = meaning2.prob(feature)

        # hack: set all but gold standard meaning probability features to zero
        if feature in meaning2.seen_features():
            meaning2_vec[i] = 1.0
        else:
            meaning2_vec[i] = 0.0

        i += 1

#        level = max(meaning1.get_level(feature), meaning2.get_level(feature))
#
#        try:
#            level_counts[level] += 1
#        except KeyError:
#            level_counts[level] = 0
#            level_counts[level] += 1
#
#    assert not -1 in level_counts
#
    cos = numpy.dot(meaning1_vec, meaning2_vec)
    x = numpy.dot(meaning1_vec, meaning1_vec)
    y = numpy.dot(meaning2_vec, meaning2_vec)
#
#    for level in level_counts:
#        cos += (k - level_counts[level]) * meaning1.unseen_prob_by_level(level) * meaning2.unseen_prob_by_level(level)
#        x += pow(meaning1.unseen_prob_by_level(level), 2) * (k - level_counts[level])
#        y += pow(meaning2.unseen_prob_by_level(level), 2) * (k - level_counts[level])
#
    x = math.sqrt(x)
    y = math.sqrt(y)

    return  cos / (x * y)
