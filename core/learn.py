import copy
import logging
import os
import sys

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
        p_sup, p_basic, p_sub, p_instance,
        feature_map
        ):
        """

        """
        self._learned_lexicon = wmmapping.Lexicon([],
            gamma_sup,
            gamma_basic,
            gamma_sub,
            gamma_instance,
            k_sup,
            k_basic,
            k_sub,
            k_instance,
            p_sup,
            p_basic,
            p_sub,
            p_instance,
            feature_map
        )

    def gamma(self, word, feature):
        return self._learned_lexicon.gamma(word, feature)

    def k(self, word, feature):
        return self._learned_lexicon.k(word, feature)

    def learned_lexicon(self):
        """ Return a copy of the learned Lexicon. """
        return copy.deepcopy(self._learned_lexicon)

    def calculate_alignments(self, words, features):
        """ Update the alignments between words and features.

        Update the alignments for each combination of word-feature pairs from
        the list words and set features.
        """
        # Alignment: P(a|u,f) = p(f|w) / sum(w' in words)p(f|w')

        for feature in features:

            # Normalization term: sum(w' in words) p(f|w')
            denom = 0.0
            for word in words:
                denom += self._learned_lexicon.prob(word, feature)

            # Calculate alignment of each word
            for word in words:

                # alignment(w|f) = p(f|w) / normalization
                alignment = self._learned_lexicon.prob(word, feature) / denom

                # assoc_t(f,w) = assoc_{t-1}(f,w) + P(a|u,f)
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
        self.calculate_alignments(words, features)

    def generalization_prob(self, word, scene):
        """ Return the probability of this learner to generalize word to scene. """

        gen_prob = 1.

        for feature in scene:

            prob = self._learned_lexicon.prob(word, feature)
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

