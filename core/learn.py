import copy
import logging
import math
import numpy as np
import os
import sys

import wmmapping


class Learner:
    """Encapsulate all learning and model updating functionality."""

    def __init__(
        self,
        gamma_sup, gamma_basic, gamma_sub, gamma_instance,
        k_sup, k_basic, k_sub, k_instance,
        p_sup, p_basic, p_sub, p_instance,
        feature_group_to_level_map,
        feature_to_feature_group_map,
        novelty=False, decay=None
    ):

        self._learned_lexicon = wmmapping.Lexicon(
            [],
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
            feature_group_to_level_map,
            feature_to_feature_group_map,
        )

        # Time wrt the word-feature pairings processed
        self._time = 1

        # Novelty and decay
        self._decay = decay
        self._novelty = novelty


    def gamma(self, word, feature):
        return self._learned_lexicon.gamma(word, feature)

    def k(self, word, feature):
        return self._learned_lexicon.k(word, feature)

    def learned_lexicon(self):
        """Return a copy of the learned Lexicon."""
        return copy.deepcopy(self._learned_lexicon)

    def calculate_alignments(self, words, features):
        """Update the alignments between words and features.

        Update the alignments for each combination of word-feature pairs from
        the list words and the set features:

            alignment: P(a|u,f) = p(f|w) / sum(w' in words) p(f|w')
        """
        for feature in features:

            # Normalization term: sum(w' in words) p(f|w')
            denom = 0.0
            for word in words:
                denom += self._learned_lexicon.prob(word, feature)

            # Calculate alignment of each word
            for word in words:

                # alignment(w|f) = P(f|w) / normalization
                alignment = self._learned_lexicon.prob(word, feature) / denom

                # Novelty
                alignment *=\
                    self._learned_lexicon.novelty(word) if self._novelty else 1

                # assoc_t(f,w) = assoc_{t-1}(f,w) + P(a|u,f)
                self._learned_lexicon.update_association(word, feature,
                                                         alignment,
                                                         self._novelty,
                                                         self._decay,
                                                         self._time)

            # End alignment calculation for each word

        # End alignment calculation for each feature

        for word in words:

            # Add features to the list of seen features for each word
            self._learned_lexicon.add_seen_features(word, features)

    def process_pair(self, words, features, outdir, time_increment=True):
        """Process the pair words-features."""
        assert isinstance(words, list)
        assert isinstance(features, list)
        assert isinstance(outdir, str)

        self.calculate_alignments(words, features)
        if time_increment:
            self._time += 1

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
                logging.error("Error -- Corpus does not exist: " + corpus_path)
                sys.exit(2)
            corpus = input.Corpus(corpus_path)
            close_corpus = True

        (words, features) = corpus.next_pair()

        while words != []:

            self.process_pair(words, features, outdir)

            (words, features) = corpus.next_pair()

        # End processing words-sentences pairs from corpus

        if close_corpus:
            corpus.close()

    def generalization_prob(self, word, scene, metric='intersection'):
        """Return the probability of Learner to generalize word to scene."""

        if metric == 'intersection':

            gen_prob = 1.

            for feature in scene:
                prob = self._learned_lexicon.prob(word, feature)
                gen_prob *= prob

        elif metric in ['truncated-cosine-same-word',
                        'cosine-full-distribution-same-word',
                        'truncated-cosine-novel-word',
                        'cosine-full-distribution-novel-word']:

            learned_meaning = self._learned_lexicon.meaning(word)

            dummy_word = 'DUMMY_WORD_' + str(scene)

            if metric.endswith('same-word'):

                # Initialise the scene meaning to be the same as the learned
                # meaning of the target word
                scene_meaning = copy.deepcopy(learned_meaning)
                scene_meaning._word = dummy_word
                self._learned_lexicon._word_meanings[dummy_word] = scene_meaning

            # Learn the meaning of the scene as a 'dummy meaning'
            self.process_pair([dummy_word], scene, './', time_increment=False)
            scene_meaning = self._learned_lexicon.meaning(dummy_word)

            gen_prob = cosine(learned_meaning, scene_meaning,
                              full_distribution=True if
                              metric.startswith('cosine-full-distribution') else
                              False)

        elif metric in ['cosine-without-test-distribution']:

            scene_meaning_vec = np.zeros(len(scene))
            learned_meaning_vec = np.zeros(len(scene))

            for i, feature in enumerate(scene):
                scene_meaning_vec[i] = 1.
                learned_meaning_vec[i] = self._learned_lexicon.prob(word, feature)

            cos = np.dot(scene_meaning_vec, learned_meaning_vec)
            squared_norm_x = np.dot(scene_meaning_vec, scene_meaning_vec)
            squared_norm_y = np.dot(learned_meaning_vec, learned_meaning_vec)

            gen_prob = cos / (math.sqrt(squared_norm_x) * math.sqrt(squared_norm_y))

        else:
            raise NotImplementedError

        return gen_prob


def cosine(meaning1, meaning2, full_distribution=True):
    """Return the cosine similarity of meaning1 and meaning2."""

    cos = 0
    squared_norm_x = 0
    squared_norm_y = 0

    feature_group_names = set([fg._name for fg in meaning1.feature_groups()]) \
        | set([fg._name for fg in meaning2.feature_groups()])

    # Loop over the feature groups and find features seen between the two
    # Meanings
    for fgn in feature_group_names:

        feature_group_1 = meaning1.feature_group(fgn)
        feature_group_2 = meaning2.feature_group(fgn)

        features = feature_group_1.seen_features() | feature_group_2.seen_features()

        meaning1_vec = np.zeros(len(features))
        meaning2_vec = np.zeros(len(features))

        for i, feature in enumerate(features):
            meaning1_vec[i] = meaning1.prob(feature)
            meaning2_vec[i] = meaning2.prob(feature)

        cos += np.dot(meaning1_vec, meaning2_vec)
        squared_norm_x += np.dot(meaning1_vec, meaning1_vec)
        squared_norm_y += np.dot(meaning2_vec, meaning2_vec)

        # If we are using the full distribution, add features to the vectors
        if full_distribution:
            # Ensure the feature groups are compatible
            assert feature_group_1.k() == feature_group_2.k()
            k = feature_group_1.k()
            seen_count = len(features)

            cos += (k - seen_count) * feature_group_1.unseen_prob() *\
                feature_group_2.unseen_prob()

            squared_norm_x += pow(feature_group_1.unseen_prob(), 2) *\
                (k - seen_count)

            squared_norm_y += pow(feature_group_2.unseen_prob(), 2) *\
                (k - seen_count)

    return cos / (math.sqrt(squared_norm_x) * math.sqrt(squared_norm_y))
