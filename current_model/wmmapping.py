import copy
import math
import pprint

"""
wmmapping.py

Data-structures for mapping the words to meanings and word-feature alignments.

"""

class Feature:
    """

    A feature event, conditional upon a word.
    The word is implicitly represented here.

    Members:
        name -- the name that uniquely identifies the feature
        association -- the association of the feature and the word

    """

    def __init__(self, name, prob=None):
        self._name = name
        self._association = 0.0

    def __eq__(self, other):
        return self._name == other

    def __ne__(self, other):
        return self._name != other

    def name(self):
        return self._name

    def association(self):
        return self._association

    def update_association(self, alignment):
        """ Add the alignment to association. """
        self._association += alignment

class FeatureGroup:
    """

    A feature group, conditional upon a word;
    takes the form of a node in a tree.

    Members:
        gamma -- the Dirichlet hyperparametre
        k -- the expected number of types in the Dirichlet
        members -- the members of this FeatureGroup
        feature -- the feature that is directly superordinate to the members of
            this FeatureGroup

    """

    def __init__(self, feature_name, gamma, k):
        self._gamma = gamma
        self._k = k
        self._members = []
        self._feature = Feature(name=feature_name)

    def __contains__(self, feature):
        """ Check if feature is a member of this FeatureGroup. """
        return any([f == feature for f in self._members])

    def __eq__(self, other):
        return self._feature == other

    def __ne__(self, other):
        return self._feature != other

    def prob(self, feature):
        if feature in self._members:
            denom = self._k * self._gamma
            denom += sum([f.node_association() for f in self._members])
            return

        else:
            return self.unseen_prob()

    def node_association(self):
        return self._feature.association()

    def unseen_prob(self):
        """
        Compute the unseen probability of this feature group using the
        associations stored in the Features .

        """
        denom = self._k * self._gamma
        denom += sum([f.node_association() for f in self._members])
        return self._gamma/denom

    def add_feature(self, feature):
        fg = FeatureGroup(feature, self._gamma, self._k)
        self._children.append(fg)
        return fg

    def prob_feature_pairs(self):
        #TODO
        pass

    def sibling_features(self, feature):
        if feature in self._features:
            return set(self._features.keys())
        elif not self._features: # no features in this group
            return None
        else:
            pass
            # TODO: what was i gonna do here

    def update_association(self, feature, alignment):
        self._features[feature].update_association(alignment)


class Meaning:
    """
    Contains the probability of feature events, conditional upon a word.

    Members:

    """

    def __init__(self, gamma, k, word=None):
        """
        Initialise the hierarchy.

        """
        self._word = word
        self._seen_features = []

        self._root = FeatureGroup(gamma, k)

        # hash the features by name to the feature group that contains them
        self._feature_to_feature_group_map = {}

        self._gamma = gamma
        self._k = k

    def __deepcopy__(self, memo):
        # TODO
        return Meaning(copy.deepcopy(self.name, memo))

    def add_features_to_hierarchy(self, features):
        """
        Add features to the hierarchy, assuming that the features are  ordered
        from superordinate to subordinate feature.

        """
        fg = self._root
        for feature in features:
            if not feature in parent_feature_group:
                fg = parent_feature_group.add_feature(feature)
                self._features[feature] = parent_feature_group
                self._feature_to_feature_group_map[feature] = fg
            fg = self._feature_to_feature_group_map[feature]

    def prob(self, feature):
        """
        Return the probability of feature given this Meaning's word.

        """
        return self._feature_to_feature_group_map[feature].prob(feature)

    def set_prob(self, feature, prob):
        """
        Set the probability of feature given this Meaning's word to prob.

        """
        return self._root.set_prob(feature, prob)

    def seen_features(self):
        """
        Return a set of all features from all levels of the hierarchy, observed
        so far with this Meaning's word.

        """
        return set(self._seen_features)

    def sibling_features(self, feature):
        if feature not in self._seen_features:
            return set()
        return self._root.sibling_features(feature)

    def sorted_prob_feature_pairs(self):
        """
        Return a list, sorted by probability, of each (prob, feature) seen so
        far.

        """
        items = self._root.prob_feature_pairs()
        ranked = [ [v[1],v[0]] for v in items ]
        ranked.sort(reverse=True)
        return ranked

    def update_association(self, feature, alignment):
        """ Update association between this Meaning's word and feature by adding
        alignment to the current association.

        """
        self._feature_to_feature_group_map[feature].update_association(feature,
                alignment)

    def __str__(self):
        """ Format this meaning to print intelligibly."""
        # TODO
        pass


class Lexicon:
    """
    A Lexicon object maps words to Meaning objects.

    Members:
        word_meanings -- dictionary mapping words to Meaning objects
        gamma --
        k --

    """

    def __init__(self, words, gamma, k):
        """
        Create an empty Lexicon of words, such that each word in words has a
        Meaning with unseen probability 1.0/beta. See Meaning docstring.

        """
        self._gamma = gamma
        self._k = k

        self._word_meanings = {}
        for word in words:
            self._word_meanings[w] = Meaning(gamma, k, word=word)

    def add_features_to_hierarchy(self, word, features):
        """
        Add features to the hierarchy for this word.

        """
        if word not in self._word_meanings:
            self._word_meanings[word] = meaning(self._gamma, self._k, word=word)
        self._word_meanings[word].add_features_to_hierarchy(features)

    def gamma(self, word, feature):
        """ Return the probability of feature being part of the meaning of word. """
        if word not in self._word_meanings:
            self._word_meanings[word] = meaning(self._gamma, self._k, word=word)
        self._word_meanings[word].gamma(feature)

    def words(self):
        """ Return a set of all words in this lexicon. """
        return set(self._word_meanings.keys())

    def meaning(self, word):
        """ Return a copy of the Meaning object corresponding to word. """
        if word in self._word_meanings:
            return copy.deepcopy(self._word_meanings[word])
        return Meaning(self._gamma, self._k, word=word)

    def add_seen_features(self, word, features):
        """ Add to the list of features encountered so far with word. """
        assert word in self._word_meanings
        self._word_meanings[word]._seen_features.extend(features[:])

    def seen_features(self, word):
        """ Return a set of features encountered - so far - with word. """
        if word in self._word_meanings:
            return self._word_meanings[word].seen_features()
        return set()

    def prob(self, word, feature):
        """ Return the probability of feature being part of the meaning of word. """
        if word not in self._word_meanings:
            self._word_meanings[word] = Meaning(self._gamma, self._k, word=word)
        self._word_meanings[word].prob(feature)

    def set_prob(self, word, feature, prob):
        """
        Set the probability of feature being part of the meaning of word to
        prob.

        """
        if word not in self._word_meanings:
            self._word_meanings[word] = Meaning(self._gamma, self._k, word=word)
        self._word_meanings[word].set_prob(feature, prob)

    def sibling_features(self, word, feature):
        if word not in self._word_meanings:
            self._word_meanings[word] = Meaning(self._gamma, self._k, word=word)
        self._word_meanings[word].sibling_features(feature)

    def update_association(self, word, feature, alignment):
        """ Update association between word and feature by adding alignment to
        the current association.

        """
        if word not in self._word_meanings:
            self._word_meanings[word] = meaning(self._gamma, self._k, word=word)
        self._word_meanings[word].update_association(feature)


class Alignments:
    """
    Store calculated alignment probabilities for each word-feature pair at each
    time step.

    Members:
        probs -- Dictionary where keys are "word feature" and values are lists of
            the form [sum_aligns, {time:alignment_value, time:alignment_value ...}]
            where sum_aligns is the sum alignment value for this word-feature
            pair of all alignment_value's so far.

            sum_aligns is the association score:
                assoc_t(w,f) = assoc_(t-1)(w,f) + alignment_t(w|f)
            where t represents the time step.

            alignment_t(w|f) is exactly the entry probs[w + " " + f][1][t]

        unseen -- #BM write

    """

    def __init__(self, alpha):
        """ Create an empty Alignments object with smoothing value 1.0/alpha """
        self._probs = {}
        # Allow no smoothing
        if alpha == 0:
            self._unseen = 0.0
        elif alpha < 0:
            self._unseen = 0.0
        else:
            self._unseen = 1.0 / alpha

    def sum_alignments(self, word, feature):
        """
        Return the sum alignment probability for the pair "word feature". If the
        entry does not exist, create one.

        """
        wf = word + " " + feature
        if wf not in self._probs:
            self.create_entry(word, feature)
        return self._probs[wf][0]

    def alignments(self, word, feature):
        """
        Return a dictionary of time : alignment_value pairs for the pair
        "word feature". If the entry does not exist, create one.

         """
        wf = word + " " + feature
        if wf not in self._probs:
            self.create_entry(word, feature)
        return self._probs[wf][1]

    def add_alignment(self, word, feature, time, alignment):
        """
        Add (time, alignment) to the probabilities for word-feature pair and
        update the sum alignment probability with alignment.

        """
        wf = word + " " + feature
        if wf not in self._probs:
            self.create_entry(word, feature)
        alignments = self._probs[wf]
        alignments[1][time] = alignment
        alignments[0] += alignment

    def add_multiplicative_alignment(self, word, feature, time, alignment):
        """
        Add (time, alignment) to the probabilities for word-feature pair and
        update the sum alignment probability with alignment.

        """
        wf = word + " " + feature
        if wf not in self._probs:
            self.create_entry(word, feature)
        alignments = self._probs[wf]
        alignments[1][time] = alignment
        alignments[0] = alignments[0]**2 + alignment

    def add_decay_sum(self, word, feature, time, alignment, decay):
        """
        Calculate the association, using an alternative forgetting sum, for the
        pair word-feature, having alignment alignment at time time. decay is a
        constant decay factor.
        This is calculated as:

        assoc_{time}(word,feature) =

            assoc_{time'}(word,feature)
           ---------------------------------------------------  + alignment
            (time - time')**(decay/assoc_{time'}(word,feature))

        Where time' is the last time that the association between word and
        feature was calculated.

        """
        wf = word + " " + feature
        if wf not in self._probs:
            self.create_entry(word, feature)
        alignments = self._probs[wf]

        # Association at time t'
        time_pr_assoc = alignments[0]
        # Last time word-feature association was calculated
        last_time = max(alignments.keys())

        assoc_decay = decay / time_pr_assoc
        alignments[0] = time_pr_assoc / math.pow(time - last_time + 1, assoc_decay)
        alignments[0] += alignment
        alignments[1][time] = alignment # To keep track of time t'

    def create_entry(self, word, feature):
        """
        Create an empty alignment entry in this data structure for
        "word feature".

        """
        wf = word + " " + feature
        self._probs[wf] = []
        self._probs[wf].append(0.0)
        self._probs[wf].append({})
