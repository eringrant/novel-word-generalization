from __future__ import division


import copy
import numpy as np
import pprint


"""
wmmapping.py

Data structures for mapping words to meanings.
"""


class UndefinedFeatureError(Exception):
    """
    Defines an exception that occurs when a feature whose feature group is not
    known is presented to the model.
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class UndefinedParameterError(Exception):
    """
    Defines an exception that occurs when an invalid parameter setting is
    applied.
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class Alignments(object):
    """
    TODO
    """
    def __init__(self):
        self._alignments = []

    def __contains__(self, time):
        """Return True if there is an alignment at time."""
        return int(time) in [t for (t, align) in self._alignments]

    def __deepcopy__(self, memo):
        c = self.__class__
        result = c.__new__(c)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))

        return result

    def __repr__(self):
        return "Alignments: " + str(self._alignments)

    def add_alignment(self, time, alignment):
        self._alignments.append((int(time), alignment))

    def alignments(self):
        return self._alignments


class Feature(object):
    """A feature event, conditional upon a word.

    Members:
        association -- the association of the feature and the word
        name -- the name that uniquely identifies the feature
        alignments -- an Alignment object containing the alignments of feature
            to the word
    """
    def __init__(self, name):
        self._association = 0.0
        self._name = name
        self._alignments = Alignments()

    def __deepcopy__(self, memo):
        c = self.__class__
        result = c.__new__(c)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))

        return result

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self._name == other._name

    def __hash__(self):
        return hash(self._name)

    def __ne__(self, other):
        return not isinstance(other, self.__class__) or self._name != other

    def __repr__(self):
        return "Feature: " + str(self._name) + "; Association: " +\
            str(self._association)

    def association(self):
        return self._association

    def name(self):
        return self._name

    def update_association(self, alignment, novelty, decay, time):
        """Add alignment to this Feature's association.

        If decay is True, then update this Feature's association score to be a
        sum over decayed Alignments.

        If decay is False, add the bare alignment to this Feature's association
        score.
        """
        self._alignments.add_alignment(time, alignment)

        if decay:
            self._association =\
                np.sum([align / np.power(time - t + 1, (decay/align)) for
                        (t, align) in self._alignments.alignments()])
        else:
            self._association += alignment

        if novelty:
            raise NotImplementedError


class FeatureGroup(object):
    """A feature group, conditional upon a word.

    Members:
        TODO
    """

    def __init__(self, gamma, k, p, name=None):

        self._name = name

        self._gamma = gamma
        self._k = k
        self._p = p

        self._features = {}

    def __contains__(self, feature):
        """Check if feature is a member of this FeatureGroup."""
        return any([f == feature for f in self._features])

    def __deepcopy__(self, memo):
        c = self.__class__
        result = c.__new__(c)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))

        # Manually copy over the Feature dictionary, since the
        # Features are mutable objects that themselves need to be copied
        for feature, feature_object in result._features.items():
            result._features[feature] = copy.deepcopy(feature_object)

        return result

    def __eq__(self, other):
        if self._name is None or other._name is None:
            raise UndefinedParameterError("Comparison of feature groups with\
                                          no label.")
        return self._name == other._name

    def __len__(self):
        return len(self._features)

    def __repr__(self):
        prefix = "Feature group %s" % self._name if self._name is not None \
            else "Unnamed feature group"
        postfix = pprint.pformat(self._features.values())
        return prefix + " (gamma=%f, k=%f)" % (self._gamma, self._k) +\
            '\n' + postfix

    def add_feature(self, feature):
        """
        TODO
        """
        assert isinstance(feature, basestring)
        assert feature not in self._features
        self._features[feature] = Feature(feature)

    def association(self, feature):
        """
        TODO
        """
        return self._features[feature].association()

    def denom(self):
        """Return the denominator for this FeatureGroup."""
        return self.summed_association() + self.k() * self.gamma()

    def gamma(self):
        """Return the gamma parameter for this FeatureGroup."""
        num_types = len([f for f in self._features.values()
                         if f.association() > 0])
        num_types = max(num_types, 1)
        return self._gamma * (num_types ** self.p())

    def k(self):
        """Return the k parameter for this FeatureGroup."""
        return self._k

    def p(self):
        """Return the p parameter for this FeatureGroup."""
        return self._p

    def prob(self, feature):
        """Return the meaning probability of feature."""

        try:
            numer = self._features[feature].association()
            numer += self.gamma()

            return numer / self.denom()

        except KeyError:
            raise UndefinedFeatureError

    def seen_features(self):
        """Return the set of all the features seen in this FeatureGroup."""
        return set(self._features.keys())

    def summed_association(self):
        """
        Return the association score summed across all features in this
        FeatureGroup.
        """
        return sum([feature.association() for feature in
                    self._features.values()])

    def unseen_prob(self):
        """
        TODO
        """
        return self.gamma() / self.denom()

    def update_association(self, feature, alignment, novelty, decay, time):
        """
        TODO
        """
        self._features[feature].update_association(alignment, novelty,
                                                   decay, time)


class Meaning(object):
    """Contains the probability of all feature events, conditional upon a word.

    Members:
        TODO
    """

    def __init__(
        self,
        gamma_sup, gamma_basic, gamma_sub, gamma_instance,
        k_sup, k_basic, k_sub, k_instance,
        p_sup, p_basic, p_sub, p_instance,
        feature_group_to_level_map,
        feature_to_feature_group_map,
        word=None
    ):
        self._word = word
        self._seen_features = []

        # (str) -> (str) maps
        self._feature_group_to_level_map = feature_group_to_level_map.copy()
        self._feature_to_feature_group_map = \
            feature_to_feature_group_map.copy()

        # Transform the mapping of features to group (str -> str) to be (str ->
        # FeatureGroup)
        self._feature_groups = {}
        for feature, feature_group in \
                self._feature_to_feature_group_map.items():

            try:
                # Hash to check if the FeatureGroup has already been created
                feature_group_object = self._feature_groups[feature_group]

            except KeyError:
                # Create the FeatureGroup object
                level = self._feature_group_to_level_map[feature_group]

                if level == 'superordinate':
                    gamma = gamma_sup
                    k = k_sup
                    p = p_sup
                elif level == 'basic-level':
                    gamma = gamma_basic
                    k = k_basic
                    p = p_basic
                elif level == 'subordinate':
                    gamma = gamma_sub
                    k = k_sub
                    p = p_sub
                elif level == 'instance':
                    gamma = gamma_instance
                    k = k_instance
                    p = p_instance
                else:
                    raise NotImplementedError

                feature_group_object = FeatureGroup(gamma, k, p,
                                                    name=feature_group)

                self._feature_groups[feature_group] = feature_group_object

            feature_group_object.add_feature(feature)
            self._feature_to_feature_group_map[feature] = feature_group_object

    def __deepcopy__(self, memo):
        c = self.__class__
        result = c.__new__(c)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))

        # Manually copy over the FeatureGroup dictionaries, since the
        # FeatureGroups are mutable object that themselves need to be copied
        for feature_group, feature_group_object in result._feature_groups.items():
            result._feature_groups[feature_group] = copy.deepcopy(feature_group_object)
        for feature, old_feature_group_object in result._feature_to_feature_group_map.items():
            result._feature_to_feature_group_map[feature] =\
                result._feature_groups[old_feature_group_object._name]

        return result

    def __repr__(self):
        """Format this meaning to print intelligibly."""
        return str(self._word) + '\n' +\
            pprint.pformat(self._feature_groups.values())

    def add_seen_features(self, features):
        """
        Add to the list of features encountered so far with the word associated
        with this Meaning.
        """
        self._seen_features.extend(features[:])
        self._seen_features = list(set(self._seen_features))  # no duplicates

    def denom(self, feature):
        """
        Return the denominator for the meaning probability calculation for
        feature in this Meaning.
        """
        return self.summed_association(feature) +\
            (self.k(feature) * self.gamma(feature))

    def feature_group(self, feature_group_name):
        """
        TODO
        """
        return self._feature_groups[feature_group_name]

    def feature_groups(self):
        """
        Return a list of the FeatureGroups that are contained within this
        Meaning.
        """
        return list(self._feature_groups.values())

    def gamma(self, feature):
        """Return the gamma parameter for feature in this Meaning."""
        feature_group = self._feature_to_feature_group_map[feature]
        return feature_group.gamma()

    def k(self, feature):
        """Return the k parameter for feature in this Meaning."""
        feature_group = self._feature_to_feature_group_map[feature]
        return feature_group.k()

    def prob(self, feature):
        """Return the probability of feature given this Meaning's word."""
        feature_group = self._feature_to_feature_group_map[feature]
        return feature_group.prob(feature)

    def seen_features(self):
        """
        Return a set of all features from all levels of the hierarchy, observed
        so far with this Meaning's word.
        """
        return set(self._seen_features)

    def summed_association(self, feature):
        """
        Return the association, summed across the FeatureGroup containing
        feature, for feature in this Meaning.
        """
        feature_group = self._feature_to_feature_group_map[feature]
        return feature_group.summed_association()

    def update_association(self, feature, alignment, novelty, decay, time):
        """
        Update the association between this Meaning's word and feature by
        adding alignment to the current association.
        """
        self._feature_to_feature_group_map[feature].\
            update_association(feature, alignment, novelty, decay, time)


class Lexicon(object):
    """
    A Lexicon object maps words to Meaning objects.

    Members:
        TODO
    """

    def __init__(
        self, words,
        gamma_sup, gamma_basic, gamma_sub, gamma_instance,
        k_sup, k_basic, k_sub, k_instance,
        p_sup, p_basic, p_sub, p_instance,
        feature_group_to_level_map,
        feature_to_feature_group_map,
    ):
        """
        TODO
        """
        self._gamma_sup = gamma_sup
        self._gamma_basic = gamma_basic
        self._gamma_sub = gamma_sub
        self._gamma_instance = gamma_instance

        self._k_sup = k_sup
        self._k_basic = k_basic
        self._k_sub = k_sub
        self._k_instance = k_instance

        self._p_sup = p_sup
        self._p_basic = p_basic
        self._p_sub = p_sub
        self._p_instance = p_instance

        self.feature_group_to_level_map = feature_group_to_level_map
        self.feature_to_feature_group_map = feature_to_feature_group_map

        self._word_meanings = {}
        for word in words:
            self.initialize_new_meaning(word)

    def initialize_new_meaning(self, word):
        """
        TODO
        """
        assert word not in self._word_meanings
        self._word_meanings[word] = Meaning(
            self._gamma_sup,
            self._gamma_basic,
            self._gamma_sub,
            self._gamma_instance,
            self._k_sup,
            self._k_basic,
            self._k_sub,
            self._k_instance,
            self._p_sup,
            self._p_basic,
            self._p_sub,
            self._p_instance,
            self.feature_group_to_level_map,
            self.feature_to_feature_group_map,
            word=word
        )
        return self._word_meanings[word]

    def add_seen_features(self, word, features):
        """
        Add features to the list of features encountered so far with word.
        """
        assert word in self._word_meanings
        self._word_meanings[word].add_seen_features(features)

    def gamma(self, word, feature):
        """
        TODO
        """
        if word not in self._word_meanings:
            self.initialize_new_meaning(word)
        self._word_meanings[word].gamma(feature)

    def k(self, word, feature):
        """
        TODO
        """
        if word not in self._word_meanings:
            self.initialize_new_meaning(word)
        self._word_meanings[word].k(feature)

    def meaning(self, word):
        """Return the Meaning object corresponding to word."""
        if word in self._word_meanings:
            return self._word_meanings[word]
        return self.initialize_new_meaning(word)

    def novelty(self, word):
        """
        TODO
        """
        raise NotImplementedError

    def prob(self, word, feature):
        """
        Return the probability of feature being part of the meaning of word.
        """
        if word not in self._word_meanings:
            self.initialize_new_meaning(word)
        return self._word_meanings[word].prob(feature)

    def seen_features(self, word):
        """Return the set of features encountered so far with word."""
        if word in self._word_meanings:
            return self._word_meanings[word].seen_features()
        return set()

    def update_association(self, word, feature, alignment, novelty, decay,
                           time):
        """
        Update association between word and feature by adding alignment to
        the current association.
        """
        if word not in self._word_meanings:
            self.initialize_new_meaning(word)
        self._word_meanings[word].update_association(feature, alignment,
                                                     novelty, decay, time)

    def words(self):
        """Return a set of all words in this Lexicon."""
        return set(self._word_meanings.keys())
