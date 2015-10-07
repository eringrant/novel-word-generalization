import copy
import pprint

"""
wmmapping.py

Data structures for mapping the words to meanings and word-feature alignments.
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


class Feature:
    """A feature event, conditional upon a word.

    Members:
        name -- the name that uniquely identifies the feature
        association -- the association of the feature and the word
    """
    def __init__(self, name):
        self._name = name
        self._association = 0.0

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

    def update_association(self, alignment):
        """Add alignment to this Feature's association."""
        self._association += alignment


class FeatureGroup:
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

    def __eq__(self, other):
        if self._name is None or other._name is None:
            raise UndefinedParameterError("Comparison of feature groups with no\
                                          label.")
        return self._name == other._name

    def __len__(self):
        return len(self._features)

    # TODO
    def __repr__(self):
        prefix = "Feature group %s" % self._name if self._name is not None \
            else "Unnamed feature group"
        postfix = pprint.pformat(self._features.values())
        return prefix + " (gamma=%f, k=%f)" % (self._gamma, self._k) +\
            '\n' + postfix

    def add_feature(self, feature):
        assert isinstance(feature, str)
        assert feature not in self._features
        self._features[feature] = Feature(feature)

    def association(self, feature):
        return self._features[feature].association()

    def update_association(self, feature, alignment):
        return self._features[feature].update_association(alignment)

    def denom(self):
        """Return the denominator for this FeatureGroup."""
        return self.summed_association() + self.k() * self.gamma()

    def gamma(self):
        """Return the gamma parameter for this FeatureGroup."""
        num_types = len([f for f in self._features.values() if f.association() > 0])
        num_types = max(num_types, 1)
        return self._gamma * (num_types ** self.p())

    def k(self):
        """Return the k parameter for this FeatureGroup."""
        return self._k

    def p(self):
        """Return the p parameter for this FeatureGroup."""
        return self._p

    def prob(self, feature):
        """
        TODO
        Mention that this gives the unseen prob when assoc=0
        """

        try:
            numer = self._features[feature].association()
            numer += self.gamma()

            return numer / self.denom()

        except KeyError:
            raise UndefinedFeatureError

    def seen_features(self):
        """Return a set of all features seen in this FeatureGroup."""
        return set(self._features.keys())

    def summed_association(self):
        """Return the summed association in this FeatureGroup."""
        return sum([feature.association() for feature in self._features.values()])

    def unseen_prob(self):
        """
        TODO
        """
        return self.gamma() / self.denom()


class Meaning:
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
        # The mapping of features to group (str -> str)
        self._feature_to_feature_group_map = feature_to_feature_group_map.copy()
        self._feature_group_to_level_map = feature_group_to_level_map.copy()

        self._word = word
        self._seen_features = []

        # Transform the mapping of features to group (str -> str) to be (str ->
        # FeatureGroup)
        self._feature_groups = {}
        for feature, feature_group in self._feature_to_feature_group_map.items():

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

                feature_group_object = FeatureGroup(gamma, k, p, name=feature_group)

                self._feature_groups[feature_group] = feature_group_object

            feature_group_object.add_feature(feature)
            self._feature_to_feature_group_map[feature] = feature_group_object

    # TODO
    def __deepcopy__(self, memo):
        return Meaning(copy.deepcopy(self.name, memo))

    # TODO
    def __str__(self):
        """ Format this meaning to print intelligibly."""
        return str(self._word) + '\n' + pprint.pformat(self._feature_groups.values())

    def add_seen_features(self, features):
        """
        Add to the list of features encountered so far with the word associated
        with this Meaning.
        """
        self._seen_features.extend(features[:])
        self._seen_features = list(set(self._seen_features))  # remove duplicates

    def k(self, feature):
        """Return the k parameter for feature in this Meaning."""
        feature_group = self._feature_to_feature_group_map[feature]
        return feature_group.k()

    def gamma(self, feature):
        """Return the gamma parameter for feature in this Meaning."""
        feature_group = self._feature_to_feature_group_map[feature]
        return feature_group.gamma()

    def summed_association(self, feature):
        """
        Return the association, summed across the FeatureGroup containing
        feature, for feature in this Meaning.
        """
        feature_group = self._feature_to_feature_group_map[feature]
        return feature_group.summed_association()

    def denom(self, feature):
        """
        Return the denominator for the meaning probability calculation for
        feature in this Meaning.
        """
        return self.summed_association(feature) + (self.k(feature) * self.gamma(feature))

    def prob(self, feature):
        """Return the probability of feature given this Meaning's word."""
        feature_group = self._feature_to_feature_group_map[feature]
        return feature_group.prob(feature)

    def feature_groups(self):
        """
        Return a list of the FeatureGroups that are contained within this
        Meaning.
        """
        return list(self._feature_groups.values())

    def feature_group(self, feature_group_name):
        """
        TODO
        """
        return self._feature_groups[feature_group_name]

    def seen_features(self):
        """
        Return a set of all features from all levels of the hierarchy, observed
        so far with this Meaning's word.
        """
        return set(self._seen_features)

    def update_association(self, feature, alignment):
        """
        Update the association between this Meaning's word and feature by
        adding alignment to the current association.
        """
        self._feature_to_feature_group_map[feature].update_association(feature,
                                                                       alignment)


class Lexicon:
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

    def gamma(self, word, feature):
        if word not in self._word_meanings:
            self.initialize_new_meaning(word)
        self._word_meanings[word].gamma(feature)

    def k(self, word, feature):
        if word not in self._word_meanings:
            self.initialize_new_meaning(word)
        self._word_meanings[word].k(feature)

    def meaning(self, word):
        """Return a copy of the Meaning object corresponding to word."""
        if word in self._word_meanings:
            return self._word_meanings[word]
        return self.initialize_new_meaning(word)

    def prob(self, word, feature):
        """Return the probability of feature being part of the meaning of word."""
        if word not in self._word_meanings:
            self.initialize_new_meaning(word)
        return self._word_meanings[word].prob(feature)

    def add_seen_features(self, word, features):
        """Add features to the list of features encountered so far with word."""
        assert word in self._word_meanings
        self._word_meanings[word].add_seen_features(features)

    def seen_features(self, word):
        """Return the set of features encountered so far with word."""
        if word in self._word_meanings:
            return self._word_meanings[word].seen_features()
        return set()

    def update_association(self, word, feature, alignment):
        """
        Update association between word and feature by adding alignment to
        the current association.
        """
        if word not in self._word_meanings:
            self.initialize_new_meaning(word)
        self._word_meanings[word].update_association(feature, alignment)

    def words(self):
        """Return a set of all words in this Lexicon."""
        return set(self._word_meanings.keys())
