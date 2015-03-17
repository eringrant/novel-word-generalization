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

    def __repr__(self):
        return "Feature: " + str(self._name) + "; Association: " + str(self._association)

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

    def __init__(self, feature_name, gamma, k, meaning):
        self._gamma = gamma
        self._k = k
        self._members = []
        self._feature = Feature(name=feature_name)
        self._meaning = meaning # the meaning hierarchy

    def __contains__(self, feature):
        """ Check if feature is a member of this FeatureGroup. """
        return any([f == feature for f in self._members])

    def __eq__(self, other):
        return self._feature == other

    def __ne__(self, other):
        return self._feature != other

    def __repr__(self):
        return "Feature group for: " + self._feature.__repr__() + ";\n\tMembers: " +  str(self._members)

    def add_feature(self, feature):
        fg = FeatureGroup(feature, self._gamma, self._k, self._meaning)
        self._members.append(fg)
        return fg

    # implementation for which gamma is a function of types across the level of
    # a hierarchy
    def gamma(self):
        raise Exception # shouldn't be called

    # implementation for which gamma is a function of types within a feature
    # group
    #def gamma(self):
    #    count = len([f for f in self._members if f.node_association() > 0])
    #    count = max(count, 1)
    #    return self._gamma * (count**2)

    def node_association(self):
        return self._feature.association()

    def update_node_association(self, alignment):
        return self._feature.update_association(alignment)

    # implementation for which gamma is a function of types across the level of
    # a hierarchy
    def prob(self, feature, gamma):
        if feature in self._members:
            denom = self._k * gamma
            denom += sum([f.node_association() for f in self._members])
            numer = find(lambda fg: fg == feature, self._members)
            numer = numer.node_association()
            numer += gamma
            return numer / denom

        else:
            return self.unseen_prob()

    # implementation for which gamma is a function of types within a feature
    # group
    #def prob(self, feature):
    #    if feature in self._members:
    #        denom = self._k * self.gamma()
    #        denom += sum([f.node_association() for f in self._members])
    #        numer = find(lambda fg: fg == feature, self._members)
    #        numer = numer.node_association()
    #        numer += self.gamma()
    #        return numer / denom

    #    else:
    #        return self.unseen_prob()

    # implementation for which gamma is a function of types across the level of
    # a hierarchy
    def unseen_prob(self, gamma):
        """
        Compute the unseen probability of this feature group using the
        associations stored in the Features .

        """
        denom = self._k * gamma
        denom += sum([f.node_association() for f in self._members])
        return gamma/denom

    # implementation for which gamma is a function of types within a feature
    # group
    #def unseen_prob(self, gamma):
    #    """
    #    Compute the unseen probability of this feature group using the
    #    associations stored in the Features .

    #    """
    #    denom = self._k * self.gamma()
    #    denom += sum([f.node_association() for f in self._members])
    #    return self.gamma()/denom

    def update_association(self, feature, alignment):
        to_update = find(lambda fg: fg == feature, self._members)
        to_update.update_node_association(alignment)


class Meaning:
    """
    Contains the probability of all feature events, conditional upon a word.

    Members:
        TODO

    """

    def __init__(self, gamma, k, word=None):
        """
        TODO

        """
        self._gamma = gamma
        self._k = k

        self._word = word
        self._seen_features = []

        # the root of the hierarchy
        self._root = FeatureGroup(None, gamma, k, self)

        # hash maps for computational efficiency
        self._feature_to_feature_group_map = {}
        self._level_to_feature_groups_map = {}
        self._feature_to_level_map = {}

    # TODO
    def __deepcopy__(self, memo):
        return Meaning(copy.deepcopy(self.name, memo))

    def __str__(self):
        """ Format this meaning to print intelligibly."""
        return str(self._root)

    def add_features_to_hierarchy(self, features):
        """
        Add features to the hierarchy, assuming that the features are  ordered
        from highest superordinate to lowest subordinate feature.

        """
        fg = self._root
        level = 0

        # add the features one-by-one to their corresponding level of the
        # hierarchy
        for feature in features:

            if not feature in fg:
                # this feature is novel (at this level of the hierarchy)
                new_fg = fg.add_feature(feature)
                self._feature_to_feature_group_map[feature] = fg

                try:
                    if fg not in self._level_to_feature_groups_map[level]:
                        self._level_to_feature_groups_map[level].append(fg)
                except KeyError:
                    self._level_to_feature_groups_map[level] = []
                    if fg not in self._level_to_feature_groups_map[level]:
                        self._level_to_feature_groups_map[level].append(fg)
                self._feature_to_level_map[feature] = level

            else:
                new_fg = find(lambda f: f == feature, fg._members)

            fg = new_fg
            level += 1

    # implementation for which gamma is a function of types across a specific
    # level of a hierarchy
    def gamma(self, feature):
        """
        Return the updated gamma parameter for feature in this Meaning.

        """
        level = self._feature_to_level_map[feature]
        fgs = self._level_to_feature_groups_map[level]
        count = 0
        for fg in fgs:
            count += len([f for f in fg._members if f.node_association() > 0])
        count = max(count, 1)
        return self._gamma * (count**2)

    # implementation for which gamma is a function of types within a feature
    # group
    #def gamma(self, feature):
    #    """
    #    Return the updated gamma parameter for feature in this Meaning.

    #    """
    #    return self._feature_to_feature_group_map[feature].gamma()

    # implementation for which gamma is a function of types across a specific
    # level of a hierarchy
    def prob(self, feature):
        """
        Return the probability of feature given this Meaning's word.

        """
        return self._feature_to_feature_group_map[feature].prob(feature,
                self.gamma(feature))

    # implementation for which gamma is a function of types within a feature
    # group
    #def prob(self, feature):
    #    """
    #    Return the probability of feature given this Meaning's word.

    #    """
    #    return self._feature_to_feature_group_map[feature].prob(feature)

    def seen_features(self):
        """
        Return a set of all features from all levels of the hierarchy, observed
        so far with this Meaning's word.

        """
        return set(self._seen_features)

    def update_association(self, feature, alignment):
        """ Update association between this Meaning's word and feature by adding
        alignment to the current association.

        """
        self._feature_to_feature_group_map[feature].update_association(feature,
            alignment)

class Lexicon:
    """
    A Lexicon object maps words to Meaning objects.

    Members:
        word_meanings -- dictionary mapping words to Meaning objects
        gamma -- the Dirichlet hyperparameter
        k -- the expected count

    """

    def __init__(self, words, gamma, k):
        """
        TODO

        """
        self._gamma = gamma
        self._k = k

        self._word_meanings = {}
        for word in words:
            self._word_meanings[w] = Meaning(gamma, k, word=word)

    def add_features_to_hierarchy(self, word, features):
        """
        Add features to the hierarchy for this word.

        Assume that the features are in hierarchical order, from highest
        superordinate level to lowest subordinate level.

        """
        if word not in self._word_meanings:
            self._word_meanings[word] = Meaning(self._gamma, self._k, word=word)
        self._word_meanings[word].add_features_to_hierarchy(features)

    def gamma(self, word, feature):
        """ Return the probability of feature being part of the meaning of word. """
        if word not in self._word_meanings:
            self._word_meanings[word] = Meaning(self._gamma, self._k, word=word)
        self._word_meanings[word].gamma(feature)

    # TODO: not implemented correctly
    def meaning(self, word):
        """ Return a copy of the Meaning object corresponding to word. """
        if word in self._word_meanings:
            return self._word_meanings[word]
        return Meaning(self._gamma, self._k, word=word)

    def prob(self, word, feature):
        """ Return the probability of feature being part of the meaning of word. """
        if word not in self._word_meanings:
            self._word_meanings[word] = Meaning(self._gamma, self._k, word=word)
        return self._word_meanings[word].prob(feature)

    def add_seen_features(self, word, features):
        """ Add to the list of features encountered so far with word. """
        assert word in self._word_meanings
        self._word_meanings[word]._seen_features.extend(features[:])

    def seen_features(self, word):
        """ Return a set of features encountered - so far - with word. """
        if word in self._word_meanings:
             self._word_meanings[word].seen_features()
        return set()

    def update_association(self, word, feature, alignment):
        """ Update association between word and feature by adding alignment to
        the current association.

        """
        if word not in self._word_meanings:
            self._word_meanings[word] = Meaning(self._gamma, self._k, word=word)
        self._word_meanings[word].update_association(feature, alignment)

    def words(self):
        """ Return a set of all words in this lexicon. """
        return set(self._word_meanings.keys())

def find(f, seq):
    """ Return first item in sequence where f(item) == True. """
    for item in seq:
        if f(item):
            return item
