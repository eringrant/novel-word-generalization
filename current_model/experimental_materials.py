from collections import defaultdict
import pprint
import copy

import input

class Hierarchy(object):

    def __init__(self):

        def tree(): return defaultdict(tree)

        self._dict = tree()

    def __repr__(self):

        def dicts(t): return {k: dicts(t[k]) for k in t}

        return pprint.pformat(dicts(self._dict))

    def add(self, path):
        """
        Add path to this Hierarchy.

        @param path A list of strings.
        """
        if isinstance(path, str):
            path = path.split(',')

        t = self._dict
        for node in path:
           t = t[node]

class UtteranceScenePair(object):

    def __init__(self, utterance, objects, lexicon, num_features=None, probabilistic=True,
            feature_restriction=None):
        """
        Initialise a scene representation.

        @param objects A list of words from which to generate the test scene.
        @param lexicon A Lexicon instance that gives gold-standard lexicon
        representations.
        @param num_features Number of features for each object. If None, number
        of features is given by the number of features associated with each
        object in lexicon.
        @param probabilistic If True, generate the scene representation of each
        object probabilistically.
        @param feature_restriction If a list, restrict the features in the
        scene to those that occur in the list.

        """
        if not isinstance(utterance, list):
            utterance = [utterance]
        self._utterance = utterance

        self._scene = []
        self._objects = objects
        self._referent_to_features_map = {}

        if isinstance(lexicon, str):
            lexicon = input.read_gold_lexicon(lexicon, 10000)

        for obj in objects:
            values_and_features = lexicon.meaning(obj).sorted_features()

            if feature_restriction is not None:
                values_and_features = [(value, feature) for (value, feature) \
                    in values_and_features if feature in feature_restriction]

            if probabilistic is True:

                # normalise the generation probabilities
                probs = np.array([np.float128(value) for (value, feature) in values_and_features])
                probs /= probs.sum()

                try:
                    features = list(np.random.choice(
                        a=[feature for (value, feature) in values_and_features],
                        size=(num_features if num_features is not None else len(a)),
                        replace=False,  # sample features without replacement
                        p=probs
                    ))
                except ValueError:
                    print('There were not enough features to sample from.')
                    raise ValueError

            else:
                # grab the top n familiar features
                if num_features is not None:
                    features = [f for v, f in values_and_features][:num_features]
                else:
                    features = [f for v, f in values_and_features]

                if len(features) < num_features:
                    print('There were not enough features to sample from.')
                    raise ValueError

            self._referent_to_features_map[obj] = features
            self._scene += features

            # remove duplicates
            self._scene = list(set(self._scene))

    def __repr__(self):
        return 'Utterance: ' + str(self._utterance) + '; Objects in scene: ' + str(self._objects) + '; Scene: ' + str(self._scene)

    def scene(self):
        return self._scene[:]

    def objects(self):
        return self._objects.copy()

    def utterance(self):
        return self._utterance

    def pair(self):
        return self._utterance, self._scene[:]

    def features_for_object(self, obj):
        return self._referent_to_features_map[obj].copy()


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
