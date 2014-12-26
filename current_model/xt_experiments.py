#!/usr/bin/python
from __future__ import print_function, division

import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter
import xml.etree.cElementTree as ET
import itertools
import copy
import pprint

import input
import learn
import learnconfig
import wmmapping
import experiment
import experimental_materials

make_inputs = False


class GeneralisationExperiment(experiment.Experiment):

    def setup(self, params, rep):

        if params['forget'] is not False:
            self.forget = True
            self.forget_decay = params['forget']
        else:
            self.forget = False
            self.forget_decay = 0

        # novelty
        if params['novelty'] is not False:
            self.novelty = True
            self.novelty_decay = params['novelty']
        else:
            self.novelty = False
            self.novelty_decay = 0

        # create temporary config file
        config_filename = 'temp_config_'
        config_filename += '_'.join([str(value) for (param, value)
                                    in sorted(params.items())
                                    if len(str(value)) < 6])
        config_filename += '.ini'

        self.config_path = experimental_materials.write_config_file(
            config_filename,
            dummy=params['dummy'],
            forget=self.forget,
            forget_decay=self.forget_decay,
            novelty=self.novelty,
            novelty_decay=self.novelty_decay,
            L=params['lambda'],
            power=params['power'],
            maxtime=params['maxtime']
        )

        # create the gold-standard lexicon and the learner
        learner_config = learnconfig.LearnerConfig(self.config_path)
        beta = learner_config.param_float("beta")
        tree = ET.parse(params['hierarchy'])
        self.lexicon = self.create_lexicon_from_etree(tree, beta)
        stopwords = []
        self.learner = learn.Learner(self.lexicon, learner_config, stopwords)

        # get the corpus
        if params['corpus-path'] == 'generate':
            self.corpus = self.generate_corpus(tree, params)
        else:
            self.corpus = params['corpus-path']

        # training_sets is a dictionary of condition to a list of
        # three training sets
        self.training_sets = {}

        self.training_sets['one example'] = [
            [experimental_materials.UtteranceScenePair(
                utterance='fep',
                objects=['green-pepper'],
                lexicon=self.lexicon,
                probabilistic=False
            )],
            [experimental_materials.UtteranceScenePair(
                utterance='fep',
                objects=['tow-truck'],
                lexicon=self.lexicon,
                probabilistic=False
            )],
            [experimental_materials.UtteranceScenePair(
                utterance='fep',
                objects=['dalmatian'],
                lexicon=self.lexicon,
                probabilistic=False
            )]
        ]

        self.training_sets['three subordinate examples'] = [
            [experimental_materials.UtteranceScenePair(
                utterance='fep',
                objects=['green-pepper'],
                lexicon=self.lexicon,
                probabilistic=False
            )] * 3,
            [experimental_materials.UtteranceScenePair(
                utterance='fep',
                objects=['tow-truck'],
                lexicon=self.lexicon,
                probabilistic=False
            )] * 3,
            [experimental_materials.UtteranceScenePair(
                utterance='fep',
                objects=['dalmatian'],
                lexicon=self.lexicon,
                probabilistic=False
            )] * 3
        ]

        self.training_sets['three basic-level examples'] = [
            [experimental_materials.UtteranceScenePair(
                utterance='fep',
                objects=[obj],
                lexicon=self.lexicon,
                probabilistic=False
            ) for obj in ['green-pepper', 'yellow-pepper', 'red-pepper']],
            [experimental_materials.UtteranceScenePair(
                utterance='fep',
                objects=[obj],
                lexicon=self.lexicon,
                probabilistic=False
            ) for obj in ['tow-truck', 'fire-truck', 'semitrailer']],
            [experimental_materials.UtteranceScenePair(
                utterance='fep',
                objects=[obj],
                lexicon=self.lexicon,
                probabilistic=False
            ) for obj in ['dalmatian', 'poodle', 'pug']]
        ]

        self.training_sets['three superordinate examples'] = [
            [experimental_materials.UtteranceScenePair(
                utterance='fep',
                objects=[obj],
                lexicon=self.lexicon,
                probabilistic=False
            ) for obj in ['green-pepper', 'potato', 'zucchini']],
            [experimental_materials.UtteranceScenePair(
                utterance='fep',
                objects=[obj],
                lexicon=self.lexicon,
                probabilistic=False
            ) for obj in ['tow-truck', 'airliner', 'sailboat']],
            [experimental_materials.UtteranceScenePair(
                utterance='fep',
                objects=[obj],
                lexicon=self.lexicon,
                probabilistic=False
            ) for obj in ['dalmatian', 'tabby', 'salmon']]
        ]

        #pprint.pprint(self.training_sets)

        # there are three test sets, corresponding to the three
        # training sets for each condition
        self.test_sets = [{}, {}, {}]
        self.test_sets[0]['subordinate matches'] = [
            'green-pepper',
            'green-pepper'
        ]
        self.test_sets[1]['subordinate matches'] = [
            'tow-truck',
            'tow-truck'
        ]
        self.test_sets[2]['subordinate matches'] = [
            'dalmatian',
            'dalmatian'
        ]
        self.test_sets[0]['basic-level matches'] = [
            'red-pepper',
            'yellow-pepper'
        ]
        self.test_sets[1]['basic-level matches'] = [
            'fire-truck',
            'semitrailer'
        ]
        self.test_sets[2]['basic-level matches'] = [
            'poodle',
            'pug'
        ]
        self.test_sets[0]['superordinate matches'] = [
            'potato',
            'zucchini'
        ]
        self.test_sets[1]['superordinate matches'] = [
            'airliner',
            'sailboat'
        ]
        self.test_sets[2]['superordinate matches'] = [
            'tabby',
            'salmon'
        ]

        # turn the test sets into scene representations
        for trial in self.test_sets:
            for cond in trial:
                trial[cond] = \
                    [experimental_materials.UtteranceScenePair(
                        utterance='fep',
                        objects=[item],
                        lexicon=self.lexicon,
                        probabilistic=False
                    ) for item in trial[cond]]

        #pprint.pprint(self.test_sets)

        return True

    def iterate(self, params, rep, n):

        results = {}

        for condition in self.training_sets:
            results[condition] = {}

            for i, training_set in enumerate(self.training_sets[condition]):

                self.learner.process_corpus(self.corpus, params['path'])

                for trial in training_set:
                    self.learner.process_pair(trial.utterance(), trial.scene(),
                                              params['path'], False)

                for cond in self.test_sets[i]:
                    for j in range(len(self.test_sets[i][cond])):
                        test_scene = self.test_sets[i][cond][j]
                        gen_prob = calculate_generalisation_probability(
                            self.learner, test_scene.utterance()[0],
                            test_scene.scene())
                        try:
                            results[condition][cond].append(gen_prob)
                        except KeyError:
                            results[condition][cond] = []
                            results[condition][cond].append(gen_prob)

                # reset the learner after each test set
                self.learner.reset()

        bar_chart(results)
        raw_input()

        return results

    def generate_corpus(self, tree, params):
        """
        @param tree An ElementTree instance.
        """
        corpus_path = 'temp_xt_corpus.dev'
        temp_corpus = open(corpus_path, 'w')

        root = tree.getroot()

        bag_of_words = []
        word_to_features_map = {}

        num_superordinate = params['num-superordinate']
        num_basic = params['num-basic-level']
        num_subordinate = params['num-subordinate']

        for sup in root.findall('.//superordinate'):

            bag_of_words.extend([sup.get('label')] * num_superordinate)

            subordinate_choices = sup.findall('.//subordinate')
            choice = subordinate_choices[np.random.randint(
                len(subordinate_choices))]

            word_to_features_map[sup.get('label')] = \
                choice.get('features').split(' ')

        for basic in sup.findall('.//basic-level'):

            bag_of_words.extend([basic.get('label')] * num_basic)

            subordinate_choices = basic.findall('.//subordinate')
            choice = subordinate_choices[np.random.randint(
                len(subordinate_choices))]

            word_to_features_map[basic.get('label')] = \
                choice.get('features').split(' ')

        for sub in basic.findall('.//subordinate'):

            bag_of_words.extend([sub.get('label')] * num_subordinate)
            word_to_features_map[sub.get('label')] = \
                sub.get('features').split(' ')

        np.random.shuffle(bag_of_words)

        for word in bag_of_words:
            feature_choices = word_to_features_map[word]

            if params['probabilistic'] is True:
                s = np.random.randint(1, len(feature_choices)+1)
                scene = list(np.random.choice(a=feature_choices, size=s,
                    replace=False))
            else:
                scene = feature_choices[:]

            # write out the corpus
            temp_corpus.write("1-----\nSENTENCE: ")
            temp_corpus.write(word)
            temp_corpus.write('\n')
            temp_corpus.write("SEM_REP:  ")
            for ft in scene:
                temp_corpus.write("," + ft)
            temp_corpus.write('\n')

        temp_corpus.close()
        return corpus_path

    def create_lexicon_from_etree(self, tree, beta):
        output_filename = 'temp_lexicon-XT.all'
        output_file = open(output_filename, 'w')
        root = tree.getroot()

        for level in ['superordinate', 'basic-level', 'subordinate']:

            for node in root.findall('.//'+level):

                word = node.get('label')
                output_file.write(word + " ")
                features = node.get('features').split(' ')

                for feature in features:
                    output_file.write(feature + ':' + \
                        str(1/float(len(features))) + ',')
                output_file.write('\n\n')

        output_file.close()
        return output_filename

def calculate_generalisation_probability(learner, target_word, target_scene):
    """
    Calculate the probability of learner to generalise the target word to the
    target scene.

    @param learner A learn.Learner instance.
    @param target_word The word for which to calculate the
    generalisation probability.
    @param tearget A list of features representing a scene.

    """
    total = np.float128(0)

    for word in learner._wordsp.all_words(0):

        f_in_y = np.sum([learner._learned_lexicon.prob(word, feature) for feature in target_scene])
        f_in_target = np.sum([learner._learned_lexicon.prob(word, feature) for feature in learner._learned_lexicon.seen_features(target_word)])
        word_freq = learner._wordsp.frequency(word)
        total += f_in_y * f_in_target * word_freq
        total /= np.sum([learner._wordsp.frequency(word) for word in learner._wordsp.all_words(0)])

        print('\t', word, ':', '\tf_in_y =', f_in_y, '\tf_in_dax =', f_in_target, '\tword freq =', word_freq)
    return total

def bar_chart(results):
    conditions = ['one example',
        'three subordinate examples',
        'three basic-level examples',
        'three superordinate examples'
    ]

    l0 = [np.mean(results[cond]['subordinate matches']) for cond in conditions]
    l1 = [np.mean(results[cond]['basic-level matches']) for cond in conditions]
    l2 = [np.mean(results[cond]['superordinate matches']) for cond in conditions]

    ind = np.array([2*n for n in range(len(results))])
    width = 0.5

    fig = plt.figure()
    ax = fig.add_subplot(111)

    p0 = ax.bar(ind,l0,width,color='r')
    p1 = ax.bar(ind+width,l1,width,color='g')
    p2 = ax.bar(ind+2*width,l2,width,color='b')

    ax.set_ylabel("generalisation probability")
    ax.set_xlabel("condition")

    xlabels = ('1', '3 sub.', '3 basic', '3 super.')
    ax.set_xticks(ind + 2 * width)
    ax.set_xticklabels(xlabels)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax.legend( (p0, p1, p2), ('sub.', 'basic', 'super.'), loc='center left', bbox_to_anchor=(1, 0.5))

    title = "Generalization scores"
    ax.set_title(title)

    plt.show()

if __name__ == "__main__":
    e = GeneralisationExperiment()
    e.start()
