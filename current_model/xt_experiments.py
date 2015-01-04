#!/usr/bin/python
from __future__ import print_function, division

# for compatibility over ssh
import matplotlib
matplotlib.use('Agg')

import copy
from datetime import datetime
import itertools
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter
import os
import pprint
import scipy.stats
import xml.etree.cElementTree as ET

import constants as CONST
import evaluate
import input
import learn
import learnconfig
import wmmapping

import experiment
import experimental_materials


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
            beta=params['beta'],
            L=params['lambda'],
            power=params['power'],
            maxtime=params['maxtime']
        )

        # create a dictionary mapping features to their level in the hierarchy
        self.feature_to_level_map = {}

        # create the gold-standard lexicon
        learner_config = learnconfig.LearnerConfig(self.config_path)
        beta = learner_config.param_float("beta")
        tree = ET.parse(params['hierarchy'])
        self.lexicon = self.create_lexicon_from_etree(tree, beta)

        # create the learner
        stopwords = []
        self.learner = learn.Learner(self.lexicon, learner_config, stopwords)

        # get the corpus
        if params['corpus-path'] == 'generate':
            self.corpus = self.generate_corpus(tree, self.learner._gold_lexicon, params)
        else:
            self.corpus = params['corpus-path']

        # training_sets is a dictionary of condition to a list of
        # three training sets
        self.training_sets = {}

        self.training_sets['one example'] = [
            [experimental_materials.UtteranceScenePair(
                utterance='fep',
                objects=[obj],
                lexicon=self.lexicon,
                probabilistic=False
            )] for obj in ['green-pepper', 'tow-truck', 'dalmatian']
        ]

        self.training_sets['three subordinate examples'] = [
            [experimental_materials.UtteranceScenePair(
                utterance='fep',
                objects=[obj],
                lexicon=self.lexicon,
                probabilistic=False
            )] * 3 for obj in ['green-pepper', 'tow-truck', 'dalmatian']
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
                        word = test_scene.utterance()[0]

                        # create the Meaning representation of the test scene
                        meaning = wmmapping.Meaning(self.learner._beta)
                        if params['basic-level-bias'] is not None:
                            d = {}
                            for feature in test_scene.scene():
                                if self.feature_to_level_map[feature] == 'basic-level':
                                    d[feature] = params['basic-level-bias']
                                else:
                                    d[feature] = 1
                            for feature in test_scene.scene():
                                meaning._meaning_probs[feature] = \
                                    d[feature]/sum([v for (f,v) in d.items()])
                        else:
                            for feature in test_scene.scene():
                                meaning._meaning_probs[feature] = \
                                    1/len(test_scene.scene())

                        gen_prob = calculate_generalisation_probability(
                            self.learner, word, meaning,
                            method=params['calculation-type'],
                            std=params['std'])
                        try:
                            results[condition][cond].append(gen_prob)
                        except KeyError:
                            results[condition][cond] = []
                            results[condition][cond].append(gen_prob)

                # reset the learner after each test set
                self.learner.reset()

        savename = ','.join([key + ':' + str(params[key]) for key in params['graph-annotation']])
        savename += '.png'
        annotation = pprint.pformat(dict((key, value) for (key, value) in params.items() if key in params['graph-annotation']))
        bar_chart(results, savename=savename, annotation=annotation)

        return results

    def generate_corpus(self, tree, lexicon, params):
        """
        @param tree An ElementTree instance containing node organised in a
        hierarchy, where the label attribute of each node is a word.
        @param lexicon A wmapping.Lexicon instance containing meanings for the
        words in tree.
        @param params The dictionary of experiment parameters.
        """
        corpus_path = 'temp_xt_corpus_'
        corpus_path += datetime.now().isoformat() + '.dev'
        temp_corpus = open(corpus_path, 'w')

        root = tree.getroot()

        # dictionary of word and random subordinate object tuples
        words_and_objects = []

        num_superordinate = params['num-superordinate']
        num_basic = params['num-basic-level']
        num_subordinate = params['num-subordinate']

        sup_count = 0
        basic_count = 0
        sub_count = 0

        for sup in root.findall('.//superordinate'):
            label = sup.get('label')

            # add the appropriate number of words to the dictionary and choose
            # a random subordinate object
            for i in range(num_superordinate):
                subordinate_choices = sup.findall('.//subordinate')
                choice = subordinate_choices[np.random.randint(
                    len(subordinate_choices))]
                words_and_objects.append((label, choice.get('label')))

            sup_count += num_superordinate

        for basic in root.findall('.//basic-level'):
            label = basic.get('label')

            # add the appropriate number of words to the dictionary and choose
            # a random subordinate object
            for i in range(num_basic):
                subordinate_choices = basic.findall('.//subordinate')
                choice = subordinate_choices[np.random.randint(
                    len(subordinate_choices))]
                words_and_objects.append((label, choice.get('label')))

            basic_count += num_basic

        for sub in root.findall('.//subordinate'):

            label = sub.get('label')
            words_and_objects.extend([(label, label) for i in range(num_subordinate)])

            sub_count += num_subordinate

        np.random.shuffle(words_and_objects)

        for (label, obj) in words_and_objects:
            feature_choices = list(lexicon.seen_features(obj))

            if params['prob'] is True:
                s = np.random.randint(1, len(feature_choices)+1)
                scene = list(np.random.choice(a=feature_choices, size=s,
                    replace=False))
            else:
                scene = feature_choices[:]

            # write out the corpus
            temp_corpus.write("1-----\nSENTENCE: ")
            temp_corpus.write(label)
            temp_corpus.write('\n')
            temp_corpus.write("SEM_REP:  ")
            for ft in scene:
                temp_corpus.write("," + ft)
            temp_corpus.write('\n')

        temp_corpus.close()

        params.update({
            'num-super' : sup_count,
            'num-basic' : basic_count,
            'num-sub' : sub_count
        })

        return corpus_path

    def create_lexicon_from_etree(self, tree, beta):
        output_filename = 'temp_xt_lexicon_'
        output_filename += datetime.now().isoformat() + '.all'
        output_file = open(output_filename, 'w')
        root = tree.getroot()

        for sup in root.findall('.//superordinate'):
            sup_features = []
            sup_features.extend(sup.get('features').split(' '))

            word = sup.get('label')
            output_file.write(word + " ")

            for feature in sup_features:
                output_file.write(feature + ':' + \
                    str(1/float(len(sup_features))) + ',')
                self.feature_to_level_map[feature] = 'superordinate'
            output_file.write('\n\n')

            for basic in sup.findall('.//basic-level'):
                basic_features = sup_features[:]
                basic_features.extend(basic.get('features').split(' '))

                for f in basic.get('features').split(' '):
                    self.feature_to_level_map[f] = 'basic-level'

                word = basic.get('label')
                output_file.write(word + " ")

                for feature in basic_features:
                    output_file.write(feature + ':' + \
                        str(1/float(len(basic_features))) + ',')
                output_file.write('\n\n')

                for sub in basic.findall('.//subordinate'):
                    sub_features = basic_features[:]
                    sub_features.extend(sub.get('features').split(' '))

                    for f in sub.get('features').split(' '):
                        self.feature_to_level_map[f] = 'subordinate'

                    word = sub.get('label')
                    output_file.write(word + " ")

                    for feature in sub_features:
                        output_file.write(feature + ':' + \
                            str(1/float(len(sub_features))) + ',')
                    output_file.write('\n\n')

        output_file.close()

        return output_filename

    def finalize(self, params, rep):
        os.remove(self.corpus)
        os.remove(self.lexicon)


def calculate_generalisation_probability(learner, target_word, target_scene_meaning, method='cosine', std=0.0001):
    """
    Calculate the probability of learner to generalise the target word to the
    target scene.

    @param learner A learn.Learner instance.
    @param target_word The word for which to calculate the
    generalisation probability.
    @param target_scene_meaning A wmmapping.Meaning instance representing a scene.
    @param method If 'cosine', use cosine similarity; if 'gaussian', use a
    Normal distribution with variance std.
    @param std

    """
    def cos(one, two):
        beta = learner._beta
        return evaluate.calculate_similarity(beta, one, two, CONST.COS)

    total = np.float128(0)
    lexicon = learner._learned_lexicon

    for word in learner._wordsp.all_words(0):

        if method == 'cosine' or method == 'cosine-norm':

            cos_y_w = cos(target_scene_meaning, lexicon.meaning(word))
            cos_target_w = cos(lexicon.meaning(target_word), lexicon.meaning(word))

            p_w = learner._wordsp.frequency(word) / np.sum([learner._wordsp.frequency(w) for w in learner._wordsp.all_words(0)])

            term = cos_y_w * cos_target_w * p_w

            #print('\t', word, ':', '\tcos_y_w =', cos_y_w, '\tcos_target_w =', cos_target_w, '\tp(w) =', p_w,
                    #'\tterm:', cos_y_w * cos_target_w * p_w)

            if method == 'cosine-norm':

                denom = np.sum([cos(lexicon.meaning(w), lexicon.meaning(word)) for w in learner._wordsp.all_words(0)])
                term /= denom
                term /= denom

            total += term

        elif method == 'gaussian':

            target_word_meaning = lexicon.meaning(target_word)
            y_factor = 1
            target_factor = 1

            for feature in target_scene:

                mean = lexicon.prob(word, feature)
                dist = scipy.stats.norm(loc=mean, scale=std)

                y_factor *= dist.pdf(target_scene_meaning.prob(feature))

            for feature in [f for f in lexicon.seen_features(target_word) if lexicon.prob(target_word, f) != lexicon.meaning(target_word).unseen_prob()]:

                mean = lexicon.prob(word, feature)
                dist = scipy.stats.norm(loc=mean, scale=std)

                target_factor *= dist.pdf(target_word_meaning.prob(feature))

            word_freq = learner._wordsp.frequency(word)

            total += y_factor * target_factor * word_freq

            total /= np.sum([learner._wordsp.frequency(w) for w in learner._wordsp.all_words(0)])

            #print('\t', word, ':', '\tfirst factor =', y_factor, '\tsecond factor =', target_factor, '\tword freq =', word_freq)

        else:
            raise NotImplementedError

    return total

def bar_chart(results, savename=None, annotation=None):
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

    ax.legend( (p0, p1, p2), ('sub.', 'basic', 'super.'), loc='center left', bbox_to_anchor=(1, 0))

    title = "Generalization scores"
    ax.set_title(title)

    if annotation is not None:
        fig.text(0.8, 0.3, annotation, ha='center', fontsize='small')

    if savename is None:
        plt.show()
    else:
        plt.savefig(savename)

if __name__ == "__main__":
    e = GeneralisationExperiment()
    e.start()
