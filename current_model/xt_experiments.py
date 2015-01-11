#!/usr/bin/python
from __future__ import print_function, division

import copy
import csv
from datetime import datetime
import heapq
import itertools
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mpmath; mpmath.mp.dps = 50
import nltk
from nltk.corpus.reader import CorpusReader
import numpy as np
import operator
import os
import pickle
import pprint
import random
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

verbose = True

class GeneralisationExperiment(experiment.Experiment):

    def pre_setup(self, params):

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
            L=(1/params['beta']),
            power=params['power'],
            alpha=params['alpha'],
            epsilon=params['epsilon'],
            maxtime=params['maxtime']
        )

        # create a dictionary mapping features to their level in the hierarchy
        self.feature_to_level_map = {}

        # create the gold-standard lexicon
        learner_config = learnconfig.LearnerConfig(self.config_path)
        beta = learner_config.param_float("beta")

        # get the corpus
        if params['corpus'] == 'generate-simple':

            tree = ET.parse(params['hierarchy'])
            self.gold_standard_lexicon = self.create_lexicon_from_etree(tree, beta)

            # create the learner
            stopwords = []
            self.learner = learn.Learner(self.gold_standard_lexicon, learner_config, stopwords)

            self.corpus = self.generate_simple_corpus(tree, self.learner._gold_lexicon, params)
            self.training_sets = self.generate_simple_training_sets()
            self.test_sets = self.generate_simple_test_sets()

        elif params['corpus'] == 'generate-naturalistic':

            self.corpus, sup, basic, sub = self.generate_naturalistic_corpus(params['corpus-path'], params['lexname'], params['beta'], params['maxtime'], params['num-features'], params)

            # create the learner
            stopwords = []
            if params['new-learner'] is True:
                self.learner = learn.Learner(params['lexname'], learner_config, stopwords)
                self.learner.process_corpus(self.corpus, params['path'])
                learner_dump = open(params['learner-path'], "wb")
                pickle.dump(self.learner, learner_dump)
                learner_dump.close()

            else:
                learner_dump = open(params['learner-path'], "rb")
                self.learner = pickle.load(learner_dump)
                learner_dump.close()

            self.gold_standard_lexicon = self.learner._gold_lexicon
            self.learner = None

            self.training_sets, fep_features = self.generate_naturalistic_training_sets(sup, basic, sub)
            self.test_sets = self.generate_naturalistic_test_sets(sup, basic, sub, fep_features)

        else:
            raise NotImplementedError

        return True

    def iterate(self, params, rep, n):

        results = {}

        for condition in self.training_sets:
            results[condition] = {}

            for i, training_set in enumerate(self.training_sets[condition]):

                learner_dump = open(params['learner-path'], "rb")
                self.learner = pickle.load(learner_dump)
                learner_dump.close()

                if verbose is True:
                    print('condition:', condition, 'training set:')
                    pprint.pprint(training_set)
                    print('\n')

                for trial in training_set:
                    self.learner.process_pair(trial.utterance(), trial.scene(),
                                              params['path'], False)

                    if verbose is True:
                        print('trial fep meaning:')
                        print([(feature, self.learner._learned_lexicon.prob(trial.utterance()[0], feature)) for feature in self.learner._learned_lexicon.seen_features(trial.utterance()[0]) if feature in trial.scene()])
                        print('\n')

                for cond in self.test_sets[i]:

                    take_average = []
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
                                meaning._seen_features.append(feature)
                        else:
                            for feature in test_scene.scene():
                                meaning._meaning_probs[feature] = \
                                    1/len(test_scene.scene())
                                meaning._seen_features.append(feature)

                        gen_prob = calculate_generalisation_probability(
                            self.learner, word, meaning,
                            method=params['calculation-type'],
                            std=params['std'],
                            delta=params['delta-interval'],
                            include_target=params['include-fep-in-loop'],
                            target_word_as_distribution=params['use-distribution-fep']
                            )
                        take_average.append(gen_prob)

                    gen_prob = np.mean(take_average)

                    try:
                        results[condition][cond].append(gen_prob)
                    except KeyError:
                        results[condition][cond] = []
                        results[condition][cond].append(gen_prob)

                # reset the learner after each test set
                self.learner = None

        savename = ','.join([key + ':' + str(params[key]) for key in params['graph-annotation']])
        savename += '.png'
        annotation = str(dict((key, value) for (key, value) in params.items() if key in params['graph-annotation']))
        bar_chart(results, savename=savename, annotation=annotation, normalise_over_test_scene=params['normalise-over-test-scene'])

        return results

    def generate_simple_corpus(self, tree, lexicon, params):
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

    def generate_simple_training_sets():
        # training_sets is a dictionary of condition to a list of
        # three training sets
        training_sets = {}

        training_sets['one example'] = [
            [experimental_materials.UtteranceScenePair(
                utterance='fep',
                objects=[obj],
                lexicon=self.gold_standard_lexicon,
                probabilistic=False
            )] for obj in ['green-pepper', 'tow-truck', 'dalmatian']
        ]

        training_sets['three subordinate examples'] = [
            [experimental_materials.UtteranceScenePair(
                utterance='fep',
                objects=[obj],
                lexicon=self.gold_standard_lexicon,
                probabilistic=False
            )] * 3 for obj in ['green-pepper', 'tow-truck', 'dalmatian']
        ]

        training_sets['three basic-level examples'] = [
            [experimental_materials.UtteranceScenePair(
                utterance='fep',
                objects=[obj],
                lexicon=self.gold_standard_lexicon,
                probabilistic=False
            ) for obj in ['green-pepper', 'yellow-pepper', 'red-pepper']],
            [experimental_materials.UtteranceScenePair(
                utterance='fep',
                objects=[obj],
                lexicon=self.gold_standard_lexicon,
                probabilistic=False
            ) for obj in ['tow-truck', 'fire-truck', 'semitrailer']],
            [experimental_materials.UtteranceScenePair(
                utterance='fep',
                objects=[obj],
                lexicon=self.gold_standard_lexicon,
                probabilistic=False
            ) for obj in ['dalmatian', 'poodle', 'pug']]
        ]

        training_sets['three superordinate examples'] = [
            [experimental_materials.UtteranceScenePair(
                utterance='fep',
                objects=[obj],
                lexicon=self.gold_standard_lexicon,
                probabilistic=False
            ) for obj in ['green-pepper', 'potato', 'zucchini']],
            [experimental_materials.UtteranceScenePair(
                utterance='fep',
                objects=[obj],
                lexicon=self.gold_standard_lexicon,
                probabilistic=False
            ) for obj in ['tow-truck', 'airliner', 'sailboat']],
            [experimental_materials.UtteranceScenePair(
                utterance='fep',
                objects=[obj],
                lexicon=self.gold_standard_lexicon,
                probabilistic=False
            ) for obj in ['dalmatian', 'tabby', 'salmon']]
        ]

        #pprint.pprint(training_sets)

        return training_sets

    def generate_simple_test_sets():

        # there are three test sets, corresponding to the three
        # training sets for each condition
        test_sets = [{}, {}, {}]
        test_sets[0]['subordinate matches'] = [
            'green-pepper',
            'green-pepper'
        ]
        test_sets[1]['subordinate matches'] = [
            'tow-truck',
            'tow-truck'
        ]
        test_sets[2]['subordinate matches'] = [
            'dalmatian',
            'dalmatian'
        ]
        test_sets[0]['basic-level matches'] = [
            'red-pepper',
            'yellow-pepper'
        ]
        test_sets[1]['basic-level matches'] = [
            'fire-truck',
            'semitrailer'
        ]
        test_sets[2]['basic-level matches'] = [
            'poodle',
            'pug'
        ]
        test_sets[0]['superordinate matches'] = [
            'potato',
            'zucchini'
        ]
        test_sets[1]['superordinate matches'] = [
            'airliner',
            'sailboat'
        ]
        test_sets[2]['superordinate matches'] = [
            'tabby',
            'salmon'
        ]

        # turn the test sets into scene representations
        for trial in test_sets:
            for cond in trial:
                trial[cond] = \
                    [experimental_materials.UtteranceScenePair(
                        utterance='fep',
                        objects=[item],
                        lexicon=self.gold_standard_lexicon,
                        probabilistic=False
                    ) for item in trial[cond]]

        #pprint.pprint(test_sets)

        return test_sets

    def generate_naturalistic_corpus(self, corpus_path, lexicon, beta, maxtime, n, params):
        temp_corpus_path = 'temp_xt_corpus_'
        temp_corpus_path += datetime.now().isoformat() + '.dev'
        temp_corpus = open(temp_corpus_path, 'w')

        corpus = input.Corpus(corpus_path)

        word_to_frequency_map = {}
        with open('lemma.al', 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            for row in reader:
                word_to_frequency_map[row[2]] = row[1]

        wn = nltk.corpus.WordNetCorpusReader(nltk.data.find('corpora/wordnet'), None)
        word_list = []

        sentence_count = 0

        while sentence_count < maxtime:
            (words, features) = corpus.next_pair()

            for word in words:
                if word.split(':')[1] == 'N':
                    word = word.split(':')[0]
                    try:
                        s = wn.synsets(word, 'n')[0]
                        word_list.append(word)
                    except IndexError:
                        pass # word not recognised by WordNet

            sentence_count += 1

        word_list = list(set(word_list))

        hierarchy = {}

        # generate (superordinate, basic, subordinate) triples
        for word in word_list:

            try:
                s = wn.synsets(word, 'n')[0]

                # only check words at the bottom of the hierarchy
                if s.hyponyms() == []:

                    lowest_to_highest = []
                    searching = True
                    encountered = []

                    while searching and str(s.name()).split('.')[0] not in encountered:

                        d = {}
                        encountered.append(str(s.name()).split('.')[0])
                        hypers = [w for w in s.hypernyms()]

                        if hypers == []:
                            searching = False

                        else:
                            for hyper in hypers:
                                try:
                                    hyper = str(hyper.name()).split('.')[0]
                                    d[hyper] = word_to_frequency_map[hyper]
                                except KeyError: # no frequency information
                                    pass
                            try:
                                s = max(hypers, key=(lambda key: d[str(key.name()).split('.')[0]]))
                                lowest_to_highest.append(str(s.name()).split('.')[0])

                            except KeyError: # no frequency information was found
                                s = hypers[0] # choose first hypernym (random)

                    if len(lowest_to_highest) >= 3: # at least three levels

                        # discard the highest level as it is tooo broad
                        # instead, use the second-highest as the superordinate level,
                        # and choose the highest frequency word as the basic level
                        lowest_to_highest.pop(-1)
                        sup = lowest_to_highest.pop(-1)+':N'
                        basic = max(lowest_to_highest, key=(lambda key: word_to_frequency_map[key]))+':N'
                        sub = word+':N'

                        try:
                            hierarchy[sup][basic].append(sub)
                        except KeyError:
                            try:
                                hierarchy[sup][basic] = []
                                hierarchy[sup][basic].append(sub)
                            except KeyError:
                                hierarchy[sup] = {}
                                hierarchy[sup][basic] = []
                                hierarchy[sup][basic].append(sub)

                    else:
                        pass

            except nltk.corpus.reader.wordnet.WordNetError:
                pass # word not recognised by WordNet

        # store choices for choosing subordinate items
        word_to_list_of_feature_bundles_map = {}

        sup_fs = []
        basic_fs = []
        sub_fs = []

        hierarchy_words = []
        basic_to_delete = []
        sup_count = len(hierarchy.keys())
        basic_count = 0
        sub_count = 0
        for sup in hierarchy:
            word_to_list_of_feature_bundles_map[sup] = []
            sup_features = [sup + '_f' + str(i) for i in range(n)]
            sup_fs.extend(sup_features)
            hierarchy_words.append(sup)
            for basic in hierarchy[sup]:
                hierarchy[sup][basic] = list(set(hierarchy[sup][basic]))
                if basic in hierarchy_words:
                    basic_to_delete.append((sup, basic))
                else:
                    word_to_list_of_feature_bundles_map[basic] = []
                    basic_features = [basic + '_f' + str(i) for i in range(n)]
                    basic_fs.extend(basic_features)
                    hierarchy_words.append(basic)
                    for sub in hierarchy[sup][basic]:
                        sub_features = [sub + '_f' + str(i) for i in range(n)]
                        sub_fs.extend(sub_features)
                        if sub in hierarchy_words:
                            hierarchy[sup][basic].remove(sub)
                        else:
                            hierarchy_words.append(sub)
                            features = sup_features + basic_features + sub_features
                            word_to_list_of_feature_bundles_map[sub] = []
                            word_to_list_of_feature_bundles_map[sub].append(features[:])
                            word_to_list_of_feature_bundles_map[basic].append(features[:])
                            word_to_list_of_feature_bundles_map[sup].append(features[:])
                    sub_count += len(hierarchy[sup][basic])
            basic_count += len(hierarchy[sup].keys()) - len(basic_to_delete)

        for sup, basic in basic_to_delete:
            del hierarchy[sup][basic]

        hierarchy_words.sort()

        # rewrite the corpus
        corpus = input.Corpus(corpus_path)
        lexicon = input.read_gold_lexicon(lexicon, beta)

        sentence_count = 0

        while sentence_count < maxtime:
            (words, features) = corpus.next_pair()

            scene = ''
            for word in words:
                if word.split(':')[1] == 'N' and word in hierarchy_words:
                    ref = word_to_list_of_feature_bundles_map[word]
                    choice = ref[np.random.randint(len(ref))]
                    for f in choice:
                        wsem = ",%s" % (f)
                        scene = scene + wsem
                else:
                    for v,f in lexicon.meaning(word).sorted_features():
                        prob = float(v)
                        r = random.random()
                        if prob > r:
                            wsem = ",%s" % (f)
                            scene = scene + wsem

            temp_corpus.write("1-----\nSENTENCE: ")
            temp_corpus.write(' '.join(words))
            temp_corpus.write('\n')
            temp_corpus.write("SEM_REP:  ")
            temp_corpus.write(scene)
            temp_corpus.write('\n')

            sentence_count += 1

        temp_corpus.close()

        params.update({
            'num-super' : sup_count,
            'num-basic' : basic_count,
            'num-sub' : sub_count
        })

        np.random.shuffle(sup_fs)
        np.random.shuffle(basic_fs)
        #np.random.shuffle(sub_fs)

        sup = list(sup_fs)
        basic = list(basic_fs)

        #sub = list(sub_fs)
        # assumption: all subordinate features are novel
        sub = ['sub_f' + str(i) for i in range(100)]

        return temp_corpus_path, sup, basic, sub

    def generate_naturalistic_training_sets(self, sup, basic, sub):

        fep_features = []

        training_sets = {}
        training_sets['one example'] = []
        training_sets['three subordinate examples'] = []
        training_sets['three basic-level examples'] = []
        training_sets['three superordinate examples'] = []

        for i in range(5):

            sup_1 = [sup.pop()]
            basic_1 = [basic.pop()]
            sub_1 = [sub.pop()]

            fep_sup = sup_1[0]
            fep_basic = basic_1[0]
            fep_sub = sub_1[0]

            training_sets['one example'].append(
                [experimental_materials.UtteranceScenePair(
                    utterance='fep',
                    scene=sup_1+basic_1+sub_1,
                    lexicon=self.gold_standard_lexicon,
                    probabilistic=False
                )]
            )

            training_sets['three subordinate examples'].append(
                [experimental_materials.UtteranceScenePair(
                    utterance='fep',
                    scene=sup_1+basic_1+sub_1,
                    lexicon=self.gold_standard_lexicon,
                    probabilistic=False
                )] * 3
            )

            sub_1 = [sub.pop()]
            sub_2 = [sub.pop()]
            sub_3 = [sub.pop()]

            training_sets['three basic-level examples'].append(
                [
                    experimental_materials.UtteranceScenePair(
                        utterance='fep',
                        scene=sup_1+basic_1+sub_1,
                        lexicon=self.gold_standard_lexicon,
                        probabilistic=False
                    ),
                    experimental_materials.UtteranceScenePair(
                        utterance='fep',
                        scene=sup_1+basic_1+sub_2,
                        lexicon=self.gold_standard_lexicon,
                        probabilistic=False
                    ),
                    experimental_materials.UtteranceScenePair(
                        utterance='fep',
                        scene=sup_1+basic_1+sub_3,
                        lexicon=self.gold_standard_lexicon,
                        probabilistic=False
                    )
                ]
            )

            basic_1 = [basic.pop()]
            basic_2 = [basic.pop()]
            basic_3 = [basic.pop()]
            sub_1 = [sub.pop()]
            sub_2 = [sub.pop()]
            sub_3 = [sub.pop()]

            training_sets['three superordinate examples'].append(
                [
                    experimental_materials.UtteranceScenePair(
                        utterance='fep',
                        scene=sup_1+basic_1+sub_1,
                        lexicon=self.gold_standard_lexicon,
                        probabilistic=False
                    ),
                    experimental_materials.UtteranceScenePair(
                        utterance='fep',
                        scene=sup_1+basic_2+sub_2,
                        lexicon=self.gold_standard_lexicon,
                        probabilistic=False
                    ),
                    experimental_materials.UtteranceScenePair(
                        utterance='fep',
                        scene=sup_1+basic_3+sub_3,
                        lexicon=self.gold_standard_lexicon,
                        probabilistic=False
                    )
                ]
            )

            fep_features.append((fep_sup, fep_basic, fep_sub))

        #pprint.pprint(training_sets)

        return training_sets, fep_features

    def generate_naturalistic_test_sets(self, sup, basic, sub, fep_features):

        test_sets = []

        for i in range(5):
            test_sets.append({})

            sup_1 = [fep_features[i][0]]
            basic_1 = [fep_features[i][1]]
            sub_1 = [fep_features[i][2]]

            test_sets[i]['subordinate matches'] = [
                experimental_materials.UtteranceScenePair(
                    utterance='fep',
                    scene=sup_1+basic_1+sub_1,
                    lexicon=self.gold_standard_lexicon,
                    probabilistic=False
                )
            ] * 2

            sub_1 = [sub.pop()]
            sub_2 = [sub.pop()]

            test_sets[i]['basic-level matches'] = [
                experimental_materials.UtteranceScenePair(
                    utterance='fep',
                    scene=sup_1+basic_1+sub_1,
                    lexicon=self.gold_standard_lexicon,
                    probabilistic=False
                ),
                experimental_materials.UtteranceScenePair(
                    utterance='fep',
                    scene=sup_1+basic_1+sub_2,
                    lexicon=self.gold_standard_lexicon,
                    probabilistic=False
                )
            ]

            basic_1 = [basic.pop()]
            basic_2 = [basic.pop()]
            sub_1 = [sub.pop()]
            sub_2 = [sub.pop()]

            test_sets[i]['superordinate matches'] = [
                experimental_materials.UtteranceScenePair(
                    utterance='fep',
                    scene=sup_1+basic_1+sub_1,
                    lexicon=self.gold_standard_lexicon,
                    probabilistic=False
                ),
                experimental_materials.UtteranceScenePair(
                    utterance='fep',
                    scene=sup_1+basic_2+sub_2,
                    lexicon=self.gold_standard_lexicon,
                    probabilistic=False
                )
            ]

        #pprint.pprint(test_sets)

        return test_sets

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
        #os.remove(self.corpus)
        #os.remove(self.lexicon)
        pass


def calculate_generalisation_probability(learner, target_word, target_scene_meaning, method='cosine', std=0.0001, delta=0.0001, include_target=True, target_word_as_distribution=False):
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
        return np.float64(evaluate.calculate_similarity(beta, one, two, CONST.COS))

    def KL_prob(mu1, mu2, sigma1, sigma2):
        """
        Compute a 'probabilty' measure for the KL divergence of two univariate
        Gaussians.

        D_{KL} = \frac{(\mu_1-\mu_2)^2}{2\sigma^2_2} +
        \frac{1}{2}\left(\frac{\sigma_1^2}{\sigma_2^2} - 1 -
        \log \frac{\sigma_1^2}{\sigma_2^2}\right)

        p_{KL} = 1-\exp( -D_{KL} )

        """
        kl = np.divide((mu1 - mu2)**2, 2*np.square(sigma2))
        kl += np.multiply(1/2, (
            np.divide(sigma1**2, sigma2**2) -\
            1 - np.log(np.divide(sigma1**2, sigma2**2))))
        return 1 - np.exp(-kl)

    lexicon = learner.learned_lexicon()

    if method == 'no-word-averaging':

        total = cos(target_scene_meaning, lexicon.meaning(target_word))

    else:

        total = np.float64(0)

        words = learner._wordsp.all_words(0)[:]
        if include_target is False:
            words.remove(target_word)

        sum_word_frequency = np.sum([learner._wordsp.frequency(w) for w in words])

        for word in words:

            if method == 'cosine' or method == 'cosine-norm':

                cos_y_w = cos(target_scene_meaning, lexicon.meaning(word))
                cos_target_w = cos(lexicon.meaning(target_word), lexicon.meaning(word))

                p_w = learner._wordsp.frequency(word) / sum_word_frequency

                term = cos_y_w * cos_target_w * p_w

                #print('\t', word, ':', '\tcos_y_w =', cos_y_w, '\tcos_target_w =', cos_target_w, '\tp(w) =', p_w,
                        #'\tterm:', cos_y_w * cos_target_w * p_w)

                if method == 'cosine-norm':

                    # this normalisation requires too much time (it is an inner loop over words)
                    denom = np.sum([cos(lexicon.meaning(w), lexicon.meaning(word)) for w in words])
                    term /= denom
                    term /= denom

                total += term

            elif method == 'gaussian-norm':

                target_word_meaning = lexicon.meaning(target_word)
                y_factor = np.float64(1)
                target_factor = np.float64(1)
                delta = np.float64(delta)

                features_to_distributions_map = {}

                for feature in target_scene_meaning.seen_features():

                    try:
                        dist = features_to_distributions_map[feature]
                    except KeyError:
                        mean = lexicon.prob(word, feature)
                        features_to_distributions_map[feature] = scipy.stats.norm(loc=mean, scale=std)
                        dist = features_to_distributions_map[feature]

                    integral = dist.cdf(np.float64(target_scene_meaning.prob(feature))+delta) -\
                        dist.cdf(np.float64(target_scene_meaning.prob(feature)))

                    y_factor *= integral * delta

                for feature in lexicon.seen_features(target_word):

                    if target_word_as_distribution is False:

                        try:
                            dist = features_to_distributions_map[feature]
                        except KeyError:
                            mean = lexicon.prob(word, feature)
                            features_to_distributions_map[feature] = scipy.stats.norm(loc=mean, scale=std)
                            dist = features_to_distributions_map[feature]

                        integral = dist.cdf(np.float64(target_word_meaning.prob(feature))+delta) -\
                            dist.cdf(np.float64(target_word_meaning.prob(feature)))

                        target_factor *= integral * delta

                    else:

                        import pdb; pdb.set_trace()

                        mu1 = lexicon.prob(word, feature)
                        mu2 = target_word_meaning.prob(feature)
                        target_factor *= KL_prob(mu1, mu2, std, std)

                word_freq = learner._wordsp.frequency(word)

                term = y_factor * target_factor * word_freq
                term /= sum_word_frequency

                total += term

                #print('\t', word, ':', '\tfirst factor =', y_factor, '\tsecond factor =', target_factor, '\tword freq =', word_freq)

            else:
                raise NotImplementedError

    return total

def bar_chart(results, savename=None, annotation=None, normalise_over_test_scene=True):

    conditions = ['one example',
        'three subordinate examples',
        'three basic-level examples',
        'three superordinate examples'
    ]

    ind = np.array([2*n for n in range(len(results))])
    width = 0.25

    nrows = int(np.ceil(len(results[conditions[0]]['subordinate matches']) / 2.0))

    fig, axes = plt.subplots(nrows=nrows, ncols=2, sharex=True, sharey=True)

    for i, ax in enumerate(axes.flat):

        if i == len(results[conditions[0]]['subordinate matches']):

            ax.set_title('Average over all training-test sets', fontsize='small')

            l0 = [np.mean(results[cond]['subordinate matches']) for cond in conditions]
            l1 = [np.mean(results[cond]['basic-level matches']) for cond in conditions]
            l2 = [np.mean(results[cond]['superordinate matches']) for cond in conditions]

            if normalise_over_test_scene is True:

                l0 = np.array(l0)
                l1 = np.array(l1)
                l2 = np.array(l2)

                denom = [np.mean(results[cond]['subordinate matches']) for cond in conditions]
                denom = np.add(denom, [np.mean(results[cond]['basic-level matches']) for cond in conditions])
                denom = np.add(denom, [np.mean(results[cond]['superordinate matches']) for cond in conditions])

                l0 /= denom
                l1 /= denom
                l2 /= denom

                l0 = list(l0)
                l1 = list(l1)
                l2 = list(l2)

            p0 = ax.bar(ind,l0,width,color='r')
            p1 = ax.bar(ind+width,l1,width,color='g')
            p2 = ax.bar(ind+2*width,l2,width,color='b')

        elif i > len(results[conditions[0]]['subordinate matches']):
            pass

        else:
            l0 = [results[cond]['subordinate matches'][i] for cond in conditions]
            l1 = [results[cond]['basic-level matches'][i] for cond in conditions]
            l2 = [results[cond]['superordinate matches'][i] for cond in conditions]

            if normalise_over_test_scene is True:

                l0 = np.array(l0)
                l1 = np.array(l1)
                l2 = np.array(l2)

                denom = [results[cond]['subordinate matches'][i] for cond in conditions]
                denom = np.add(denom, [results[cond]['basic-level matches'][i] for cond in conditions])
                denom = np.add(denom, [results[cond]['superordinate matches'][i] for cond in conditions])

                l0 /= denom
                l1 /= denom
                l2 /= denom

                l0 = list(l0)
                l1 = list(l1)
                l2 = list(l2)

            p0 = ax.bar(ind,l0,width,color='r')
            p1 = ax.bar(ind+width,l1,width,color='g')
            p2 = ax.bar(ind+2*width,l2,width,color='b')

        xlabels = ('1', '3 sub.', '3 basic', '3 super.')
        ax.set_xticks(ind + 2 * width)
        ax.set_xticklabels(xlabels)

    #ax.set_ylabel("gen. prob.")
    #ax.set_xlabel("condition")
    plt.ylim((0,1))

    lgd = plt.legend( (p0, p1, p2), ('sub.', 'basic', 'super.'), loc='lower center', bbox_to_anchor=(0, -0.1, 1, 1), bbox_transform=plt.gcf().transFigure )

    title = "Generalization scores"

    if annotation is not None:
        title += '\n'+annotation

    fig.suptitle(title)

    if savename is None:
        plt.show()
    else:
        plt.savefig(savename, bbox_extra_artists=(lgd,), bbox_inches='tight')

if __name__ == "__main__":
    e = GeneralisationExperiment()
    e.start()
