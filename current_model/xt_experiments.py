#!/usr/bin/python
from __future__ import print_function, division

import csv
from datetime import datetime
import logging
import math
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mpmath; mpmath.mp.dps = 50
import nltk
from nltk.corpus.reader import CorpusReader
import numpy as np
import os
import pickle
import random
import scipy.stats
import xml.etree.cElementTree as ET

import constants as CONST
import evaluate
import filewriter
import input
import learn
import learnconfig
import wmmapping

import experiment
import experimental_materials

latex = True

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
            #L=(1),
            power=params['power'],
            alpha=params['alpha'],
            epsilon=params['epsilon'],
            maxtime=params['maxtime']
        )

        # create a dictionary mapping features to their level in the hierarchy
        self.feature_to_level_map = {}

        # create the gold-standard lexicon
        self.learner_config = learnconfig.LearnerConfig(self.config_path)
        beta = self.learner_config.param_float("beta")

        # get the corpus
        if params['corpus'] == 'generate-simple':

            # get the location of the hierarchy
            tree = ET.parse(params['hierarchy'])
            self.gold_standard_lexicon = self.create_lexicon_from_etree(tree, beta)

            # create the learner
            stopwords = []
            self.learner = learn.Learner(self.gold_standard_lexicon, self.learner_config, stopwords,
                    k_sub=params['k-sub'], k_basic=params['k-basic'], k_sup=params['k-sup'])

            self.corpus = self.generate_simple_corpus(tree, self.learner._gold_lexicon, params)

        elif params['corpus'] == 'generate-naturalistic':

            self.corpus, self.sup, self.basic, self.sub = self.generate_naturalistic_corpus(params)

        return True

    def setup(self, params, rep):

        self.learner_path = params['learner-path']
        self.learner_path += datetime.now().isoformat() + '.pkl'

        if params['corpus'] == 'generate-simple':
            self.training_sets = self.generate_simple_training_sets()
            self.test_sets = self.generate_simple_test_sets()

        elif params['corpus'] == 'generate-naturalistic':
            # create the learner
            stopwords = []
            if params['new-learner'] is True:
                #self.learner = learn.Learner(params['lexname'], self.learner_config, stopwords)
                #self.learner.process_corpus(self.corpus, params['path'])
                #learner_dump = open(self.learner_path, "wb")
                #pickle.dump(self.learner, learner_dump)
                #learner_dump.close()
                self.learner = learn.Learner(params['lexname'], self.learner_config, stopwords,
                    k_sub=params['k-sub'], k_basic=params['k-basic'], k_sup=params['k-sup'])
                learner_dump = open(self.learner_path, "wb")
                pickle.dump(self.learner, learner_dump)
                learner_dump.close()
            else:
                learner_dump = open(self.learner_path, "rb")
                self.learner = pickle.load(learner_dump)
                learner_dump.close()

            self.gold_standard_lexicon = self.learner._gold_lexicon
            self.learner = None

            self.training_sets, fep_features = self.generate_naturalistic_training_sets(self.sup, self.basic, self.sub, params['num-sets'], num_sub_features=params['num-sub-features'], num_basic=params['num-basic-features'], num_sup=params['num-sup-features'])
            self.test_sets = self.generate_naturalistic_test_sets(self.sup, self.basic, self.sub, fep_features, params['num-sets'], num_sub_features=params['num-sub-features'], num_basic=params['num-basic-features'], num_sup=params['num-sup-features'])

        else:
            raise NotImplementedError

        return True

    def iterate(self, params, rep, n):

        results = {}
        p_fep_fep = {}

        if latex is True:
            print('\\documentclass{article}')
            print('\\usepackage[landscape]{geometry}')
            print('\\usepackage{underscore}')
            print('\\newcommand{\specialcell}[2][l]{\\begin{tabular}[#1]{@{}l@{}}#2\\end{tabular}}')
            print('\\begin{document}')

        conds = ['one example', 'three subordinate examples', 'three basic-level examples',
                 'three superordinate examples']

        assert set(self.training_sets.keys()) == set(conds)

        #for condition in self.training_sets:
        for condition in conds:

            results[condition] = {}

            for i, training_set in enumerate(self.training_sets[condition]):

                learner_dump = open(self.learner_path, "rb")
                self.learner = pickle.load(learner_dump)
                learner_dump.close()

                if latex is True:
                    print('\\subsection*{'+condition+'}')
                    print('\\begin{tabular}{l c c}')
                    print('training trial 1 & trial 2 & trial 3 \\\\')

                for trial in training_set:
                    self.learner.process_pair(trial.utterance(), trial.scene(),
                                              params['path'], False)
                    if latex is True:
                        print(str(trial.scene())[1:-1], '&')

                if latex is True:
                    print("""\\\\\hline""")
                    print('& feature: probability & generalization probability measure  \\\\')
                    print("""\hline""")
                    print("""\hline""")

                for cond in self.test_sets[i]:

                    take_average = 0
                    count = 0

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

                        if latex is True:
                            print(cond + ' & ')
                            print("""\specialcell{""")

                        gen_prob, p_f_f, feature_count = calculate_generalisation_probability(
                            self.learner, word, meaning,
                            method=params['calculation-type'],
                            std=params['std'],
                            delta=params['delta-interval'],
                            include_target=params['include-fep-in-loop'],
                            target_word_as_distribution=params['use-distribution-fep'],
                            ratio_to_mean=params['ratio-to-mean'],
                            log=params['log']
                            )

                        ## account for all features accross training, test
                        ## there should be 11 sub, 7 basic, 1 sup

                        #sub_count = 0
                        #basic_count = 0
                        #sup_count = 0

                        #for feature in set(trial.scene() + test_scene.scene()):
                        #    if feature.startswith('sub'):
                        #        sub_count += 1
                        #    elif feature.startswith('bas'):
                        #        basic_count += 1
                        #    elif feature.startswith('sup'):
                        #        sup_count += 1
                        #    else:
                        #        raise NotImplementedError

                        #assert sup_count == 1

                        #while sub_count < 11:
                        #    if latex is True:
                        #        print('unseen ftr:', filewriter.round_to_sig_digits((1-1./params['k-sub']), 4), '\\\\')
                        #    gen_prob *= (1-1./params['k-sub'])
                        #    sub_count += 1

                        #while basic_count < 7:
                        #    if latex is True:
                        #        print('unseen ftr:', filewriter.round_to_sig_digits((1-1./params['k-basic']), 4), '\\\\')
                        #    gen_prob *= (1-1./params['k-basic'])
                        #    basic_count += 1

                        if latex is True:
                            print("""}""")
                            print('&', filewriter.round_to_sig_digits(gen_prob, 4), '\\\\')
                            print("""\hline""")
                        p_fep_fep[condition] = p_f_f

                        take_average = mpmath.fadd(take_average, gen_prob)
                        count += 1

                        if latex is True:
                            print('\n')

                    gen_prob = mpmath.fdiv(take_average, count)

                    try:
                        results[condition][cond].append(gen_prob)
                    except KeyError:
                        results[condition][cond] = []
                        results[condition][cond].append(gen_prob)

                if latex is True:
                    print('\\end{tabular}')
                    print('\n')

                # reset the learner after each test set
                self.learner = None

        if params['compare-to-fep'] is False:
            p_fep_fep = None
        if latex is True:
            print('\\end{document}')

        savename = ','.join([key + ':' + str(params[key]) for key in params['graph-annotation']])
        savename += '.png'
        annotation = str(dict((key, value) for (key, value) in params.items() if key in params['graph-annotation']))
        bar_chart(results, p_fep_fep=p_fep_fep, savename=savename, annotation=annotation, normalise_over_test_scene=params['normalise-over-test-scene'])

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
            for m in range(num_superordinate):
                subordinate_choices = sup.findall('.//subordinate')
                choice = subordinate_choices[np.random.randint(
                    len(subordinate_choices))]
                words_and_objects.append((label, choice.get('label')))

            sup_count += num_superordinate

        for basic in root.findall('.//basic-level'):
            label = basic.get('label')

            # add the appropriate number of words to the dictionary and choose
            # a random subordinate object
            for m in range(num_basic):
                subordinate_choices = basic.findall('.//subordinate')
                choice = subordinate_choices[np.random.randint(
                    len(subordinate_choices))]
                words_and_objects.append((label, choice.get('label')))

            basic_count += num_basic

        for sub in root.findall('.//subordinate'):

            label = sub.get('label')
            words_and_objects.extend([(label, label) for m in range(num_subordinate)])

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

    def generate_naturalistic_corpus(self, params):
        temp_corpus_path = 'temp_xt_corpus_'
        temp_corpus_path += datetime.now().isoformat() + '.dev'
        temp_corpus = open(temp_corpus_path, 'w')

        corpus = input.Corpus(params['corpus-path'])

        word_to_frequency_map = {}
        with open('lemma.al', 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            for row in reader:
                word_to_frequency_map[row[2]] = row[1]

        wn = nltk.corpus.WordNetCorpusReader(nltk.data.find('corpora/wordnet'), None)
        word_list = []

        sentence_count = 0

        while sentence_count < params['maxtime']:
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

                        # discard the highest level as it is too broad
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
            sup_features = [sup + '_f' + str(i) for i in range(params['num-features'])]
            sup_fs.extend(sup_features)
            for feature in sup_features:
                self.feature_to_level_map[feature] = 'superordinate'
            hierarchy_words.append(sup)
            for basic in hierarchy[sup]:
                hierarchy[sup][basic] = list(set(hierarchy[sup][basic]))
                if basic in hierarchy_words:
                    basic_to_delete.append((sup, basic))
                else:
                    word_to_list_of_feature_bundles_map[basic] = []
                    basic_features = [basic + '_f' + str(i) for i in range(params['num-features'])]
                    basic_fs.extend(basic_features)
                    for feature in basic_features:
                        self.feature_to_level_map[feature] = 'basic'
                    hierarchy_words.append(basic)
                    for sub in hierarchy[sup][basic]:
                        sub_features = [sub + '_f' + str(i) for i in range(params['num-features'])]
                        sub_fs.extend(sub_features)
                        for feature in sub_features:
                            self.feature_to_level_map[feature] = 'subordinate'
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
        corpus = input.Corpus(params['corpus-path'])
        lexicon = input.read_gold_lexicon(params['lexname'], params['beta'])

        sentence_count = 0

        while sentence_count < params['maxtime']:
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

        #np.random.shuffle(sup_fs)
        #np.random.shuffle(basic_fs)
        #np.random.shuffle(sub_fs)

        #sup = list(sup_fs)
        #basic = list(basic_fs)
        #sub = list(sub_fs)

        # assumption: all subordinate features are novel
        sub = ['sub_f' + str(i) for i in range(100000)]

        basic = ['basic' + str(i) for i in range(100000)]
        sup = ['sup_f' + str(i) for i in range(100000)]

        return temp_corpus_path, sup, basic, sub

    def generate_naturalistic_training_sets(self, sup, basic, sub, num_sets, num_sub_features=1, num_basic=1, num_sup=1):

        fep_features = []

        training_sets = {}
        training_sets['one example'] = []
        training_sets['three subordinate examples'] = []
        training_sets['three basic-level examples'] = []
        training_sets['three superordinate examples'] = []

        for i in range(num_sets):

            fep_sup = [sup.pop() for m in range(num_sup)]
            fep_basic = [basic.pop() for m in range(num_basic)]
            fep_sub = [sub.pop() for m in range(num_sub_features)]

            training_sets['one example'].append(
                [experimental_materials.UtteranceScenePair(
                    utterance='fep',
                    scene=fep_sup+fep_basic+fep_sub,
                    lexicon=self.gold_standard_lexicon,
                    probabilistic=False
                )]
            )

            training_sets['three subordinate examples'].append(
                [experimental_materials.UtteranceScenePair(
                    utterance='fep',
                    scene=fep_sup+fep_basic+fep_sub,
                    lexicon=self.gold_standard_lexicon,
                    probabilistic=False
                )] * 3
            )

            sub_2 = [sub.pop() for m in range(num_sub_features)]
            sub_3 = [sub.pop() for m in range(num_sub_features)]

            training_sets['three basic-level examples'].append(
                [
                    experimental_materials.UtteranceScenePair(
                        utterance='fep',
                        scene=fep_sup+fep_basic+fep_sub,
                        lexicon=self.gold_standard_lexicon,
                        probabilistic=False
                    ),
                    experimental_materials.UtteranceScenePair(
                        utterance='fep',
                        scene=fep_sup+fep_basic+sub_2,
                        lexicon=self.gold_standard_lexicon,
                        probabilistic=False
                    ),
                    experimental_materials.UtteranceScenePair(
                        utterance='fep',
                        scene=fep_sup+fep_basic+sub_3,
                        lexicon=self.gold_standard_lexicon,
                        probabilistic=False
                    )
                ]
            )

            basic_2 = [basic.pop() for m in range(num_basic)]
            basic_3 = [basic.pop() for m in range(num_basic)]

            sub_2 = [sub.pop() for m in range(num_sub_features)]
            sub_3 = [sub.pop() for m in range(num_sub_features)]

            training_sets['three superordinate examples'].append(
                [
                    experimental_materials.UtteranceScenePair(
                        utterance='fep',
                        scene=fep_sup+fep_basic+fep_sub,
                        lexicon=self.gold_standard_lexicon,
                        probabilistic=False
                    ),
                    experimental_materials.UtteranceScenePair(
                        utterance='fep',
                        scene=fep_sup+basic_2+sub_2,
                        lexicon=self.gold_standard_lexicon,
                        probabilistic=False
                    ),
                    experimental_materials.UtteranceScenePair(
                        utterance='fep',
                        scene=fep_sup+basic_3+sub_3,
                        lexicon=self.gold_standard_lexicon,
                        probabilistic=False
                    )
                ]
            )

            fep_features.append((fep_sup, fep_basic, fep_sub))

        return training_sets, fep_features

    def generate_naturalistic_test_sets(self, sup, basic, sub, fep_features, num_sets, num_sub_features=1, num_basic=1, num_sup=1):

        test_sets = []

        for i in range(num_sets):
            test_sets.append({})

            fep_sup = fep_features[i][0]
            fep_basic = fep_features[i][1]
            fep_sub = fep_features[i][2]

            test_sets[i]['subordinate matches'] = [
                experimental_materials.UtteranceScenePair(
                    utterance='fep',
                    scene=fep_sup+fep_basic+fep_sub,
                    lexicon=self.gold_standard_lexicon,
                    probabilistic=False
                )
            ]

            sub_1 = [sub.pop() for m in range(num_sub_features)]
            sub_2 = [sub.pop() for m in range(num_sub_features)]

            test_sets[i]['basic-level matches'] = [
                experimental_materials.UtteranceScenePair(
                    utterance='fep',
                    scene=fep_sup+fep_basic+sub_2,
                    lexicon=self.gold_standard_lexicon,
                    probabilistic=False
                )
            ]

            basic_1 = [basic.pop() for m in range(num_basic)]
            basic_2 = [basic.pop() for m in range(num_basic)]
            basic_3 = [basic.pop() for m in range(num_basic)]
            basic_4 = [basic.pop() for m in range(num_basic)]
            sub_1 = [sub.pop() for m in range(num_sub_features)]
            sub_2 = [sub.pop() for m in range(num_sub_features)]
            sub_3 = [sub.pop() for m in range(num_sub_features)]
            sub_4 = [sub.pop() for m in range(num_sub_features)]

            test_sets[i]['superordinate matches'] = [
                experimental_materials.UtteranceScenePair(
                    utterance='fep',
                    scene=fep_sup+basic_4+sub_4,
                    lexicon=self.gold_standard_lexicon,
                    probabilistic=False
                )
            ]

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

    def finalize(self, params):
        os.remove(self.corpus)
        os.remove(self.config_path)
        #os.remove(self.learner_path)

        if latex is True:
            print("""\end{document}""")


def calculate_generalisation_probability(learner, target_word, target_scene_meaning, method='cosine', std=0.0001, delta=0.0001, include_target=True, target_word_as_distribution=False, ratio_to_mean=False, log=False, include_unseen_features=True):
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

    def norm_prob(x, delta, mu, sigma):
        return mpmath.fsub(mpmath.ncdf(x+delta, mu=mu, sigma=sigma),
            mpmath.ncdf(x, mu=mu, sigma=sigma))

    def KL_prob(mu1, mu2, sigma1, sigma2):
        """
        Compute a 'probabilty' measure for the KL divergence of two univariate
        Gaussians.

        D_{KL} = \frac{(\mu_1-\mu_2)^2}{2\sigma^2_2} +
        \frac{1}{2}\left(\frac{\sigma_1^2}{\sigma_2^2} - 1 -
        \log \frac{\sigma_1^2}{\sigma_2^2}\right)

        p_{KL} = 1-\exp( -D_{KL} )

        """
        numer = mpmath.power(mpmath.fsub(mu1, mu2), 2)
        denom = mpmath.fmul(2, mpmath.power(sigma2, 2))
        kl = mpmath.fdiv(numer, denom)

        frac = mpmath.fdiv(mpmath.power(sigma1, 2), mpmath.power(sigma2, 2))
        term = mpmath.fsub(frac, 1)
        term = mpmath.fsub(term, mpmath.ln(frac))
        term = mpmath.fdiv(term, 2)

        kl = mpmath.fadd(kl, term)

        return 1 - mpmath.exp(mpmath.fneg(kl))

    def dirichlet_pdf(x, alpha):
        """
        x is a vector of n observations
        alpha is a vector of n concentration paramters

        """
        #assert np.sum(x) == 1.0
        return (mpmath.gamma(mpmath.fsum(alpha)) /
            reduce(mpmath.fmul, [mpmath.gamma(a) for a in alpha]) *
            reduce(mpmath.fmul, [mpmath.power(x[i], alpha[i]-1.0) for i in range(len(alpha))]))

    lexicon = learner.learned_lexicon()

    if method == 'no-word-averaging':

        total = cos(target_scene_meaning, lexicon.meaning(target_word))

    else:

        total = 0

        words = learner._wordsp.all_words(0)[:]

        sum_word_frequency = int(np.sum([learner._wordsp.frequency(w) for w in learner._wordsp.all_words(0)]))

        for word in words:

            p_w = mpmath.fdiv(learner._wordsp.frequency(word), sum_word_frequency)

            if method == 'cosine' or method == 'cosine-norm':

                p_fep_fep = mpmath.mpmathify(cos(lexicon.meaning(target_word), lexicon.meaning(target_word)))

                if include_unseen_features is False:
                    lexicon.meaning(word)._meaning_probs = \
                        dict((feature, lexicon.meaning(feature)) for feature in target_scene_meaning.seen_features())

                cos_y_w = mpmath.mpmathify(cos(target_scene_meaning, lexicon.meaning(word)))
                cos_target_w = mpmath.mpmathify(cos(lexicon.meaning(target_word), lexicon.meaning(word)))

                term = mpmath.fmul(mpmath.fmul(cos_y_w, cos_target_w), p_w)

                #print('\t', word, ':', '\tcos_y_w =', cos_y_w, '\tcos_target_w =', cos_target_w, '\tp(w) =', p_w,
                        #'\tterm:', cos_y_w * cos_target_w * p_w)

                if method == 'cosine-norm':

                    #TODO
                    # this normalisation requires too much time (it is an inner loop over words)
                    denom = np.sum([cos(lexicon.meaning(w), lexicon.meaning(word)) for w in words])
                    term = mpmath.fdiv(term, denom)
                    term = mpmath.fdiv(term, denom)

                total = mpmath.fadd(total, term)

            elif method == 'gaussian-norm':

                if log is True:
                    p_fep_fep = 1
                else:
                    p_fep_fep = 0

                for feature in lexicon.meaning(target_word).seen_features():
                    mean = lexicon.prob(target_word, feature)
                    prob = norm_prob(target_scene_meaning.prob(feature), delta, mean, std*mean)

                    if log is True:
                        prob = math.log(prob)
                        p_fep_fep = mpmath.fadd(p_fep_fep, prob)
                    else:
                        p_fep_fep = mpmath.fmul(p_fep_fep, prob)

                if log is True:
                    p_w = math.log(p_w)

                target_word_meaning = lexicon.meaning(target_word)

                if log is True:
                    y_factor = 0.0
                    target_factor = 0.0
                else:
                    y_factor = 1.0
                    target_factor = 1.0

                for feature in target_scene_meaning.seen_features():

                    mean = lexicon.prob(word, feature)
                    if ratio_to_mean is True:
                        prob = norm_prob(target_scene_meaning.prob(feature), delta, mean, std*mean)
                    else:
                        prob = norm_prob(target_scene_meaning.prob(feature), delta, mean, std)

                    if log is True:
                        prob = math.log(prob)
                        y_factor = mpmath.fadd(y_factor, prob)

                    else:
                        y_factor = mpmath.fmul(y_factor, prob)

                for feature in lexicon.seen_features(target_word):

                    if target_word_as_distribution is False:

                        mean = lexicon.prob(word, feature)

                        if ratio_to_mean is True:
                            prob = norm_prob(target_word_meaning.prob(feature), delta, mean, std*mean)
                        else:
                            prob = norm_prob(target_word_meaning.prob(feature), delta, mean, std)

                        if log is True:
                            prob = math.log(prob)
                            target_factor = mpmath.fadd(target_factor, prob)

                        else:
                            target_factor = mpmath.fmul(target_factor, prob)

                    else:

                        mu1 = lexicon.prob(word, feature)
                        mu2 = target_word_meaning.prob(feature)

                        if ratio_to_mean is True:
                            target_factor = mpmath.fmul(target_factor, KL_prob(mu1, mu2, std*mu1, std*mu2))
                        else:
                            target_factor = mpmath.fmul(target_factor, KL_prob(mu1, mu2, std, std))

                if log is True:
                    term = mpmath.fadd(mpmath.fadd(y_factor, target_factor), p_w)
                else:
                    term = mpmath.fmul(mpmath.fmul(y_factor, target_factor), p_w)

                total = mpmath.fadd(total, term)

            elif method == 'dirichlet-meanprob':

                x = []
                alpha = []

                for feature in lexicon.meaning(target_word).seen_features():

                    x.append(lexicon.prob(target_word, feature))
                    alpha.append(lexicon.prob(target_word, feature))

                x = np.array(x)
                x /= np.sum(x)

                p_fep_fep = dirichlet_pdf(x, alpha)

                x = []
                alpha = []

                seen = []

                for feature in target_scene_meaning.seen_features():

                    x.append(target_scene_meaning.prob(feature))
                    alpha.append(lexicon.prob(word, feature))

                    seen.append(feature)

                if include_unseen_features is True:
                    for feature in [f for f in learner._features if f not in seen]:
                        x.append(0.0)
                        alpha.append(lexicon.prob(word, feature))

                total = dirichlet_pdf(x, alpha)

                if latex is True:
                    print([filewriter.round_to_sig_digits(x, 4) for x in alpha], '&', filewriter.round_to_sig_digits(total, 4), '\\\\')

            elif method == 'dirichlet-assoc':

                x = []
                alpha = []

                for feature in lexicon.meaning(target_word).seen_features():

                    x.append(lexicon.prob(target_word, feature))
                    alpha.append(learner.association(target_word, feature) + learner.get_lambda())

                x = np.array(x)
                x /= np.sum(x)

                p_fep_fep = dirichlet_pdf(x, alpha)

                x = []
                alpha = []

                seen = []

                for feature in target_scene_meaning.seen_features():

                    x.append(target_scene_meaning.prob(feature))
                    alpha.append(learner.association(word, feature) + learner.get_lambda())

                    seen.append(feature)

                if include_unseen_features is True:
                    for feature in [f for f in learner._features if f not in seen]:

                        x.append(0.0)
                        alpha.append(learner.association(word, feature) + learner.get_lambda())

                total = dirichlet_pdf(x, alpha)

                if latex is True:
                    print([filewriter.round_to_sig_digits(x, 4) for x in alpha])

            elif method == 'simple':

                num_features = 0
                if log:
                    p_fep_fep = 0
                else:
                    p_fep_fep = 1

                for feature in lexicon.meaning(target_word).all_features():

                    if log:
                        p_fep_fep += np.log(lexicon.prob(target_word, feature))
                    else:
                        p_fep_fep *= lexicon.prob(target_word, feature)

                total = 1
                seen = []
                numbers = []
                features = []

                for feature in target_scene_meaning.all_features():

                    if log:
                        total += np.log(lexicon.prob(target_word, feature))
                    else:
                        total *= lexicon.prob(target_word, feature)
                        features.append(feature)

                        if latex is True:
                            print(feature+':', filewriter.round_to_sig_digits(lexicon.prob(target_word, feature), 4),'\\\\')

                    seen.append(feature)
                    num_features += 1

                for feature in [f for f in lexicon.meaning(target_word).all_features() if f not in seen]:

                    if log:
                        total += np.log((1 - lexicon.prob(target_word, feature)))
                    else:
                        total *= (1 - lexicon.prob(target_word, feature))
                        numbers.append(1 - lexicon.prob(target_word, feature))
                        features.append(feature)

                        if latex is True:
                            print(feature+':', filewriter.round_to_sig_digits(1-lexicon.prob(target_word, feature), 4),'\\\\')

                    seen.append(feature)
                    num_features += 1

                for feature in [f for f in learner._features if f not in seen]:

                    if log:
                        total += np.log((1 - lexicon.prob(target_word, feature)))
                    else:
                        total *= (1 - lexicon.prob(target_word, feature))
                        numbers.append(1 - lexicon.prob(target_word, feature))
                        features.append(feature)


                        if latex is True:
                            print(feature+':', filewriter.round_to_sig_digits(1-lexicon.prob(target_word, feature), 4),'\\\\')
                    num_features += 1

                if latex is True:
                    #print('&', filewriter.round_to_sig_digits(total, 4))
                    #print('&', p_fep_fep, '\\\\')
                    pass

            elif method == 'dirichlet-multiply':

                num_features = None
                p_fep_fep = None
                sub_factor, basic_factor, sup_factor = None, None, None
                other_features = learner._features

                for feature in target_scene_meaning.seen_features():
                    if feature.startswith('sub'):
                        sub_factor = learner.association(word, feature) + 1
                        denom = np.sum([learner.association(word, f) for f in other_features if f.startswith('sub')]) + learner.k_sub
                        sub_factor /= denom
                        if latex is True:
                            print(feature+':', filewriter.round_to_sig_digits(sub_factor, 4),'\\\\')
                    elif feature.startswith('bas'):
                        basic_factor = learner.association(word, feature) + 1
                        denom = np.sum([learner.association(word, f) for f in other_features if f.startswith('basic')]) + learner.k_basic
                        basic_factor /= denom
                        if latex is True:
                            print(feature+':', filewriter.round_to_sig_digits(basic_factor, 4),'\\\\')
                    elif feature.startswith('sup'):
                        sup_factor = learner.association(word, feature) + 1
                        denom = np.sum([learner.association(word, f) for f in other_features if f.startswith('sup')]) + learner.k_sup
                        sup_factor /= denom
                        if latex is True:
                            print(feature+':', filewriter.round_to_sig_digits(sup_factor, 4),'\\\\')
                    else:
                        raise NotImplementedError


                total = sub_factor * basic_factor * sup_factor

            elif method == 'dirichlet-add':

                num_features = None
                p_fep_fep = None
                sub_factor, basic_factor, sup_factor = None, None, None
                other_features = learner._features

                for feature in target_scene_meaning.seen_features():
                    if feature.startswith('sub'):
                        sub_factor = learner.association(word, feature) + 1
                        denom = np.sum([learner.association(word, f) for f in other_features if f.startswith('sub')]) + learner.k_sub
                        sub_factor /= denom
                        if latex is True:
                            print(feature+':', filewriter.round_to_sig_digits(sub_factor, 4),'\\\\')
                    elif feature.startswith('bas'):
                        basic_factor = learner.association(word, feature) + 1
                        denom = np.sum([learner.association(word, f) for f in other_features if f.startswith('basic')]) + learner.k_basic
                        basic_factor /= denom
                        if latex is True:
                            print(feature+':', filewriter.round_to_sig_digits(basic_factor, 4),'\\\\')
                    elif feature.startswith('sup'):
                        sup_factor = learner.association(word, feature) + 1
                        denom = np.sum([learner.association(word, f) for f in other_features if f.startswith('sup')]) + learner.k_sup
                        sup_factor /= denom
                        if latex is True:
                            print(feature+':', filewriter.round_to_sig_digits(sup_factor, 4),'\\\\')
                    else:
                        raise NotImplementedError

                total = sub_factor + basic_factor + sup_factor

            else:
                raise NotImplementedError

    return total, p_fep_fep, num_features


def bar_chart(results, p_fep_fep=None, savename=None, annotation=None, normalise_over_test_scene=True):

    conditions = ['one example',
        'three subordinate examples',
        'three basic-level examples',
        'three superordinate examples'
    ]

    ind = np.array([2*n for n in range(len(results))])
    width = 0.25

    nrows = int(np.ceil(len(results[conditions[0]]['subordinate matches']) / 2.0))

    if len(results[conditions[0]]['subordinate matches']) == 1:

         l0 = [np.mean(results[cond]['subordinate matches']) for cond in conditions]
         l1 = [np.mean(results[cond]['basic-level matches']) for cond in conditions]
         l2 = [np.mean(results[cond]['superordinate matches']) for cond in conditions]

         if normalise_over_test_scene is True:

             l0 = np.array(l0)
             l1 = np.array(l1)
             l2 = np.array(l2)

             if p_fep_fep is not None:
                 l0 = [num / p_fep_fep[conditions[i]] for (i, num) in enumerate(l0)]
                 l1 = [num / p_fep_fep[conditions[i]] for (i, num) in enumerate(l1)]
                 l2 = [num / p_fep_fep[conditions[i]] for (i, num) in enumerate(l2)]

             else:
                 denom = [np.mean(results[cond]['subordinate matches']) for cond in conditions]
                 denom = np.add(denom, [np.mean(results[cond]['basic-level matches']) for cond in conditions])
                 denom = np.add(denom, [np.mean(results[cond]['superordinate matches']) for cond in conditions])

                 try:
                     l0 /= denom
                     l1 /= denom
                     l2 /= denom
                 except ZeroDivisionError:
                     pass

             l0 = list(l0)
             l1 = list(l1)
             l2 = list(l2)

         width = 0.5
         fig = plt.figure()
         ax = fig.add_subplot(111)
         p0 = ax.bar(ind,l0,width,color='r')
         p1 = ax.bar(ind+width,l1,width,color='g')
         p2 = ax.bar(ind+2*width,l2,width,color='b')
         ax.set_ylabel("generalisation probability")
         ax.set_xlabel("condition")

         m = np.max(l0 + l1 + l2)

    else:
        fig, axes = plt.subplots(nrows=nrows, ncols=2, sharex=True, sharey=True)

        m = 0

        for i, ax in enumerate(axes.flat):

            if i == len(results[conditions[0]]['subordinate matches']):

                #ax.set_title('Average over all training-test sets', fontsize='small')

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

                    try:
                        l0 /= denom
                        l1 /= denom
                        l2 /= denom
                    except ZeroDivisionError:
                        pass

                    l0 = list(l0)
                    l1 = list(l1)
                    l2 = list(l2)

                p0 = ax.bar(ind,l0,width,color='r')
                p1 = ax.bar(ind+width,l1,width,color='g')
                p2 = ax.bar(ind+2*width,l2,width,color='b')

            elif i > len(results[conditions[0]]['subordinate matches']):
                pass

            else:
                ax.set_title('Training-test set ' + str(i+1), fontsize='small')

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

                    try:
                        l0 /= denom
                        l1 /= denom
                        l2 /= denom
                    except ZeroDivisionError:
                        pass

                    l0 = list(l0)
                    l1 = list(l1)
                    l2 = list(l2)

                else:
                    new_m = np.max(l0 + l1 + l2)
                    if new_m > m:
                        m = new_m

                p0 = ax.bar(ind,l0,width,color='r')
                p1 = ax.bar(ind+width,l1,width,color='g')
                p2 = ax.bar(ind+2*width,l2,width,color='b')

            xlabels = ('1', '3 sub.', '3 basic', '3 super.')
            ax.set_xticks(ind + 2 * width)
            ax.set_xticklabels(xlabels)

    #ax.set_ylabel("gen. prob.")
    #ax.set_xlabel("condition")
    if normalise_over_test_scene is True:
        plt.ylim((0,1))
    else:
        plt.ylim((0,float(m)))

    lgd = plt.legend( (p0, p1, p2), ('sub.', 'basic', 'super.'), loc='lower center', bbox_to_anchor=(0, -0.1, 1, 1), bbox_transform=plt.gcf().transFigure )

    title = "Generalization scores"

    if annotation is not None:
        title += '\n'+annotation

    #fig.suptitle(title)

    if savename is None:
        plt.show()
    else:
        plt.savefig(savename, bbox_extra_artists=(lgd,), bbox_inches='tight')

if __name__ == "__main__":
    e = GeneralisationExperiment()
    e.start()
