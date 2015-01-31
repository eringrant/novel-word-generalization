#!/usr/bin/python

# python2 xt_experiments.py -o test_results.csv -c xt_experiments.cfg -n 1

from __future__ import print_function, division

import csv
from datetime import datetime
import gc
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
import pprint
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

latex = False

class GeneralisationExperiment(experiment.Experiment):

    def setup(self, params, rep):
        """ Runs in each child process. """

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


        try:
            L = 1/params['beta']
        except ZeroDivisionError:
            L = 0

        self.config_path = experimental_materials.write_config_file(
            config_filename,
            dummy=params['dummy'],
            forget=self.forget,
            forget_decay=self.forget_decay,
            novelty=self.novelty,
            novelty_decay=self.novelty_decay,
            beta=params['beta'],
            L=L,
            power=params['power'],
            alpha=params['alpha'],
            epsilon=params['epsilon'],
            maxtime=params['maxtime']
        )

        self.learner_config = learnconfig.LearnerConfig(self.config_path)

        # generate features
        self.sub = ['sub_f' + str(i) for i in range(100000)]
        self.basic = ['basic' + str(i) for i in range(100000)]
        self.sup = ['sup_f' + str(i) for i in range(100000)]

        self.subordinate_feature_to_basic_level = {}
        self.basic_feature_to_subordinates = {}

        self.learner_path = params['learner-path']
        self.learner_path += datetime.now().isoformat() + '.pkl'

        # create the learner
        stopwords = []
        self.learner = learn.Learner(params['lexname'], self.learner_config,
            stopwords,
            k_sub=params['k-sub'], k_basic=params['k-basic'],
            k_sup=params['k-sup'],
            alpha_sub=params['alpha-sub'], alpha_basic=params['alpha-basic'],
            alpha_sup=params['alpha-sup'])
        learner_dump = open(self.learner_path, "wb")
        pickle.dump(self.learner, learner_dump)
        learner_dump.close()
        self.learner = None

        self.training_sets, fep_features = \
            self.generate_training_sets(
                self.sup, self.basic, self.sub, params['num-sets'],
                num_sub_features=params['num-sub-features'],
                num_basic_features=params['num-basic-features'],
                num_sup_features=params['num-sup-features']
            )

        self.test_sets = \
            self.generate_test_sets(
                self.sup, self.basic, self.sub,
                fep_features, params['num-sets'],
                num_sub_features=params['num-sub-features'],
                num_basic_features=params['num-basic-features'],
                num_sup_features=params['num-sup-features'])

        return True

    def iterate(self, params, rep, n):
        """ Runs in each child process for n iterations. """

        results = {}

        if latex is True:
            print('\\documentclass{article}')
            print('\\usepackage[landscape]{geometry}')
            print('\\usepackage{underscore}')
            print('\\newcommand{\specialcell}[2][l]{\\begin{tabular}[#1]{@{}l@{}}#2\\end{tabular}}')
            print('\\begin{document}')

        conds = ['one example', 'three subordinate examples',
                'three basic-level examples',
                'three superordinate examples']

        print('========================================')
        print('Parameters:')
        pprint.pprint(dict((k, params[k]) for k in params['graph-annotation']))
        print('')

        assert set(self.training_sets.keys()) == set(conds)

        for condition in conds:

            results[condition] = {}

            for i, training_set in enumerate(self.training_sets[condition]):

                learner_dump = open(self.learner_path, "rb")
                self.learner = pickle.load(learner_dump)
                learner_dump.close()

                if latex is True:
                    print('\\subsection*{'+condition+'}')
                    print('\\begin{tabular}{l c c c}')
                    print('& training trial 1 & trial 2 & trial 3 \\\\')

                for trial in training_set:

                    sub_features = []
                    basic_features = []
                    sup_features = []

                    for feature in trial.scene():
                        if feature.startswith('sub'):
                            sub_features.append(feature)
                        elif feature.startswith('bas'):
                            basic_features.append(feature)
                        elif feature.startswith('sup'):
                            sup_features.append(feature)

                    for feature in basic_features:
                        try:
                            self.basic_feature_to_subordinates[feature].extend(sub_features)
                        except KeyError:
                            self.basic_feature_to_subordinates[feature] = []
                            self.basic_feature_to_subordinates[feature].extend(sub_features)

                    for feature in sub_features:
                        try:
                            self.subordinate_feature_to_basic_level[feature].extend(basic_features)
                        except KeyError:
                            self.subordinate_feature_to_basic_level[feature] = []
                            self.subordinate_feature_to_basic_level[feature].extend(basic_features)

                    self.learner.process_pair(trial.utterance(), trial.scene(),
                                              params['path'],
                                              self.subordinate_feature_to_basic_level,
                                              self.basic_feature_to_subordinates)

                    if latex is True:
                        print('&', str(trial.scene())[1:-1])

                print('============================================')
                print('Condition: ' + condition)
                print(self.learner._learned_lexicon.meaning('fep'))

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
                        #meaning = wmmapping.Meaning(self.learner._beta)
                        meaning = wmmapping.Meaning(0.1, 0.1, 0.1, 1, 1, 1)
                        if params['basic-level-bias'] is not None:
                            # deprecated
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

                        print('\t' + cond + ':')

                        gen_prob, feature_count = calculate_generalisation_probability(
                            self.learner, word, meaning,
                            method=params['calculation-type'],
                            log=params['log'],
                            alpha_sub=params['alpha-sub'],
                            alpha_basic=params['alpha-basic'],
                            alpha_sup=params['alpha-sup']
                            )

                        if params['include-unseen-features'] is True:
                            # account for all features accross training, test
                            # there should be 11 sub, 7 basic, 1 sup

                            sub_count = 1
                            basic_count = 1
                            sup_count = 1

                            while sub_count < 11:
                                if latex is True:
                                    print('unseen ftr:', filewriter.round_to_sig_digits((1-1./params['k-sub']), 4), '\\\\')

                                if params['calculation-type'] == 'feature-bundles':
                                    gen_prob *=\
                                    (1-self.learner._learned_lexicon._unseen_sub)
                                else:
                                    gen_prob *= 0.5
                                sub_count += 1

                            while basic_count < 7:
                                if latex is True:
                                    print('unseen ftr:', filewriter.round_to_sig_digits((1-1./params['k-basic']), 4), '\\\\')

                                if params['calculation-type'] == 'feature-bundles':
                                    gen_prob *=\
                                    (1-self.learner._learned_lexicon._unseen_basic)
                                else:
                                    gen_prob *= 0.5
                                basic_count += 1

                        if latex is True:
                            print("""}""")
                            print('&', filewriter.round_to_sig_digits(gen_prob, 4), '\\\\')
                            print("""\hline""")

                        print('\t\tgeneralisation probability:', gen_prob)

                        take_average = mpmath.fadd(take_average, gen_prob)
                        count += 1

                        if latex is True:
                            print('\n')

                    # average the test trials within the category
                    #gen_prob = mpmath.fdiv(take_average, count)
                    assert count == 1

                    try:
                        results[condition][cond].append(gen_prob)
                    except KeyError:
                        results[condition][cond] = []
                        results[condition][cond].append(gen_prob)

                if latex is True:
                    print('\\end{tabular}')
                    print('\n')

                # reset the learner after each training set
                self.learner = None

        if latex is True:
            print('\\end{document}')

        savename = ','.join([key + ':' + str(params[key]) for key in params['graph-annotation']])
        savename += '.png'
        annotation = str(dict((key, value) for (key, value) in params.items() if key in params['graph-annotation']))
        bar_chart(results, savename=savename, annotation=annotation, normalise_over_test_scene=params['normalise-over-test-scene'])

        os.remove(self.config_path)
        os.remove(self.learner_path)

        gc.collect()

        return results

    def generate_training_sets(self, sup, basic, sub, num_sets, num_sub_features=1, num_basic_features=1, num_sup_features=1):

        fep_features = []

        training_sets = {}
        training_sets['one example'] = []
        training_sets['three subordinate examples'] = []
        training_sets['three basic-level examples'] = []
        training_sets['three superordinate examples'] = []

        for i in range(num_sets):

            # features of the subordinate instance / test match
            fep_sup = [sup.pop() for m in range(num_sup_features)]
            fep_basic = [basic.pop() for m in range(num_basic_features)]
            fep_sub = [sub.pop() for m in range(num_sub_features)]

            for f in fep_sub:
                try:
                    self.subordinate_feature_to_basic_level[f].extend(fep_basic)
                except KeyError:
                    self.subordinate_feature_to_basic_level[f] = []
                    self.subordinate_feature_to_basic_level[f].extend(fep_basic)

            training_sets['one example'].append(
                [experimental_materials.UtteranceScenePair(
                    utterance='fep',
                    scene=fep_sup+fep_basic+fep_sub,
                    probabilistic=False
                )]
            )

            training_sets['three subordinate examples'].append(
                [experimental_materials.UtteranceScenePair(
                    utterance='fep',
                    scene=fep_sup+fep_basic+fep_sub,
                    probabilistic=False
                )] * 3
            )

            sub_2 = [sub.pop() for m in range(num_sub_features)]
            sub_3 = [sub.pop() for m in range(num_sub_features)]

            for f in sub_2:
                try:
                    self.subordinate_feature_to_basic_level[f].extend(fep_basic)
                except KeyError:
                    self.subordinate_feature_to_basic_level[f] = []
                    self.subordinate_feature_to_basic_level[f].extend(fep_basic)

            for f in sub_3:
                try:
                    self.subordinate_feature_to_basic_level[f].extend(fep_basic)
                except KeyError:
                    self.subordinate_feature_to_basic_level[f] = []
                    self.subordinate_feature_to_basic_level[f].extend(fep_basic)

            training_sets['three basic-level examples'].append(
                [
                    experimental_materials.UtteranceScenePair(
                        utterance='fep',
                        scene=fep_sup+fep_basic+fep_sub,
                        probabilistic=False
                    ),
                    experimental_materials.UtteranceScenePair(
                        utterance='fep',
                        scene=fep_sup+fep_basic+sub_2,
                        probabilistic=False
                    ),
                    experimental_materials.UtteranceScenePair(
                        utterance='fep',
                        scene=fep_sup+fep_basic+sub_3,
                        probabilistic=False
                    )
                ]
            )

            basic_2 = [basic.pop() for m in range(num_basic_features)]
            sub_4 = [sub.pop() for m in range(num_sub_features)]

            for f in sub_4:
                try:
                    self.subordinate_feature_to_basic_level[f].extend(basic_2)
                except KeyError:
                    self.subordinate_feature_to_basic_level[f] = []
                    self.subordinate_feature_to_basic_level[f].extend(basic_2)

            basic_3 = [basic.pop() for m in range(num_basic_features)]
            sub_5 = [sub.pop() for m in range(num_sub_features)]

            for f in sub_5:
                try:
                    self.subordinate_feature_to_basic_level[f].extend(basic_3)
                except KeyError:
                    self.subordinate_feature_to_basic_level[f] = []
                    self.subordinate_feature_to_basic_level[f].extend(basic_3)

            training_sets['three superordinate examples'].append(
                [
                    experimental_materials.UtteranceScenePair(
                        utterance='fep',
                        scene=fep_sup+fep_basic+fep_sub,
                        probabilistic=False
                    ),
                    experimental_materials.UtteranceScenePair(
                        utterance='fep',
                        scene=fep_sup+basic_2+sub_4,
                        probabilistic=False
                    ),
                    experimental_materials.UtteranceScenePair(
                        utterance='fep',
                        scene=fep_sup+basic_3+sub_5,
                        probabilistic=False
                    )
                ]
            )

            fep_features.append((fep_sup, fep_basic, fep_sub))

        return training_sets, fep_features

    def generate_test_sets(self, sup, basic, sub, fep_features, num_sets, num_sub_features=1, num_basic_features=1, num_sup_features=1):

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
                    probabilistic=False
                )
            ]

            sub_2 = [sub.pop() for m in range(num_sub_features)]

            for f in sub_2:
                try:
                    self.subordinate_feature_to_basic_level[f].extend(fep_basic)
                except KeyError:
                    self.subordinate_feature_to_basic_level[f] = []
                    self.subordinate_feature_to_basic_level[f].extend(fep_basic)

            test_sets[i]['basic-level matches'] = [
                experimental_materials.UtteranceScenePair(
                    utterance='fep',
                    scene=fep_sup+fep_basic+sub_2,
                    probabilistic=False
                )
            ]

            basic_2 = [basic.pop() for m in range(num_basic_features)]
            sub_3 = [sub.pop() for m in range(num_sub_features)]

            for f in sub_3:
                try:
                    self.subordinate_feature_to_basic_level[f].extend(basic_2)
                except KeyError:
                    self.subordinate_feature_to_basic_level[f] = []
                    self.subordinate_feature_to_basic_level[f].extend(basic_2)

            test_sets[i]['superordinate matches'] = [
                experimental_materials.UtteranceScenePair(
                    utterance='fep',
                    scene=fep_sup+basic_2+sub_3,
                    probabilistic=False
                )
            ]

        return test_sets

    def finalize(self, params):
        if latex is True:
            print("""\end{document}""")


def calculate_generalisation_probability(learner, target_word,
        target_scene_meaning, method='cosine', log=False,
        alpha_sub=1, alpha_basic=1, alpha_sup=1):

    def cos(one, two):
        beta_sub = learner.k_sub
        beta_basic = learner.k_basic
        beta_sup = learner.k_sup
        return evaluate.calculate_similarity(beta_sub, beta_basic, beta_sup, one, two, CONST.COS)

    lexicon = learner.learned_lexicon()
    feature_count = 0

    if method == 'cosine':

        total = cos(target_scene_meaning, lexicon.meaning(target_word))

    elif method == 'simple':

        if log:
            total = 0
        else:
            total = 1

        for feature in target_scene_meaning.all_features():

            if log:
                total += np.log(lexicon.prob(target_word, feature))
            else:
                total *= lexicon.prob(target_word, feature)

                if latex is True:
                    print(feature+':', filewriter.round_to_sig_digits(lexicon.prob(target_word, feature), 4),'\\\\')

            feature_count += 1

    elif method == 'independent-features':

        total = 1
        alpha = 1
        beta = 1

        for feature in target_scene_meaning.seen_features():

            if feature in learner._features:
                denom = learner._wordsp.frequency(target_word) + alpha + beta
                total *= (learner.association(target_word, feature) + alpha) / denom

            else:
                total *= alpha / (alpha+beta)

            feature_count += 1

    elif method == 'feature-bundles':

        sup_factor = 1
        basic_factor = 1
        sub_factor = 1

        fep_features = learner._learned_lexicon.seen_features(target_word)

        fep_sub_features = []
        fep_basic_features = []
        fep_sup_features = []

        for feature in fep_features:
            if feature.startswith('sub'):
                fep_sub_features.append(feature)
            elif feature.startswith('bas'):
                fep_basic_features.append(feature)
            elif feature.startswith('sup'):
                fep_sup_features.append(feature)

        target_sub_features = []
        target_basic_features = []
        target_sup_features = []

        for feature in target_scene_meaning.seen_features():
            if feature.startswith('sub'):
                target_sub_features.append(feature)
            elif feature.startswith('bas'):
                target_basic_features.append(feature)
            elif feature.startswith('sup'):
                target_sup_features.append(feature)

        basic = None

        for feature in target_scene_meaning.seen_features():
            if feature.startswith('bas'):
                basic = feature

        for feature in target_scene_meaning.seen_features():

            if feature.startswith('sub'):

                sub_factor = learner._learned_lexicon.meaning(target_word).prob(feature,
                        basic_feature=basic)

                if latex is True:
                    print(feature+':', filewriter.round_to_sig_digits(sub_factor, 4),'\\\\')

            elif feature.startswith('bas'):

                basic_factor = learner._learned_lexicon.meaning(target_word).prob(feature)

                if latex is True:
                    print(feature+':', filewriter.round_to_sig_digits(basic_factor, 4),'\\\\')

            elif feature.startswith('sup'):

                sup_factor = learner._learned_lexicon.meaning(target_word).prob(feature)

                if latex is True:
                    print(feature+':', filewriter.round_to_sig_digits(sup_factor, 4),'\\\\')

            else:
                raise NotImplementedError

            feature_count += 1

        print('\t\tsub feature:', sub_factor, '\n\t\tbasic feature:', basic_factor,
                '\n\t\tsup feature:', sup_factor)

        total = sub_factor * basic_factor * sup_factor

    else:
        raise NotImplementedError

    return total, feature_count


def bar_chart(results, savename=None, annotation=None, normalise_over_test_scene=True):

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
