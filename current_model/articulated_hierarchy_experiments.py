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

def test_learner(gamma, k):

    # run seat experiments (3 levels)
    seat_training_sets = {}
    seat_training_sets['one example'] = []
    seat_training_sets['three subordinate examples'] = []
    seat_training_sets['three basic-level examples'] = []
    seat_training_sets['three superordinate examples'] = []

    seat_fep_features = ['seat11', 'seat21', 'seat31', 'instance1']
    seat_sub2_features = ['seat11', 'seat21', 'seat31', 'instance2']
    seat_sub3_features = ['seat11', 'seat21', 'seat31', 'instance3']

    seat_sub_match_features = ['seat11', 'seat21', 'seat31', 'instance4']

    seat_basic2_features = ['seat11', 'seat21', 'seat33', 'instance6']
    seat_basic3_features = ['seat11', 'seat21', 'seat34', 'instance7']

    seat_basic_match_features = ['seat11', 'seat21', 'seat35', 'instance8']

    seat_sup2_features = ['seat11', 'seat23', 'seat37', 'instance10']
    seat_sup3_features = ['seat11', 'seat24', 'seat38', 'instance11']

    seat_sup_match_features = ['seat11', 'seat25', 'seat39', 'instance12']

    seat_training_sets['one example'].append(
        [experimental_materials.UtteranceScenePair(
            utterance='fep',
            scene=seat_fep_features,
            probabilistic=False
        )]
    )

    seat_training_sets['three subordinate examples'].append(
        [experimental_materials.UtteranceScenePair(
            utterance='fep',
            scene=seat_fep_features,
            probabilistic=False
        ),
        experimental_materials.UtteranceScenePair(
            utterance='fep',
            scene=seat_sub2_features,
            probabilistic=False
        ),
        experimental_materials.UtteranceScenePair(
            utterance='fep',
            scene=seat_sub3_features,
            probabilistic=False
        )]
    )

    seat_training_sets['three basic-level examples'].append(
        [
            experimental_materials.UtteranceScenePair(
                utterance='fep',
                scene=seat_fep_features,
                probabilistic=False
            ),
            experimental_materials.UtteranceScenePair(
                utterance='fep',
                scene=seat_basic2_features,
                probabilistic=False
            ),
            experimental_materials.UtteranceScenePair(
                utterance='fep',
                scene=seat_basic3_features,
                probabilistic=False
            )
        ]
    )

    seat_training_sets['three superordinate examples'].append(
        [
            experimental_materials.UtteranceScenePair(
                utterance='fep',
                scene=seat_fep_features,
                probabilistic=False
            ),
            experimental_materials.UtteranceScenePair(
                utterance='fep',
                scene=seat_sup2_features,
                probabilistic=False
            ),
            experimental_materials.UtteranceScenePair(
                utterance='fep',
                scene=seat_sup3_features,
                probabilistic=False
            )
        ]
    )

    seat_test_sets = []
    seat_test_sets.append({})

    i = 0

    seat_test_sets[i]['subordinate matches'] = [
        experimental_materials.UtteranceScenePair(
            details='subordinate test object 1',
            utterance='fep',
            scene=seat_sub_match_features,
            probabilistic=False
        )
    ]

    seat_test_sets[i]['basic-level matches'] = [
        experimental_materials.UtteranceScenePair(
            details='basic-level test object 1',
            utterance='fep',
            scene=seat_basic_match_features,
            probabilistic=False
        )
    ]

    seat_test_sets[i]['superordinate matches'] = [
        experimental_materials.UtteranceScenePair(
            details='superordinate test object 1',
            utterance='fep',
            scene=seat_sup_match_features,
            probabilistic=False
        )
    ]


    conds = ['one example', 'three subordinate examples',
            'three basic-level examples',
            'three superordinate examples']

    seat_results = {}

    for condition in conds:

        seat_results[condition] = {}

        for i, training_set in enumerate(seat_training_sets[condition]):

            learner = learn.Learner(gamma, k)

            for trial in training_set:

                learner.process_pair(trial.utterance(), trial.scene(), './')

            print('============================================')
            print('Condition: ' + condition)
            print(str(learner._learned_lexicon.meaning('fep')))

            for cond in seat_test_sets[i]:

                print("\tMatch: " + cond)

                take_average = 0
                count = 0

                for j in range(len(seat_test_sets[i][cond])):
                    test_scene = seat_test_sets[i][cond][j]
                    word = test_scene.utterance()[0]
                    scene = test_scene.scene()

                    gen_prob = learner.generalisation_prob(word, scene)

                    take_average = mpmath.fadd(take_average, gen_prob)
                    count += 1

                # average the test trials within the category
                gen_prob = mpmath.fdiv(take_average, count)

                try:
                    seat_results[condition][cond].append(gen_prob)
                except KeyError:
                    seat_results[condition][cond] = []
                    seat_results[condition][cond].append(gen_prob)

    #pprint.pprint(seat_results)

    savename  = 'seats_and_containers_experiments/'
    savename += 'seats_gamma_' + str(gamma) + '_k_' + str(k) + '.png'
    annotation = 'gamma = ' + str(gamma) + '; k = ' + str(k)
    bar_chart(seat_results, savename=savename, annotation=annotation,
        normalise_over_test_scene=True,
        subtract_null_hypothesis=1./k**4)

    # run container experiments (3 levels)
    container_training_sets = {}
    container_training_sets['one example'] = []
    container_training_sets['three subordinate examples'] = []
    container_training_sets['three basic-level examples'] = []
    container_training_sets['three superordinate examples'] = []

    container_fep_features = ['container11', 'container21', 'container31', 'container41', 'container51', 'instance11', 'instance21']
    container_sub2_features = ['container11', 'container21', 'container31', 'container41', 'container51', 'instance12', 'instance22']
    container_sub3_features = ['container11', 'container21', 'container31', 'container41', 'container51', 'instance13', 'instance23']

    container_sub_match_features = ['container11', 'container21', 'container31', 'container41', 'container51', 'instance14', 'instance24']

    container_basic2_features = ['container11', 'container21', 'container31', 'container43', 'container53', 'instance16', 'instance26']
    container_basic3_features = ['container11', 'container21', 'container31', 'container44', 'container54', 'instance17', 'instance27']

    container_basic_match_features = ['container11', 'container21', 'container31', 'container45', 'container55', 'instance18', 'instance28']

    container_sup2_features = ['container11', 'container23', 'container33', 'container47', 'container57', 'instance110', 'instance210']
    container_sup3_features = ['container11', 'container24', 'container34', 'container48', 'container58', 'instance111', 'instance211']

    container_sup_match_features = ['container11', 'container25', 'container35', 'container49', 'container59', 'instance112', 'instance212']

    container_training_sets['one example'].append(
        [experimental_materials.UtteranceScenePair(
            utterance='fep',
            scene=container_fep_features,
            probabilistic=False
        )]
    )

    container_training_sets['three subordinate examples'].append(
        [experimental_materials.UtteranceScenePair(
            utterance='fep',
            scene=container_fep_features,
            probabilistic=False
        ),
        experimental_materials.UtteranceScenePair(
            utterance='fep',
            scene=container_sub2_features,
            probabilistic=False
        ),
        experimental_materials.UtteranceScenePair(
            utterance='fep',
            scene=container_sub3_features,
            probabilistic=False
        )]
    )

    container_training_sets['three basic-level examples'].append(
        [
            experimental_materials.UtteranceScenePair(
                utterance='fep',
                scene=container_fep_features,
                probabilistic=False
            ),
            experimental_materials.UtteranceScenePair(
                utterance='fep',
                scene=container_basic2_features,
                probabilistic=False
            ),
            experimental_materials.UtteranceScenePair(
                utterance='fep',
                scene=container_basic3_features,
                probabilistic=False
            )
        ]
    )

    container_training_sets['three superordinate examples'].append(
        [
            experimental_materials.UtteranceScenePair(
                utterance='fep',
                scene=container_fep_features,
                probabilistic=False
            ),
            experimental_materials.UtteranceScenePair(
                utterance='fep',
                scene=container_sup2_features,
                probabilistic=False
            ),
            experimental_materials.UtteranceScenePair(
                utterance='fep',
                scene=container_sup3_features,
                probabilistic=False
            )
        ]
    )

    container_test_sets = []
    container_test_sets.append({})

    i = 0

    container_test_sets[i]['subordinate matches'] = [
        experimental_materials.UtteranceScenePair(
            details='subordinate test object 1',
            utterance='fep',
            scene=container_sub_match_features,
            probabilistic=False
        )
    ]

    container_test_sets[i]['basic-level matches'] = [
        experimental_materials.UtteranceScenePair(
            details='basic-level test object 1',
            utterance='fep',
            scene=container_basic_match_features,
            probabilistic=False
        )
    ]

    container_test_sets[i]['superordinate matches'] = [
        experimental_materials.UtteranceScenePair(
            details='superordinate test object 1',
            utterance='fep',
            scene=container_sup_match_features,
            probabilistic=False
        )
    ]


    conds = ['one example', 'three subordinate examples',
            'three basic-level examples',
            'three superordinate examples']

    container_results = {}

    for condition in conds:

        container_results[condition] = {}

        for i, training_set in enumerate(container_training_sets[condition]):

            learner = learn.Learner(gamma, k)

            for trial in training_set:

                learner.process_pair(trial.utterance(), trial.scene(), './')

            print('============================================')
            print('Condition: ' + condition)

            for cond in container_test_sets[i]:

                take_average = 0
                count = 0

                for j in range(len(container_test_sets[i][cond])):
                    test_scene = container_test_sets[i][cond][j]
                    word = test_scene.utterance()[0]
                    scene = test_scene.scene()


                    gen_prob = learner.generalisation_prob(word, scene)

                    take_average = mpmath.fadd(take_average, gen_prob)
                    count += 1

                # average the test trials within the category
                gen_prob = mpmath.fdiv(take_average, count)

                try:
                    container_results[condition][cond].append(gen_prob)
                except KeyError:
                    container_results[condition][cond] = []
                    container_results[condition][cond].append(gen_prob)


    savename  = 'seats_and_containers_experiments/'
    savename += 'containers_gamma_' + str(gamma) + '_k_' + str(k) + '.png'
    annotation = 'gamma = ' + str(gamma) + '; k = ' + str(k)
    bar_chart(container_results, savename=savename, annotation=annotation,
        normalise_over_test_scene=True,
        subtract_null_hypothesis=1./k**4)

def bar_chart(results, savename=None, annotation=None,
        normalise_over_test_scene=True, subtract_null_hypothesis=None):

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

        elif subtract_null_hypothesis is not None:

            l0 = np.array(l0)
            l1 = np.array(l1)
            l2 = np.array(l2)

            l0 -= subtract_null_hypothesis
            l1 -= subtract_null_hypothesis
            l2 -= subtract_null_hypothesis

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


                elif subtract_null_hypothesis is not None:

                    l0 = np.array(l0)
                    l1 = np.array(l1)
                    l2 = np.array(l2)

                    print(l0)
                    raw_input()

                    l0 -= subtract_null_hypothesis
                    l1 -= subtract_null_hypothesis
                    l2 -= subtract_null_hypothesis

                    print(l0)
                    raw_input()

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

    ks = [0.1, 0.5, 1, 2, 5, 10, 20]
    gammas = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20]


    #for k in ks:
    #    for gamma in gammas:
    #        test_learner(gamma, k)

    test_learner(0.5, 5)
