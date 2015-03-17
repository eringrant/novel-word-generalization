#!/usr/bin/python

# python2 articulated_hierarchy_experiments.py -o test_results.csv -c articulated_hierarchy_experiments.cfg -n 1

from __future__ import print_function, division

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mpmath; mpmath.mp.dps = 100
import numpy as np
import pprint

import learn

import experiment
import experimental_materials

latex = False

class GeneralisationExperiment(experiment.Experiment):

    def setup(self, params, rep):
        """ Runs in each child process. """
        gamma = params['gamma']
        k = params['k']

        training_sets, test_sets = generate_training_and_test_sets(
            params['num-sup-levels'],
            params['num-basic-levels'],
            params['num-sub-levels'],
            params['num-instance-levels'],
            params['num-features']
        )

        conds = ['one example',
                'three subordinate examples',
                'three basic-level examples',
                'three superordinate examples'
        ]

        results = {}

        for condition in conds:

            results[condition] = {}

            for training_set in training_sets[condition]:

                learner = learn.Learner(gamma, k)

                for trial in training_set:

                    learner.process_pair(trial.utterance(), trial.scene(), './')

                print('============================================')
                print('Condition: ' + condition)
                print(learner._learned_lexicon.meaning('fep'))

                for cond in test_sets:

                    print("\tMatch: " + cond)

                    take_average = 0
                    count = 0

                    for j in range(len(test_sets[cond])):
                        test_scene = test_sets[cond][j]
                        word = test_scene.utterance()[0]
                        scene = test_scene.scene()

                        #import pdb; pdb.set_trace()
                        gen_prob = learner.generalisation_prob(word, scene)

                        take_average = mpmath.fadd(take_average, gen_prob)
                        count += 1

                    # average the test trials within the category
                    gen_prob = mpmath.fdiv(take_average, count)

                    try:
                        results[condition][cond].append(gen_prob)
                    except KeyError:
                        results[condition][cond] = []
                        results[condition][cond].append(gen_prob)

        #pprint.pprint(results)

        savename  = 'articulated_hierarchy_experiments/'
        savename += 'gamma_' + str(gamma) + ',k_' + str(k)
        savename += ',n_sup_lvls_' + str(params['num-sup-levels'])
        savename += ',n_basic_lvls_' + str(params['num-basic-levels'])
        savename += ',n_sub_lvls_' + str(params['num-sub-levels'])
        savename += ',n_inst_lvls_' + str(params['num-instance-levels'])
        savename += '.png'

        null_hypothesis = params['num-sup-levels'] + \
            params['num-basic-levels'] + params['num-sub-levels'] + \
            params['num-instance-levels'] + params['num-sub-levels']
        null_hypothesis *= params['num-features']
        null_hypothesis = mpmath.power(k, null_hypothesis)
        null_hypothesis = mpmath.fdiv(1., null_hypothesis)

        bar_chart(results, savename=savename,
            normalise_over_test_scene=True,
            subtract_null_hypothesis=null_hypothesis)

def generate_training_and_test_sets(num_sup_levels, num_basic_levels, num_sub_levels,
        num_instance_levels, num_features):
    """

    """
    # generate training examples
    fep_sup_features = []
    fep_bas_features = []
    fep_sub_features = []
    fep_instance_features = []

    for n in range(1, num_sup_levels+1):
        fep_sup_features += ['fsup' + str(n) + str(i) + '1' for i in range(1, num_features+1)]
    for n in range(1, num_basic_levels+1):
        fep_bas_features += ['fbas' + str(n) + str(i) + '1' for i in range(1, num_features+1)]
    for n in range(1, num_sub_levels+1):
        fep_sub_features += ['fsub' + str(n) + str(i) + '1' for i in range(1, num_features+1)]
    for n in range(1, num_instance_levels+1):
        fep_instance_features += ['finstance' + str(n) + str(i) + '1' for i in range(1, num_features+1)]

    fep_features = fep_sup_features + fep_bas_features + fep_sub_features + fep_instance_features

    sub_example_2_features = fep_sup_features[:] + fep_bas_features[:] + fep_sub_features[:]
    for n in range(1, num_instance_levels+1):
        sub_example_2_features += ['finstance' + str(n) + str(i) + '2' for i in range(1, num_features+1)]
    sub_example_3_features = fep_sup_features[:] + fep_bas_features[:] + fep_sub_features[:]
    for n in range(1, num_instance_levels+1):
        sub_example_3_features += ['finstance' + str(n) + str(i) + '3' for i in range(1, num_features+1)]

    basic_example_2_features = fep_sup_features[:] + fep_bas_features[:]
    for n in range(1, num_sub_levels+1):
        basic_example_2_features += ['fsub' + str(n) + str(i) + '2' for i in range(1, num_features+1)]
    for n in range(1, num_instance_levels+1):
        basic_example_2_features += ['finstance' + str(n) + str(i) + '4' for i in range(1, num_features+1)]
    basic_example_3_features = fep_sup_features[:] + fep_bas_features[:]
    for n in range(1, num_sub_levels+1):
        basic_example_3_features += ['fsub' + str(n) + str(i) + '3' for i in range(1, num_features+1)]
    for n in range(1, num_instance_levels+1):
        basic_example_3_features += ['finstance' + str(n) + str(i) + '5' for i in range(1, num_features+1)]

    sup_example_2_features = fep_sup_features[:]
    for n in range(1, num_basic_levels+1):
        sup_example_2_features += ['fbasic' + str(n) + str(i) + '2' for i in range(1, num_features+1)]
    for n in range(1, num_sub_levels+1):
        sup_example_2_features += ['fsub' + str(n) + str(i) + '4' for i in range(1, num_features+1)]
    for n in range(1, num_instance_levels+1):
        sup_example_2_features += ['finstance' + str(n) + str(i) + '6' for i in range(1, num_features+1)]
    sup_example_3_features = fep_sup_features[:]
    for n in range(1, num_basic_levels+1):
        sup_example_3_features += ['fbasic' + str(n) + str(i) + '3' for i in range(1, num_features+1)]
    for n in range(1, num_sub_levels+1):
        sup_example_3_features += ['fsub' + str(n) + str(i) + '5' for i in range(1, num_features+1)]
    for n in range(1, num_instance_levels+1):
        sup_example_3_features += ['finstance' + str(n) + str(i) + '7' for i in range(1, num_features+1)]

    training_sets = {}
    training_sets['one example'] = []
    training_sets['three subordinate examples'] = []
    training_sets['three basic-level examples'] = []
    training_sets['three superordinate examples'] = []

    training_sets['one example'].append(
        [experimental_materials.UtteranceScenePair(
            utterance='fep',
            scene=fep_features,
            probabilistic=False
        )]
    )

    training_sets['three subordinate examples'].append(
        [experimental_materials.UtteranceScenePair(
            utterance='fep',
            scene=fep_features,
            probabilistic=False
        ),
        experimental_materials.UtteranceScenePair(
            utterance='fep',
            scene=sub_example_2_features,
            probabilistic=False
        ),
        experimental_materials.UtteranceScenePair(
            utterance='fep',
            scene=sub_example_3_features,
            probabilistic=False
        )]
    )

    training_sets['three basic-level examples'].append(
        [
            experimental_materials.UtteranceScenePair(
                utterance='fep',
                scene=fep_features,
                probabilistic=False
            ),
            experimental_materials.UtteranceScenePair(
                utterance='fep',
                scene=basic_example_2_features,
                probabilistic=False
            ),
            experimental_materials.UtteranceScenePair(
                utterance='fep',
                scene=basic_example_3_features,
                probabilistic=False
            )
        ]
    )

    training_sets['three superordinate examples'].append(
        [
            experimental_materials.UtteranceScenePair(
                utterance='fep',
                scene=fep_features,
                probabilistic=False
            ),
            experimental_materials.UtteranceScenePair(
                utterance='fep',
                scene=sup_example_2_features,
                probabilistic=False
            ),
            experimental_materials.UtteranceScenePair(
                utterance='fep',
                scene=sup_example_3_features,
                probabilistic=False
            )
        ]
    )

    # generate test matches
    sub_match_features = fep_sup_features[:] + fep_bas_features[:] + fep_sub_features[:]
    for n in range(1, num_instance_levels+1):
        sub_match_features += ['finstance' + str(n) + str(i) + '8' for i in range(1, num_features+1)]

    basic_match_features = fep_sup_features[:] + fep_bas_features[:]
    for n in range(1, num_sub_levels+1):
        basic_match_features += ['fsub' + str(n) + str(i) + '6' for i in range(1, num_features+1)]
    for n in range(1, num_instance_levels+1):
        basic_match_features += ['finstance' + str(n) + str(i) + '9' for i in range(1, num_features+1)]

    sup_match_features = fep_sup_features[:]
    for n in range(1, num_basic_levels+1):
        sup_match_features += ['fbasic' + str(n) + str(i) + '4' for i in range(1, num_features+1)]
    for n in range(1, num_sub_levels+1):
        sup_match_features += ['fsub' + str(n) + str(i) + '7' for i in range(1, num_features+1)]
    for n in range(1, num_instance_levels+1):
        sup_match_features += ['finstance' + str(n) + str(i) + '10' for i in range(1, num_features+1)]

    test_sets = {}

    test_sets['subordinate matches'] = [
        experimental_materials.UtteranceScenePair(
            details='subordinate test object',
            utterance='fep',
            scene=sub_match_features,
            probabilistic=False
        )
    ]

    test_sets['basic-level matches'] = [
        experimental_materials.UtteranceScenePair(
            details='basic-level test object',
            utterance='fep',
            scene=basic_match_features,
            probabilistic=False
        )
    ]

    test_sets['superordinate matches'] = [
        experimental_materials.UtteranceScenePair(
            details='superordinate test object',
            utterance='fep',
            scene=sup_match_features,
            probabilistic=False
        )
    ]

    return training_sets, test_sets

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

                    l0 -= subtract_null_hypothesis
                    l1 -= subtract_null_hypothesis
                    l2 -= subtract_null_hypothesis

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
