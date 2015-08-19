#!/usr/bin/python

# Use the following command to run the code (with one core):
# python articulated_hierarchy_experiments.py -c exp.cfg -n 1

from __future__ import print_function, division

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mpmath; mpmath.mp.dps = 100
import numpy as np
import os
import pickle
import pprint
import pydot
import re

import learn

import experiment
import experimental_materials

latex = False


class GeneralisationExperiment(experiment.Experiment):

    def setup(self, params, rep):
        """ Runs in each child process. """

        gamma_sup = params['gamma-sup']
        gamma_basic = params['gamma-basic']
        gamma_sub = params['gamma-sub']
        gamma_instance = params['gamma-instance']

        k_sup = params['k-sup']
        k_basic = params['k-basic']
        k_sub = params['k-sub']
        k_instance = params['k-instance']

        p_sup = params['p-sup']
        p_basic = params['p-basic']
        p_sub = params['p-sub']
        p_instance = params['p-instance']

        shift = params['shift']

        uni_freq = params['unigram-frequency']
        bi_freq = params['bigram-frequency']

        hierarchy_save_directory = params['hierarchy-save-directory']

        if params['hierarchy'] == 'articulated':
            training_sets, test_sets, unseen_object =\
            generate_articulated_training_and_test_sets(uni_freq, bi_freq,
                params['fix-leaf-feature'], type=params['leaf-node-type'])
        elif params['hierarchy'] == 'xt_hierarchy':
            training_sets, test_sets =\
            generate_xt_hypothesis_space_training_and_test_sets(uni_freq, bi_freq,
                params['fix-leaf-feature'], type=params['leaf-node-type'])
            unseen_object = None
        elif params['hierarchy'] == 'simple':
            training_sets, test_sets, unseen_object = generate_simple_training_and_test_sets(
                        params['num-sup-levels'],
                        params['num-basic-levels'],
                        params['num-sub-levels'],
                        params['num-instance-levels'],
                        params['num-features']
                    )
        else:
            raise NotImplementedError

#       # Feature hierarchy visualization
#        visualise_training_and_test_sets(training_sets, test_sets,
#            hierarchy_save_directory, uni_freq, bi_freq, params['name'],
#            params['hierarchy'])

        # get the prior probability of an object
        learner = learn.Learner(
            gamma_sup, gamma_basic, gamma_sub, gamma_instance,
            k_sup, k_basic, k_sub, k_instance,
            p_sup, p_basic, p_sub, p_instance,
            shift,
            modified_gamma=params['modified-gamma'],
            flat_hierarchy=params['flat-hierarchy']
        )

        if unseen_object is not None:
            unseen_prob = learner.generalisation_prob('fep', unseen_object.scene(), fixed_levels=True)

        conds = ['one example',
                'three subordinate examples',
                'three basic-level examples',
                'three superordinate examples'
        ]
        results = {}

        # loop through the training conditions
        for condition in conds:

            results[condition] = {}

            # loop through the different training sets (e.g., animals, vegetables, vehicles)
            for training_set_num, training_set in enumerate(training_sets[condition]):

                print('********************************************')
                print("TRAINING SET NUMBER", training_set_num)
                print('============================================')
                print('Condition: ' + condition)
                print('============================================')

                # loop through the test conditions
                for cond in test_sets:

                    learner = learn.Learner(
                        gamma_sup, gamma_basic, gamma_sub, gamma_instance,
                        k_sup, k_basic, k_sub, k_instance,
                        p_sup, p_basic, p_sub, p_instance,
                        shift,
                        modified_gamma=params['modified-gamma'],
                        flat_hierarchy=params['flat-hierarchy']
                    )

                    for i, trial in enumerate(training_set):

                        learner.process_pair(trial.utterance(), trial.scene(), './')

                    print("\tMatch: " + cond)
                    print('--------------------------------------------')

                    take_average = 0
                    count = 0

                    # loop though the test objects
                    for j in range(len(test_sets[cond][training_set_num])):
                        test_scene = test_sets[cond][training_set_num][j]
                        word = test_scene.utterance()[0] # assume the test utterance is a single word
                        scene = test_scene.scene()

                        if params['gen-prob'] == 'cosine':

                            # NOT CURRENTLY WORKING
                            raise NotImplementedError

                            #word = ['test']

                            ## represent the test scene with a Meaning
                            #learner.process_pair(word, scene, './')

                            #meaning1 = learner._learned_lexicon.meaning('fep')
                            #meaning2 = learner._learned_lexicon.meaning(word[0])

                            ## hack: add all the training set features to avoid
                            ## hashing errors
                            #for c in conds:
                            #    for ts in training_sets[c]:
                            #        for t in ts:
                            #            meaning2.add_features_to_hierarchy(t.scene())

                            ## hack: similarly, add all the test set features to avoid
                            ## hashing errors
                            #for c in test_sets:
                            #    for m in range(len(test_sets[cond][training_set_num])):
                            #        ts = test_sets[cond][training_set_num][m]
                            #        meaning1.add_features_to_hierarchy(ts.scene())

                            #gen_prob = learn.cosine(k, meaning1, meaning2)

                        elif params['gen-prob'] == 'product-fixed-levels':
                            gen_prob = learner.generalisation_prob(
                                word, scene,
                                fixed_levels=True,
                            )

                        elif params['gen-prob'] == 'product-variable-levels':
                            gen_prob = learner.generalisation_prob(word, scene, fixed_levels=False)

                        else:
                            raise NotImplementedError

                        if params['subtract-prior']:
                            gen_prob -= unseen_prob

                        print()
                        print("\tGeneralisation probability:", '\t', gen_prob)

                        take_average = mpmath.fadd(take_average, gen_prob)
                        count += 1

                    # average the test trials within the category
                    gen_prob = mpmath.fdiv(take_average, count)

                    try:
                        results[condition][cond].append(gen_prob)
                    except KeyError:
                        results[condition][cond] = []
                        results[condition][cond].append(gen_prob)

                    print('--------------------------------------------')


        # Create a title for the plots PNG image
        title = 'results'
        #title += ',' + replace_with_underscores(params['name'])
        #title += ',' + 'uni_' + str(uni_freq)
        #title += ',' + 'bi_' + str(bi_freq)
        #title += ',' + 'psup_' + str(p_sup)
        #title += ',' + 'pbasic_' + str(p_basic)
        #title += ',' + 'psub_' + str(p_sub)
        #title += ',' + 'pinstance_' + str(p_instance)
        #title += ',' + 'shift' + str(shift)
        title += ',' + 'gammasup_' + str(gamma_sup)
        title += ',' + 'gammabasic_' + str(gamma_basic)
        title += ',' + 'gammasub_' + str(gamma_sub)
        title += ',' + 'gammainstance_' + str(gamma_instance)
        #title += ',' + 'ksup_' + str(k_sup)
        #title += ',' + 'kbasic_' + str(k_basic)
        #title += ',' + 'ksub_' + str(k_sub)
        #title += ',' + 'kinstance_' + str(k_instance)
        #title += ',' + 'flf_' + str(params['fix-leaf-feature'])
        #title += ',' + 'mod-gamma_' + str(params['modified-gamma'])
        #title += ',' + 'flat-hier_' + str(params['flat-hierarchy'])
        #title += ',' + 'gen-prob_' + str(params['gen-prob'])
        title += ',' + 'struct_' + str(params['hierarchy'])
        title += ',' + 'prior_' + str(params['subtract-prior'])
        title = os.path.join(params['results-save-directory'], title)

        # If the parameters are equal (i.e., these are the child parameters)
        #if gamma_sup == gamma_basic and gamma_basic == gamma_sub and gamma_sub == gamma_instance and \
        #k_sup == k_basic and k_basic == k_sub and k_sub == k_instance:
        if True:

            if params['gen-prob'] == 'cosine':
                bar_chart(
                    results, savename=title + '.png', normalise_over_test_scene=False,
                    labels=['vegetables', 'vehicles', 'animals'],
                    y_limit=(0.5, 1.0)
                )

            else:
                bar_chart(
                    results, savename=title + '.png',
                    normalise_over_test_scene=True,
                    labels=['vegetables', 'vehicles', 'animals']
                )

            # Write the results to a file used to generate PGF plots in LaTeX
            overwrite_results(results, title + '.dat')

def generate_simple_training_and_test_sets(num_sup_levels, num_basic_levels, num_sub_levels,
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
    test_sets['subordinate matches'] = []
    test_sets['basic-level matches'] = []
    test_sets['superordinate matches'] = []

    test_sets['subordinate matches'].append([
        experimental_materials.UtteranceScenePair(
            details='subordinate test object',
            utterance='fep',
            scene=sub_match_features,
            probabilistic=False
        )
    ])

    test_sets['basic-level matches'].append([
        experimental_materials.UtteranceScenePair(
            details='basic-level test object',
            utterance='fep',
            scene=basic_match_features,
            probabilistic=False
        )
    ])

    test_sets['superordinate matches'].append([
        experimental_materials.UtteranceScenePair(
            details='superordinate test object',
            utterance='fep',
            scene=sup_match_features,
            probabilistic=False
        )
    ])

    unseen_sup_features = []
    unseen_bas_features = []
    unseen_sub_features = []
    unseen_instance_features = []

    for n in range(1, num_sup_levels+1):
        unseen_sup_features += ['fsup' + str(n) + str(i) + '0' for i in range(1, num_features+1)]
    for n in range(1, num_basic_levels+1):
        unseen_bas_features += ['fbas' + str(n) + str(i) + '0' for i in range(1, num_features+1)]
    for n in range(1, num_sub_levels+1):
        unseen_sub_features += ['fsub' + str(n) + str(i) + '0' for i in range(1, num_features+1)]
    for n in range(1, num_instance_levels+1):
        unseen_instance_features += ['finstance' + str(n) + str(i) + '0' for i in range(1, num_features+1)]

    unseen_features = unseen_sup_features + unseen_bas_features + unseen_sub_features + unseen_instance_features

    unseen_object = experimental_materials.UtteranceScenePair(
            details='unseen object',
            utterance='fep',
            scene=unseen_features,
            probabilistic=False
        )

    return training_sets, test_sets, unseen_object

def generate_xt_hypothesis_space_training_and_test_sets(uni_freq, bi_freq, fix_leaf_feature, type=1):
    """

    """
    # access the feature mappings
    with open('xt_hypothesis_space_feature_map.pkl', 'rb') as f:
        feature_map = pickle.load(f)

    # organise the sets
    training_sets = {}
    training_sets['one example'] = []
    training_sets['three subordinate examples'] = []
    training_sets['three basic-level examples'] = []
    training_sets['three superordinate examples'] = []

    training_sets['one example'].extend([
        ['16'],
        ['31'],
        ['1'],
    ])

    training_sets['three subordinate examples'].extend([
        ['16', '17', '18'],
        ['31', '32', '33'],
        ['1', '2', '3'],
    ])

    training_sets['three basic-level examples'].extend([
        ['16', '21', '22'],
        ['31', '36', '37'],
        ['1', '6', '7'],
    ])

    training_sets['three superordinate examples'].extend([
        ['16', '25', '26'],
        ['31', '40', '41'],
        ['1', '10', '11'],
    ])

    test_sets = {}
    test_sets['subordinate matches'] = []
    test_sets['basic-level matches'] = []
    test_sets['superordinate matches'] = []

    test_sets['subordinate matches'].extend([
        ['19', '20'],
        ['34', '35'],
        ['4', '5'],
    ])

    test_sets['basic-level matches'].extend([
        ['23', '24'],
        ['38', '39'],
        ['8', '9'],
    ])

    test_sets['superordinate matches'].extend([
        ['27', '28', '29', '30'],
        ['42', '43', '44', '45'],
        ['12', '13', '14', '15'],
    ])

    # convert to scene representation
    for cond in training_sets:
        reps = []
        for i in range(len(training_sets[cond])):
            rep = []
            for item in training_sets[cond][i]:
                l = feature_map[item][:]
                rep.append(
                    experimental_materials.UtteranceScenePair(
                        utterance='fep',
                        scene=l,
                        probabilistic=False
                    )
                )
            reps.append(rep)
        training_sets[cond] = reps

    for cond in test_sets:
        reps = []
        for i in range(len(test_sets[cond])):
            rep = []
            for item in test_sets[cond][i]:
                l = feature_map[item][:]
                rep.append(
                    experimental_materials.UtteranceScenePair(
                        utterance='fep',
                        scene=l,
                        probabilistic=False
                    )
                )
            reps.append(rep)
        test_sets[cond] = reps

    with open('xt_hypothesis_space_sets.txt', 'w') as f:
        f.write(pprint.pformat(training_sets))
        f.write(pprint.pformat(test_sets))
    return training_sets, test_sets

def generate_articulated_training_and_test_sets(uni_freq, bi_freq, fix_leaf_feature, type=1):
    """

    """
    # access the feature mappings
    with open('feature_map.pkl', 'rb') as f:
        feature_map = pickle.load(f)

    # organise the sets
    training_sets = {}
    training_sets['one example'] = []
    training_sets['three subordinate examples'] = []
    training_sets['three basic-level examples'] = []
    training_sets['three superordinate examples'] = []

    if type == 1:  # animals, vegetables and vehicles
        training_sets['one example'].extend([
            ['liver-spotted_dalmatian.n.01'],
            ['green_pepper.n.01'],
            ['ladder_truck.n.01']
        ])

        training_sets['three subordinate examples'].extend([
            ['liver-spotted_dalmatian.n.01', 'liver-spotted_dalmatian.n.01',
                'liver-spotted_dalmatian.n.01'],
            ['green_pepper.n.01', 'green_pepper.n.01', 'green_pepper.n.01'],
            ['ladder_truck.n.01', 'ladder_truck.n.01', 'ladder_truck.n.01']
        ])

        training_sets['three basic-level examples'].extend([
            ['liver-spotted_dalmatian.n.01', 'shih-tzu.n.01', 'beagle.n.01'],
            ['green_pepper.n.01', 'cayenne.n.03', 'bell_pepper.n.02'],
            ['ladder_truck.n.01', 'garbage_truck.n.01', 'tandem_trailer.n.01']
        ])

        training_sets['three superordinate examples'].extend([
            ['liver-spotted_dalmatian.n.01', 'hippopotamus.n.01', 'toucanet.n.01'],
            ['green_pepper.n.01', 'uruguay_potato.n.02', 'gherkin.n.02'],
            ['ladder_truck.n.01', 'trail_bike.n.01', 'subcompact.n.01']
        ])

        test_sets = {}
        test_sets['subordinate matches'] = []
        test_sets['basic-level matches'] = []
        test_sets['superordinate matches'] = []

        test_sets['subordinate matches'].extend([
            ['liver-spotted_dalmatian.n.01'],
            ['green_pepper.n.01'],
            ['ladder_truck.n.01']
        ])

        test_sets['basic-level matches'].extend([
            ['king_charles_spaniel.n.01', 'pembroke.n.01'],
            ['tabasco.n.03', 'pimento.n.02'],
            ['dump_truck.n.01', 'transporter.n.01']
        ])

        test_sets['superordinate matches'].extend([
            ['tabby.n.01', 'grizzly.n.01', 'california_sea_lion.n.01',
                'farm_horse.n.01'],
            ['carrot.n.03', 'crisphead_lettuce.n.01', 'shallot.n.03',
                'pumpkin.n.02'],
            ['sports_car.n.01', 'berlin.n.03', 'hearse.n.01', 'gypsy_cab.n.01']
        ])

    elif type == 2:  # clothing, containers and seats
        training_sets['one example'].extend([
            ['dress_shirt.n.01'],  # clothing
            ['cigar_box.n.01'],  # containers
            ['ladder-back.n.01']  # seats
        ])

        training_sets['three subordinate examples'].extend([
            ['dress_shirt.n.01']*3,
            ['cigar_box.n.01']*3,
            ['ladder-back.n.01']*3
        ])

        training_sets['three basic-level examples'].extend([
            ['dress_shirt.n.01', 'polo_shirt.n.01', 'dashiki.n.01'],
            ['cigar_box.n.01', 'mailbox.n.01', 'shoebox.n.02'],
            ['ladder-back.n.01', 'lawn_chair.n.01', "captain's_chair.n.01"]
        ])

        training_sets['three superordinate examples'].extend([
            ['dress_shirt.n.01', 'sable_coat.n.01', 'chino.n.01'],
            ['cigar_box.n.01', 'beer_glass.n.01', 'garbage.n.03'],
            ['ladder-back.n.01', 'park_bench.n.01', 'daybed.n.01']
        ])

        test_sets = {}
        test_sets['subordinate matches'] = []
        test_sets['basic-level matches'] = []
        test_sets['superordinate matches'] = []

        test_sets['subordinate matches'].extend([
            ['dress_shirt.n.01'],
            ['cigar_box.n.01'],
            ['ladder-back.n.01']
        ])

        test_sets['basic-level matches'].extend([
            ['tank_top.n.01', 'kurta.n.01'],
            ['matchbox.n.01', 'cereal_box.n.01'],
            ['swivel_chair.n.01', 'boston_rocker.n.01']
        ])

        test_sets['superordinate matches'].extend([
            ['pajama.n.01', 'full_skirt.n.01', 'oilskin.n.01', 'pea_jacket.n.01'],
            ['planter.n.03', 'ashtray.n.01', 'goblet.n.01', 'shot_glass.n.01'],
            ['settee.n.02', 'pew.n.01', 'settle.n.01', 'love_seat.n.01']
        ])


    else:
        raise NotImplementedError

    # convert to scene representation
    for cond in training_sets:
        reps = []
        for i in range(len(training_sets[cond])):
            rep = []
            for item in training_sets[cond][i]:
                l = []
                for j, f in enumerate(feature_map[item]):
                    if j + 1 == len(feature_map[item]) and fix_leaf_feature:
                        l.append(f)
                    elif exceeds_frequency_threshold(f.split('.')[0].replace('_', ' '), uni_freq, bi_freq):
                        l.append(f)
                rep.append(
                    experimental_materials.UtteranceScenePair(
                        utterance='fep',
                        scene=l,
                        probabilistic=False
                    )
                )
            reps.append(rep)
        training_sets[cond] = reps

    for cond in test_sets:
        reps = []
        for i in range(len(test_sets[cond])):
            rep = []
            for item in test_sets[cond][i]:
                l = []
                for j, f in enumerate(feature_map[item]):
                    if j + 1 == len(feature_map[item]) and fix_leaf_feature:
                        l.append(f)
                    elif exceeds_frequency_threshold(f.split('.')[0].replace('_', ' '), uni_freq, bi_freq):
                        l.append(f)
                rep.append(
                    experimental_materials.UtteranceScenePair(
                        utterance='fep',
                        scene=l,
                        probabilistic=False
                    )
                )
            reps.append(rep)
        test_sets[cond] = reps

    with open('sets.txt', 'w') as f:
        f.write(pprint.pformat(training_sets))
        f.write(pprint.pformat(test_sets))
    return training_sets, test_sets

def exceeds_frequency_threshold(gram, uni_freq, bi_freq):
    #with open('all_ngrams.pkl') as f:
    with open('google_ngrams_freq.pkl') as f:
        a = pickle.load(f)

    if a[gram] == 0:
        return False

    if len(gram.split(' ')) == 1:
        if a[gram] > uni_freq or uni_freq == float('-inf'):
            return True
        else:
            return False

    elif len(gram.split(' ')) == 2:
        if a[gram] > bi_freq or bi_freq == float('-inf'):
            return True
        else:
            return False

    else:
        return False

def bar_chart(results, savename=None, annotation=None,
        normalise_over_test_scene=True, subtract_null_hypothesis=None,
        labels=None, y_limit=None):

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

        if subtract_null_hypothesis is not None:

            l0 = np.array(l0)
            l1 = np.array(l1)
            l2 = np.array(l2)

            l0 -= subtract_null_hypothesis
            l1 -= subtract_null_hypothesis
            l2 -= subtract_null_hypothesis

            l0 = list(l0)
            l1 = list(l1)
            l2 = list(l2)

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

        error0 = [np.std(results[cond]['subordinate matches']) for cond in conditions]
        error1 = [np.std(results[cond]['basic-level matches']) for cond in conditions]
        error2 = [np.std(results[cond]['superordinate matches']) for cond in conditions]

        width = 0.5
        fig = plt.figure()
        ax = fig.add_subplot(111)
        p0 = ax.bar(ind,l0,width,color='r',yerr=error0)
        p1 = ax.bar(ind+width,l1,width,color='g',yerr=error1)
        p2 = ax.bar(ind+2*width,l2,width,color='b',yerr=error2)

        ax.set_ylabel("generalisation probability")
        ax.set_xlabel("condition")

        if y_limit:
            ax.set_ylim(y_limit)

        m = np.max(l0 + l1 + l2)

    else:
        assert labels is not None
        fig, axes = plt.subplots(nrows=nrows, ncols=2, sharex=True, sharey=True)

        m = 0

        for i, ax in enumerate(axes.flat):

            if i == len(results[conditions[0]]['subordinate matches']):

                ax.set_title('Average over all training-test sets', fontsize='small')

                l0 = [np.mean(results[cond]['subordinate matches']) for cond in conditions]
                l1 = [np.mean(results[cond]['basic-level matches']) for cond in conditions]
                l2 = [np.mean(results[cond]['superordinate matches']) for cond in conditions]


                if subtract_null_hypothesis is not None:

                    l0 = np.array(l0)
                    l1 = np.array(l1)
                    l2 = np.array(l2)

                    l0 -= subtract_null_hypothesis
                    l1 -= subtract_null_hypothesis
                    l2 -= subtract_null_hypothesis

                    l0 = list(l0)
                    l1 = list(l1)
                    l2 = list(l2)

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
                ax.set_title(str(labels[i]), fontsize='small')

                l0 = [results[cond]['subordinate matches'][i] for cond in conditions]
                l1 = [results[cond]['basic-level matches'][i] for cond in conditions]
                l2 = [results[cond]['superordinate matches'][i] for cond in conditions]


                if subtract_null_hypothesis is not None:

                    l0 = np.array(l0)
                    l1 = np.array(l1)
                    l2 = np.array(l2)

                    l0 -= subtract_null_hypothesis
                    l1 -= subtract_null_hypothesis
                    l2 -= subtract_null_hypothesis

                    l0 = list(l0)
                    l1 = list(l1)
                    l2 = list(l2)

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


                p0 = ax.bar(ind,l0,width,color='r')
                p1 = ax.bar(ind+width,l1,width,color='g')
                p2 = ax.bar(ind+2*width,l2,width,color='b')

            xlabels = ('1', '3 sub.', '3 basic', '3 super.')
            ax.set_xticks(ind + 2 * width)
            ax.set_xticklabels(xlabels)

            if y_limit:
                ax.set_ylim(y_limit)

    #ax.set_ylabel("gen. prob.")
    #ax.set_xlabel("condition")
    if y_limit:
        plt.ylim(y_limit)
    elif normalise_over_test_scene is True:
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
        # add check for significant results
        #if l0[0] < 0.65 and l1[0] > 0.35 and l1[1] < 0.3 and l1[3] / l0[3] > 0.5:
        if True:

            plt.savefig(savename, bbox_extra_artists=(lgd,), bbox_inches='tight')

def visualise_training_and_test_sets(training_sets, test_sets, save_directory, uni, bi, freq_corpus, struct):

    conds = ['one example',
            'three subordinate examples',
            'three basic-level examples',
            'three superordinate examples'
    ]

    num_sets = len(training_sets[conds[0]])
    assert num_sets == len(test_sets['subordinate matches'])

    for training_set_num in range(num_sets):

        for condition in conds:

            graph = pydot.Dot(graph_type='graph', ranksep=0.2, resolution=96)
            graph = set_graph_defaults(graph)
            training_set = training_sets[condition][training_set_num]

            for trial in training_set:

                d = todict(trial.scene())
                graph = visit(graph, d)

            title = 'hierarchy'
            title += ',' + replace_with_underscores(freq_corpus)
            title += ',' + 'uni_' + str(uni)
            title += ',' + 'bi_' + str(bi)
            title += ',' + 'struct_' + str(struct)
            title += ',' 'train_set_num_' + str(training_set_num)
            title += ',' + replace_with_underscores(str(condition))
            title += ',' + 'training_set.png'
            title = os.path.join(save_directory, title)
            graph.write_png(title)

        for cond in test_sets:

            graph = pydot.Dot(graph_type='graph', ranksep=0.2, resolution=96)
            graph = set_graph_defaults(graph)

            try:
                for match_num in range(len(test_sets[cond][training_set_num])):
                    test_item = test_sets[cond][training_set_num][match_num]
                    d = todict(test_item.scene())
                    graph = visit(graph, d)
            except Exception:
                import pdb; pdb.set_trace()

            title = 'hierarchy'
            title += ',' + replace_with_underscores(freq_corpus)
            title += ',' + 'uni_' + str(uni)
            title += ',' + 'bi_' + str(bi)
            title += ',' + 'struct_' + str(struct)
            title += ',' 'train_set_num_' + str(training_set_num)
            title += ',' + replace_with_underscores(str(cond))
            title += ',' + 'test_set.png'
            title = os.path.join(save_directory, title)
            graph.write_png(title)

def set_graph_defaults(graph):
    graph.set_node_defaults(shape='oval', fixedsize='true',height=.20, width=.60, fontsize=8)
    return graph

def replace_with_underscores(s):
    s = re.sub(r"[^\w\s-]", '', s)
    s = re.sub(r"\s+", '_', s)
    return s

def todict(lst):
    d = {}
    current_level = d
    for part in lst:
        if part not in current_level:
            current_level[part] = {}
        current_level = current_level[part]
    return d

def draw(graph, parent_name, child_name):
    edge = pydot.Edge(parent_name, child_name)
    edge.set_len(0.1)
    graph.add_edge(edge)
    return graph

def visit(graph, node, parent=None):
    for k,v in node.iteritems():
        if isinstance(v, dict):
            # We start with the root node whose parent is None
            # we don't want to graph the None node
            if parent:
                graph = draw(graph, parent, k)
            graph = visit(graph, v, k)
        else:
            graph = draw(graph, parent, k)
            # drawing the label using a distinct name
            graph = draw(graph, k, k+'_'+v)
    return graph

def overwrite_results(results, savename):

    conditions = [
        'one example',
        'three subordinate examples',
        'three basic-level examples',
        'three superordinate examples'
    ]

    abbrev_condition_names = {
        'one example' : '1 ex.',
        'three subordinate examples' : '3 sub.',
        'three basic-level examples' : '3 basic',
        'three superordinate examples' : '3 super.'
    }

    with open(savename, 'w') as f:
        f.write("condition,sub. match,basic match,super. match\n")
        for condition in conditions:
            normalisation = \
                np.mean(results[condition]['subordinate matches']) + \
                np.mean(results[condition]['basic-level matches']) + \
                np.mean(results[condition]['superordinate matches'])
            f.write(abbrev_condition_names[condition])
            f.write(',')
            f.write(str(np.mean(results[condition]['subordinate matches'])/normalisation))
            f.write(',')
            f.write(str(np.mean(results[condition]['basic-level matches'])/normalisation))
            f.write(',')
            f.write(str(np.mean(results[condition]['superordinate matches'])/normalisation))
            f.write("\n")

    print('Wrote results out to', savename)

if __name__ == "__main__":
    e = GeneralisationExperiment()
    e.start()
