#!/usr/bin/python

# python2 articulated_hierarchy_experiments.py -o test_results.csv -c articulated_hierarchy_experiments.cfg -n 1

from __future__ import print_function, division

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mpmath; mpmath.mp.dps = 100
import numpy as np
import pickle
import pprint
import pydot
import nltk
from nltk.corpus import brown
from nltk.probability import *
import re


import learn

import experiment
import experimental_materials

latex = False

class GeneralisationExperiment(experiment.Experiment):

    def pre_setup(self, params):
        self.freqd = generate_frequency_distribution()
        return True

    def setup(self, params, rep):
        """ Runs in each child process. """
        gamma = params['gamma']
        k = params['k']

        training_sets, test_sets = generate_training_and_test_sets(self.freqd, params['freq'])

        visualise_training_and_test_sets(training_sets, test_sets)

        conds = ['one example',
                'three subordinate examples',
                'three basic-level examples',
                'three superordinate examples'
        ]
        results = {}

        for condition in conds:

            results[condition] = {}

            for training_set_num, training_set in enumerate(training_sets[condition]):

                print('============================================')
                print('Condition: ' + condition)
                print('============================================')

                for cond in test_sets:

                    learner = learn.Learner(gamma, k)

                    for trial in training_set:

                        learner.process_pair(trial.utterance(), list(reversed(trial.scene())), './')

                    print("\tMatch: " + cond)
                    print('--------------------------------------------')

                    print('Learned meaning of fep:')
                    print(learner._learned_lexicon.meaning('fep'))
                    print('--------------------------------------------')

                    take_average = 0
                    count = 0

                    for j in range(len(test_sets[cond][training_set_num])):
                        test_scene = test_sets[cond][training_set_num][j]
                        word = test_scene.utterance()[0]
                        scene = test_scene.scene()

                        gen_prob = learner.generalisation_prob(word, list(reversed(scene)))
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

        #pprint.pprint(results)

        savename  = 'hypothesis_space_experiments/'
        savename += 'gamma_' + str(gamma) + ',k_' + str(k)
        savename += '.png'

        bar_chart(results, savename=savename,
            normalise_over_test_scene=True,
            labels=['animals', 'vegetables', 'vehicles']
        )

def generate_training_and_test_sets(freqd, p):
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

    #import pdb; pdb.set_trace()
    # convert to scene representation
    for cond in training_sets:
        reps = []
        for i in range(len(training_sets[cond])):
            rep = []
            for item in training_sets[cond][i]:
                l = []
                for f in list(reversed(feature_map[item])):
                    if freqd.freq(f.split('.')[0].replace('_', ' ')) > p:
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
                for f in list(reversed(feature_map[item])):
                    if freqd.freq(f.split('.')[0].replace('_', ' ')) > p:
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

def bar_chart(results, savename=None, annotation=None,
        normalise_over_test_scene=True, subtract_null_hypothesis=None,
        labels=None):

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
        assert labels is not None
        fig, axes = plt.subplots(nrows=nrows, ncols=2, sharex=True, sharey=True)

        m = 0

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
                ax.set_title(str(labels[i]), fontsize='small')

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

def visualise_training_and_test_sets(training_sets, test_sets):

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

                d = todict(reversed(trial.scene())) #TODO: backwards
                graph = visit(graph, d)

            graph.write_png('hierarchy_' + str(training_set_num) + '_' + replace_with_underscores(str(condition)) + '_training_set.png')

        for cond in test_sets:

            graph = pydot.Dot(graph_type='graph', ranksep=0.2, resolution=96)
            graph = set_graph_defaults(graph)

            for match_num in range(len(test_sets[cond][training_set_num])):

                test_item = test_sets[cond][training_set_num][match_num]
                d = todict(reversed(test_item.scene())) #TODO: backwards
                graph = visit(graph, d)

            graph.write_png('hierarchy_' + str(training_set_num) + '_' + replace_with_underscores(str(cond)) + '_test_set.png')

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

def generate_frequency_distribution():

    words = FreqDist()

    for sentence in brown.sents():
        for word in sentence:
            words[word.lower()] += 1

    return words


if __name__ == "__main__":
    e = GeneralisationExperiment()
    e.start()
