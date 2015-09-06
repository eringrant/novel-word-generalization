#!/usr/bin/python


from __future__ import print_function, division


from argparse import ArgumentParser
from ConfigParser import ConfigParser
import itertools
import logging
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
import os
import pydot
import re
import sys
import types


import generalization_experiment


"""
Use the following command to run the code (using one CPU core):
python novel_word_generalization_experiments.py -c exp.cfg -n 1

"""



def items_to_params(items):
    params = {}
    for t,v in items:
        try: # evaluating the parameter
            params[t] = eval(v)
            if isinstance(params[t], np.ndarray):
                params[t] = params[t].tolist()
        except (NameError, SyntaxError):
            params[t] = v
    return params


def generate_conditions(paramlist):

    if type(paramlist) == types.DictType:
        paramlist = [paramlist]

    iparamlist = []
    for params in paramlist:
        if ('experiment' in params and params['experiment'] == 'single'):
            iparamlist.append(params) # only do one repetition of experiment
        else:
            iterparams = [p for p in params if hasattr(params[p], '__iter__')]
            if len(iterparams) > 0:
                iterfunc = itertools.product
                for il in iterfunc(*[params[p] for p in iterparams]):
                    par = params.copy() # keep the params having only one value
                    for i, ip in enumerate(iterparams):
                        par[ip] = il[i]
                    iparamlist.append(par)
            else:
                iparamlist.append(params)

    return iparamlist


def replace_with_underscores(s):
    s = re.sub(r"[^\w\s-]", '', s)
    s = re.sub(r"\s+", '_', s)
    return s


def run_trial(params):
    """
    Conduct a trial of the novel word generalization experiment, under the
    parameter settings specified in params.
    """
    experiment = generalization_experiment.Experiment(params)
    results = experiment.run()

    # Create a title for the plots PNG image
    title = 'results'
    title += ',' + 'featurespace_' + params['feature_space']
    title += ',' + 'gammasup_' + str(params['gamma_sup'])
    title += ',' + 'gammabasic_' + str(params['gamma_basic'])
    title += ',' + 'gammasub_' + str(params['gamma_sub'])
    title += ',' + 'gammainstance_' + str(params['gamma_instance'])
    title += ',' + 'ksup_' + str(params['k_sup'])
    title += ',' + 'kbasic_' + str(params['k_basic'])
    title += ',' + 'ksub_' + str(params['k_sub'])
    title += ',' + 'kinstance_' + str(params['k_instance'])
    title += ',' + 'psup_' + str(params['p_sup'])
    title += ',' + 'pbasic_' + str(params['p_basic'])
    title += ',' + 'psub_' + str(params['p_sub'])
    title += ',' + 'pinstance_' + str(params['p_instance'])
    title += ',' + 'subtractprior_' + str(params['subtract-prior'])
    title = os.path.join(params['output-path'], 'plots', title)

    if not os.path.exists(params['output-path']):
        os.makedirs(params['output-path'])
    if not os.path.exists(os.path.join(params['output-path'], 'plots')):
        os.makedirs(os.path.join(params['output-path'], 'plots'))

    visualise_training_and_test_sets(
        experiment.training_sets,
        experiment.test_sets,
        savename=title
    )

    bar_chart(
        results, savename=title + '.png',
        normalise_over_test_scene=True,
        labels=['vegetables', 'vehicles', 'animals']
    )

    # Write the results to a file used to generate PGF plots in LaTeX
    overwrite_results(results, title + '.dat')


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

def visualise_training_and_test_sets(training_sets, test_sets,
                                     savename='training_and_test_sets'):

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



def script(config_file, num_cores, **kwargs):

    # Parse the configuration file
    config_parser = ConfigParser()
    if not config_parser.read(config_file):
        raise SystemExit('Config file %s not found.' % config_file)

    # Generate the experimental conditions (by Cartesian product)
    paramlist = []
    for exp in config_parser.sections():
        params = items_to_params(config_parser.items(exp))
        params['name'] = exp
        paramlist.append(params)

    np.random.shuffle(paramlist) # randomise the order of experiment conditions

    exp_list = generate_conditions(paramlist)

    # Run the experiment(s), using the specified number of cores
    if num_cores == 1:
        for e in exp_list:
            run_trial(e)

    else:
        pool = Pool(processes=num_cores)
        pool.map(run_trial, exp_list)
        pool.close()


def parse_args(args):
    parser = ArgumentParser()

    parser.add_argument('--logging', type=str, default='INFO',
                        metavar='logging', choices=['DEBUG', 'INFO', 'WARNING',
                                                    'ERROR', 'CRITICAL'],
                        help='Logging level')

    parser.add_argument('--config_file', '-c', metavar='config_file', type=str,
                        required=True, help='The experiment config file')

    parser.add_argument('--num_cores', '-n', metavar='num__cores',
                        type=int, default=1, help='Number of processes used; default is 1')

    return parser.parse_args(args)


def main(args=sys.argv[1:]):
    args = parse_args(args)
    logging.basicConfig(level=args.logging)
    script(**vars(args))


if __name__ == '__main__':
    sys.exit(main())
