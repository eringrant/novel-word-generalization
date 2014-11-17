#!/bin/python

# number of iterations for each condition
num_iterations = 20

parameter_values = {
# config file parameters
    'dummy' :                           [True, False],
    'forget' :                          [True, False],
    'forget-decay' :                    0,        # TODO: enforce parameter search when forget is True
    'novelty' :                         [True, False],
    'novelty-decay' :                   0,        # TODO: enforce parameter search when novelty is True
    'remove-singleton-utterances' :     [True, False],
    'maxtime' :                         [n for n in range(100, 1100, 100)], # max num words; 100 - 1000
# lear

# cartesian product of all parameter settings
experiment_conditions = (dict(izip(parameter_values, x)) for x in product(*parameter_values.itervalues()))



def run_experiments():



if __name__ == '__main__':
    run_experiments()

