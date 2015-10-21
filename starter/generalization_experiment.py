from __future__ import print_function
import json
import numpy as np
import os

from novel_word_generalization.core import learn


"""
generalization_experiment.py

Starter code containing a class to run one trial of the novel word
generalization experiment.
"""


# Mapping from the feature space to the stimuli file path
stimuli_files = {
    'simple': os.path.join('simple', 'stimuli.json'),
    'clothing': os.path.join('clothing', 'stimuli.json'),
    'containers': os.path.join('containers', 'stimuli.json'),
    'seats': os.path.join('seats', 'stimuli.json'),
    'xt-animals': os.path.join('xt_animals', 'stimuli.json'),
    'xt-vegetables': os.path.join('xt_vegetables', 'stimuli.json'),
    'xt-vehicles': os.path.join('xt_vehicles', 'stimuli.json'),
}

# Mapping from the feature space to the feature-level specification file path
feature_group_to_level_maps = {
    'simple': os.path.join('simple', 'feature_group_to_level_map.json'),
    'clothing': os.path.join('clothing', 'feature_group_to_level_map.json'),
    'containers': os.path.join('containers',
                               'feature_group_to_level_map.json'),
    'seats': os.path.join('seats', 'feature_group_to_level_map.json'),
    'xt-animals': os.path.join('xt_animals',
                               'feature_group_to_level_map.json'),
    'xt-vegetables': os.path.join('xt_vegetables',
                                  'feature_group_to_level_map.json'),
    'xt-vehicles': os.path.join('xt_vehicles',
                                'feature_group_to_level_map.json'),
}

# Mapping from the feature space to the feature-level specification file path
feature_to_feature_group_maps = {
    'simple': os.path.join('simple', 'feature_to_feature_group_map.json'),
    'clothing': os.path.join('clothing', 'feature_to_feature_group_map.json'),
    'containers': os.path.join('containers',
                               'feature_to_feature_group_map.json'),
    'seats': os.path.join('seats', 'feature_to_feature_group_map.json'),
    'xt-animals': os.path.join('xt_animals',
                               'feature_to_feature_group_map.json'),
    'xt-vegetables': os.path.join('xt_vegetables',
                                  'feature_to_feature_group_map.json'),
    'xt-vehicles': os.path.join('xt_vehicles',
                                'feature_to_feature_group_map.json'),
}


class InvalidParameterError(Exception):
    """
    Defines an exception that occurs when an invalid parameter value is
    specified.

    Members:
        value -- a description of the invalid parameter causing the error
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class Experiment(object):
    """An object that conducts a novel word generalization experiment trial.

    Members:
        params -- the  parameter settings for this Experiment
        feature_group_to_level_map -- a dict of (feature group -> hieararchy
        level)
        feature_to_feature_group_map -- a dict of (feature -> feature group)
        training_sets -- a dict of (training condition -> training set)
        test_sets -- a dict of (test condition -> test set)
        unseen_prob -- the prior probability of the learner to generalize a
            novel word to an object, before seeing any evidence, under the
            parameter setting specified in params
    """

    def __init__(self, params):
        """Initialize this Experiment.

        Initialize this Experiment according to the parameter settings
        specified in params.
        """
        self.params = params

        if not self.params['feature-space'] in ['simple', 'clothing',
                                                'containers', 'seats',
                                                'xt-animals', 'xt-vegetables',
                                                'xt-vehicles']:
            raise InvalidParameterError("undefined feature space")

        # Access the data files (stimuli)
        with open(os.path.join(self.params['data-path'],
                               stimuli_files[self.params['feature-space']]),
                  'r') as stimuli_file:
            stimuli = json.load(stimuli_file)
        self.training_sets = stimuli['training set']
        self.test_sets = stimuli['test set']

        # Access the information about the data that is assumed to belong to
        # the learner
        with open(os.path.join(self.params['data-path'],
                               feature_group_to_level_maps[self.params['feature-space']]),
                  'r') as feature_group_to_level_map:
            self.feature_group_to_level_map =\
                json.load(feature_group_to_level_map)
        with open(os.path.join(self.params['data-path'],
                               feature_to_feature_group_maps[self.params['feature-space']]),
                  'r') as feature_to_feature_group_map:
            self.feature_to_feature_group_map =\
                json.load(feature_to_feature_group_map)

        # Get the "unseen object" for computation of prior probability
        self.unseen_object = stimuli['unseen object features']

    def run(self):
        """Conduct this Experiment and return the results."""

        print("Conducting an experimental trial, with parameters:")
        print("\t", "feature space  = ", self.params['feature-space'])
        print("\t", "metric  = ", self.params['metric'])
        print("\t", "gamma_sup  = ", self.params['gamma-sup'])
        print("\t", "gamma_basic  = ", self.params['gamma-basic'])
        print("\t", "gamma_sub  = ", self.params['gamma-sub'])
        print("\t", "gamma_instance  = ", self.params['gamma-instance'])
        print("\t", "k_sup  = ", self.params['k-sup'])
        print("\t", "k_basic  = ", self.params['k-basic'])
        print("\t", "k_sub  = ", self.params['k-sub'])
        print("\t", "k_instance  = ", self.params['k-instance'])
        print("\t", "p_sup  = ", self.params['p-sup'])
        print("\t", "p_basic  = ", self.params['p-basic'])
        print("\t", "p_sub  = ", self.params['p-sub'])
        print("\t", "p_instance  = ", self.params['p-instance'])

        results = {}

        for training_condition in self.training_sets:

            print("\t\t", "Executing training condition:", training_condition,
                  "...")

            results[training_condition] = {}

            for test_condition in self.test_sets:

                print("\t\t\t", "Tested", test_condition)

                # Initialize the learner
                learner = learn.Learner(
                    gamma_sup=self.params['gamma-sup'],
                    gamma_basic=self.params['gamma-basic'],
                    gamma_sub=self.params['gamma-sub'],
                    gamma_instance=self.params['gamma-instance'],
                    k_sup=self.params['k-sup'],
                    k_basic=self.params['k-basic'],
                    k_sub=self.params['k-sub'],
                    k_instance=self.params['k-instance'],
                    p_sup=self.params['p-sup'],
                    p_basic=self.params['p-basic'],
                    p_sub=self.params['p-sub'],
                    p_instance=self.params['p-instance'],
                    feature_group_to_level_map=self.feature_group_to_level_map,
                    feature_to_feature_group_map=self.feature_to_feature_group_map
                )

                for trial in self.training_sets[training_condition]:

                    words = [self.params['word']]
                    scene = self.training_sets[training_condition][trial]

                    learner.process_pair(words, scene, './')

                gen_probs = []

                if self.params['subtract-prior']:
                    # Compute the prior (unseen) probability of an object
                    # (before learning)
                    unseen_prob =\
                        learner.generalization_prob(self.params['word'],
                                                    self.unseen_object,
                                                    metric=self.params['metric'])

                for test_object in self.test_sets[test_condition]:
                    scene = self.test_sets[test_condition][test_object]

                    gen_prob = learner.generalization_prob(
                        self.params['word'],
                        scene,
                        metric=self.params['metric']
                    )

                    if self.params['subtract-prior']:
                        gen_prob -= unseen_prob

                    gen_probs.append(gen_prob)

                gen_probs = np.array(gen_probs, dtype=np.float128)
                results[training_condition][test_condition] = gen_probs

        return results
