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
    'simple' : os.path.join('simple', 'stimuli.json'),
    'clothing' : os.path.join('clothing', 'stimuli.json'),
    'containers' : os.path.join('containers', 'stimuli.json'),
    'seats' : os.path.join('seats', 'stimuli.json'),
    'xt-animals' : os.path.join('xt_animals', 'stimuli.json'),
    'xt-vegetables' : os.path.join('xt_vegetables', 'stimuli.json'),
    'xt-vehicles' : os.path.join('xt_vehicles', 'stimuli.json'),
}

# Mapping from the feature space to the feature-level specification file path
feature_group_to_level_maps = {
    'simple' : os.path.join('simple', 'feature_group_to_level_map.json'),
    'clothing' : os.path.join('clothing', 'feature_group_to_level_map.json'),
    'containers' : os.path.join('containers', 'feature_group_to_level_map.json'),
    'seats' : os.path.join('seats', 'feature_group_to_level_map.json'),
    'xt-animals' : os.path.join('xt_animals', 'feature_group_to_level_map.json'),
    'xt-vegetables' : os.path.join('xt_vegetables', 'feature_group_to_level_map.json'),
    'xt-vehicles' : os.path.join('xt_vehicles', 'feature_group_to_level_map.json'),
}

# Mapping from the feature space to the feature-level specification file path
feature_to_feature_group_maps = {
    'simple' : os.path.join('simple', 'feature_to_feature_group_map.json'),
    'clothing' : os.path.join('clothing', 'feature_to_feature_group_map.json'),
    'containers' : os.path.join('containers', 'feature_to_feature_group_map.json'),
    'seats' : os.path.join('seats', 'feature_to_feature_group_map.json'),
    'xt-animals' : os.path.join('xt_animals', 'feature_to_feature_group_map.json'),
    'xt-vegetables' : os.path.join('xt_vegetables', 'feature_to_feature_group_map.json'),
    'xt-vehicles' : os.path.join('xt_vehicles', 'feature_to_feature_group_map.json'),
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
    """ Execute a trial of the novel word generalization experiment.

    Members:
        params -- the  parameter settings for Experiment
        stimuli -- a dict of (feature space -> stimuli)
        feature_map -- a dict of (feature space -> (feature -> feature_group))
        training_sets -- a dict of (training condition -> training set)
        test_sets -- a dict of (test condition -> test set)
        learner -- the Learner upon which to perform the experiment,
            initialized with the parameter settings specified in params
        unseen_prob -- the prior probability of the learner to generalize a
            novel word to an object, before seeing any evidence, under the
            parameter settting specified in params
    """

    def __init__(self, params):
        """
        Initiliaze this Experiment according to the parameter settings specified
        in params.
        """
        self.params = params

        if not self.params['feature-space'] in ['simple', 'clothing',
                                                'containers', 'seats',
                                                'xt-animals', 'xt-vegetables',
                                                'xt-vehicles']:
            raise InvalidParameterError("undefined feature space")

        # Access the data files (stimuli and feature-level specifications)
        with open(os.path.join(self.params['data-path'],
                               stimuli_files[self.params['feature-space']]),
                               'r') as stimuli_file:
            self.stimuli = json.load(stimuli_file)

        # Access the information about the data that is assumed to belong to the
        # learner
        with open(os.path.join(self.params['data-path'],
                               feature_group_to_level_maps[self.params['feature-space']]),
                  'r') as feature_group_to_level_map:
            self.feature_group_to_level_map = json.load(feature_group_to_level_map)
        with open(os.path.join(self.params['data-path'],
                               feature_to_feature_group_maps[self.params['feature-space']]),
                  'r') as feature_to_feature_group_map:
            self.feature_to_feature_group_map = json.load(feature_to_feature_group_map)

        # Load the training and test sets
        self.training_sets = self.stimuli['training set']
        self.test_sets = self.stimuli['test set']

        # Initialize the learner (for the unseen probability computation)
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
            feature_to_feature_group_map=self.feature_to_feature_group_map,
        )

        # Compute the prior probability of an object
        self.unseen_prob =\
            learner.generalization_prob(self.params['word'],
                                        self.stimuli['unseen object features'])


    def run(self):
        """ Conduct this Experiment. """

        results = {}

        for training_condition in self.training_sets:

            results[training_condition] = {}

            for test_condition in self.test_sets:

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
                    feature_to_feature_group_map=self.feature_to_feature_group_map,
                )

                for trial in self.training_sets[training_condition]:

                    self.learner.process_pair(self.params['word'],
                                              self.training_sets[training_condition][trial],
                                              './')

                gen_probs = np.array((len(self.test_sets[test_condition])),
                                     dtype=np.float128)

                for i, test_object in enumerate(self.test_sets[test_condition]):
                    scene = self.test_sets[test_condition][test_object]

                    gen_prob = learner.generalisation_prob(
                        self.params['word'],
                        scene
                    )

                    if self.params['subtract-prior']:
                        gen_prob -= self.unseen_prob

                    gen_probs[i] = gen_prob

                results[training_condition][test_condition] = gen_probs

        return results
