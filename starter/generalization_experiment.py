import json
import numpy as np
import os

import learn


"""
generalization_experiment.py

Starter code containing a class to run one trial of the novel word
generalization experiment.
"""


# Mapping from the feature space to the stimuli file path
stimuli_files = {
    'simple' : os.path.join('stimuli', 'simple_stimuli.json'),
    'clothing' : os.path.join('stimuli', 'clothing_stimuli.json'),
    'containers' : os.path.join('stimuli', 'container_stimuli.json'),
    'seats' : os.path.join('stimuli', 'seat_stimuli.json'),
    'xt-animals' : os.path.join('stimuli', 'xt_animal_stimuli.json'),
    'xt-vegetables' : os.path.join('stimuli', 'xt_vegetable_stimuli.json'),
    'xt-vehicles' : os.path.join('stimuli', 'xt_vehicle_stimuli.json'),
}

# Mapping from the feature space to the feature-level specification file path
feature_files = {
    'simple' : os.path.join('features', 'simple_features.json'),
    'clothing' : os.path.join('features', 'clothing_features.json'),
    'containers' : os.path.join('features', 'container_features.json'),
    'seats' : os.path.join('features', 'seats_feature.json'),
    'xt-animals' : os.path.join('features', 'xt_animal_features.json'),
    'xt-vegetables' : os.path.join('features', 'xt_vegetable_features.json'),
    'xt-vehicles' : os.path.join('features', 'xt_vehicle_features.json'),
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

        if not self.params['feature_space'] in ['simple', 'clothing',
                                                'containers', 'seats',
                                                'xt-animals', 'xt-vegetables',
                                                'xt-vehicles']:
            raise InvalidParameterError("undefined feature space")

        # Access the data files (stimuli and feature-level specifications)
        with open(os.path.join(self.params['data-path'],
                               stimuli_files[self.params['feature_space']]),
                               'rb') as stimuli_file:
            self.stimuli = json.load(stimuli_file)
        with open(os.path.join(self.params['data-path'],
                               feature_files[self.params['feature_space']]),
                               'rb') as feature_file:
            self.feature_map = json.load(feature_file)

        # Load the training and test sets
        self.training_sets =\
            self.stimuli[self.params['feature-space']]['training sets']
        self.test_sets =\
            self.stimuli[self.params['feature-space']]['test sets']

        # Initialize the learner (unseen probability computation
        self.learner = learn.Learner(
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
            feature_map=self.feature_map
        )

        # Compute the prior probability of an object
        self.unseen_prob =\
            self.learner.generalization_prob(self.params['word'],
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
