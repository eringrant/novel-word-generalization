import os
import unittest
import numpy as np
import itertools

import learn
import learnconfig
import referent_probability_experiments

class testReferentProbabilityCalculation(unittest.TestCase):

    def setUp(self):
        # create the experiment object
        self.experiment = referent_probability_experiments.NovelReferentExperiment()

        # create the config file
        config_string = """[Smoothing]
beta=10000
lambda=-1
power=1
epsilon=0.01
alpha=20

[Similarity]
simtype=COS
theta=0.7

[Features]
dummy=false
forget=false
forget-decay=0
novelty=false
novelty-decay=0
assoc-type=SUM
category=false
semantic-network=false
hub-type=hub-freq-degree
hub-num=75

[Statistics]
stats=true
context-stats=false
familiarity-smoothing=0.01
familiarity-measure=COUNT
age-of-exposure-norm=100
maxtime=1000"""

        with open('toy_config', 'w') as f:
            f.write(config_string)

        self.learner_config = learnconfig.LearnerConfig('toy_config')

    def tearDown(self):
        os.remove('toy_config')

    def assertDeepAlmostEqual(self, expected, actual, places=3, **kwargs):
        is_root = not '__trace' in kwargs
        trace = kwargs.pop('__trace', 'ROOT')
        try:
            if isinstance(expected, (int, float, long, complex, np.float64, np.float128)):
                self.assertAlmostEqual(expected, actual, places=places)
            elif isinstance(expected, (list, tuple, np.ndarray)):
                self.assertEqual(len(expected), len(actual))
                for index in xrange(len(expected)):
                    v1, v2 = expected[index], actual[index]
                    self.assertDeepAlmostEqual(v1, v2, places=places,
                                          __trace=repr)
            elif isinstance(expected, dict):
                self.assertEqual(set(expected), set(actual))
                for key in expected:
                    self.assertDeepAlmostEqual(expected[key], actual[key],
                                          places=places, __trace=repr(key))
            else:
                self.assertEqual(expected, actual)
        except AssertionError as exc:
            exc.__dict__.setdefault('traces', []).append(trace)
            if is_root:
                trace = ' -> '.join(reversed(exc.traces))
                exc = AssertionError("%s\nTRACE: %s" % (exc.message, trace))
            raise exc

    def test_single_feature_no_overlap(self):
        """NB: hand-tested for correctness."""

        # create the testing lexicon and corpora
        test_lex = """milk:N DAIRY:0.5,WHITE:0.3,BEVERAGE:0.2,"""
        test_data = """1-----
SENTENCE: she:N drinks:V milk:N
SEM_REP:  ,FEMALE,DRINK,WHITE,DAIRY
1-----
SENTENCE: buy:V some:N milk:N
SEM_REP:  ,BUY,SOME,DAIRY,BEVERAGE
1-----
SENTENCE: she:N had:V milk:N
SEM_REP:  ,FEMALE,POSSESSION,WHITE,DAIRY"""

        with open('toy_lex', 'w') as f:
            f.write(test_lex)

        with open('toy_data', 'w') as f:
            f.write(test_data)

        self.experiment.learner = learn.Learner('toy_lex', self.learner_config, [])
        self.experiment.learner.process_corpus('toy_data', './')

        # test scene
        self.experiment.utterance = ['coffee:N']
        self.experiment.scene = ['DAIRY', 'HOT']
        self.experiment.referent_to_features_map = {'milk:N' : ['DAIRY'], 'coffee:N' : ['HOT']}

        self.experiment.learner.process_pair(self.experiment.utterance, self.experiment.scene, './', False)

        actual_mul = self.experiment.calculate_referent_probability('MUL')
        actual_sum = self.experiment.calculate_referent_probability('SUM')

        # not testing the learning procedure here
        words_seen = ['she:N', 'had:V', 'milk:N', 'drinks:V', 'buy:V', 'some:N', 'coffee:N']
        total_word_frequency = np.sum([self.experiment.learner._wordsp.frequency(word) for word in words_seen])

        joint_probs = {}

        for feature in self.experiment.scene:
            for word in self.experiment.learner._wordsp.all_words(0):
                joint_probs[(feature, word)] = \
                    np.divide(np.float128(self.experiment.learner._learned_lexicon.prob(word, feature)) \
                    * self.experiment.learner._wordsp.frequency(word),
                    total_word_frequency)

        mul_ref_prob_milk = np.divide(joint_probs[('DAIRY', 'coffee:N')],
            np.sum([joint_probs[('DAIRY', word)] for word in words_seen]))

        mul_ref_prob_coffee = np.divide(joint_probs[('HOT', 'coffee:N')],
            np.sum([joint_probs[('HOT', word)] for word in words_seen]))

        sum_ref_prob_milk = np.divide(joint_probs[('DAIRY', 'coffee:N')],
            np.sum([joint_probs[('DAIRY', word)] for word in words_seen]))

        sum_ref_prob_coffee = np.divide(joint_probs[('HOT', 'coffee:N')],
            np.sum([joint_probs[('HOT', word)] for word in words_seen]))

        expected_mul = {}
        expected_mul[('coffee:N', 'coffee:N')] = mul_ref_prob_coffee
        expected_mul[('coffee:N', 'milk:N')] = mul_ref_prob_milk

        expected_sum = {}
        expected_sum[('coffee:N', 'coffee:N')] = sum_ref_prob_coffee
        expected_sum[('coffee:N', 'milk:N')] = sum_ref_prob_milk

        self.assertDeepAlmostEqual(actual_mul, expected_mul)
        self.assertDeepAlmostEqual(actual_sum, expected_sum)

        os.remove('toy_data')
        os.remove('toy_lex')

    def test_two_features_one_overlap(self):
        """NB: hand-tested for correctness."""

        # create the testing lexicon and corpora
        test_lex = """milk:N DAIRY:0.5,WHITE:0.3,BEVERAGE:0.2,"""
        test_data = """1-----
SENTENCE: she:N drinks:V milk:N
SEM_REP:  ,FEMALE,DRINK,WHITE,DAIRY
1-----
SENTENCE: buy:V some:N milk:N
SEM_REP:  ,BUY,SOME,DAIRY,BEVERAGE
1-----
SENTENCE: she:N had:V milk:N
SEM_REP:  ,FEMALE,POSSESSION,WHITE,DAIRY"""

        with open('toy_lex', 'w') as f:
            f.write(test_lex)

        with open('toy_data', 'w') as f:
            f.write(test_data)

        self.experiment.learner = learn.Learner('toy_lex', self.learner_config, [])
        self.experiment.learner.process_corpus('toy_data', './')

        # test scene
        self.experiment.utterance = ['coffee:N']
        self.experiment.scene = ['DAIRY', 'WHITE', 'HOT']
        self.experiment.referent_to_features_map = {'milk:N' : ['DAIRY', 'WHITE'], 'coffee:N' : ['HOT', 'WHITE']}

        self.experiment.learner.process_pair(self.experiment.utterance, self.experiment.scene, './', False)

        actual_mul = self.experiment.calculate_referent_probability('MUL')
        actual_sum = self.experiment.calculate_referent_probability('SUM')

        # not testing the learning procedure here
        words_seen = ['she:N', 'had:V', 'milk:N', 'drinks:V', 'buy:V', 'some:N', 'coffee:N']
        total_word_frequency = np.sum([self.experiment.learner._wordsp.frequency(word) for word in words_seen])

        joint_probs = {}

        for feature in self.experiment.scene:
            for word in self.experiment.learner._wordsp.all_words(0):
                joint_probs[(feature, word)] = \
                    np.divide(np.float128(self.experiment.learner._learned_lexicon.prob(word, feature)) \
                    * self.experiment.learner._wordsp.frequency(word),
                    total_word_frequency)

        mul_ref_prob_milk = np.multiply(np.divide(joint_probs[('DAIRY', 'coffee:N')],
            np.sum([joint_probs[('DAIRY', word)] for word in words_seen])),
            np.divide(joint_probs[('WHITE', 'coffee:N')],
            np.sum([joint_probs[('WHITE', word)] for word in words_seen])))

        mul_ref_prob_coffee = np.multiply(np.divide(joint_probs[('HOT', 'coffee:N')],
            np.sum([joint_probs[('HOT', word)] for word in words_seen])),
            np.divide(joint_probs[('WHITE', 'coffee:N')],
            np.sum([joint_probs[('WHITE', word)] for word in words_seen])))

        sum_ref_prob_milk = np.divide(
            np.sum([joint_probs[(feature, 'coffee:N')] for feature in ['DAIRY', 'WHITE']]),
            np.sum([np.sum([joint_probs[(feature, word)] for word in words_seen]) for feature in ['DAIRY', 'WHITE']]))

        sum_ref_prob_coffee = np.divide(
            np.sum([joint_probs[(feature, 'coffee:N')] for feature in ['HOT', 'WHITE']]),
            np.sum([np.sum([joint_probs[(feature, word)] for word in words_seen]) for feature in ['HOT', 'WHITE']]))

        expected_mul = {}
        expected_mul[('coffee:N', 'coffee:N')] = mul_ref_prob_coffee
        expected_mul[('coffee:N', 'milk:N')] = mul_ref_prob_milk

        expected_sum = {}
        expected_sum[('coffee:N', 'coffee:N')] = sum_ref_prob_coffee
        expected_sum[('coffee:N', 'milk:N')] = sum_ref_prob_milk

        self.assertDeepAlmostEqual(actual_mul, expected_mul)
        self.assertDeepAlmostEqual(actual_sum, expected_sum)

        os.remove('toy_data')
        os.remove('toy_lex')
        self.experiment.learner.reset()

    def test_ten_features_two_overlap(self):
        """NB: hand-tested for correctness."""

        # create the testing lexicon and corpora
        test_lex = """milk:N DAIRY:0.5,WHITE:0.3,BEVERAGE:0.2,"""

        with open('toy_lex', 'w') as f:
            f.write(test_lex)

        self.experiment.learner = learn.Learner('toy_lex', self.learner_config, [])
        self.experiment.learner.process_corpus('test_input_wn_fu_cs_scaled_categ.dev_without_gumdrop:N', './')

        # ensure that the learner has seen the familiar object with all its features
        scene = ['snowman', 'creation#2', 'entity#1', 'artifact#1', 'model#4', 'representation#2', 'physical entity#1', 'object#1', 'matter#3', 'figure#4']
        self.experiment.learner.process_pair(['snowman:N'], scene, './', False)

        # test scene
        self.experiment.utterance = ['gumdrop:N']
        self.experiment.scene = ['snowman', 'sweet#3', 'food#1', 'creation#2', 'entity#1', 'artifact#1', 'model#4', 'dainty#1', 'substance#7', 'candy#1', 'whole#2', 'gumdrop', 'representation#2', 'physical entity#1', 'object#1', 'matter#3', 'nutriment#2', 'figure#4']
        self.experiment.referent_to_features_map = {
            'snowman:N' : ['snowman', 'creation#2', 'entity#1', 'artifact#1', 'model#4', 'representation#2', 'physical entity#1', 'object#1', 'matter#3', 'figure#4'],
            'gumdrop:N' : ['sweet#3', 'food#1',  'entity#1', 'dainty#1', 'substance#7', 'candy#1', 'whole#2', 'gumdrop', 'nutriment#2', 'physical entity#1']
            }

        self.experiment.learner.process_pair(self.experiment.utterance, self.experiment.scene, './', False)

        actual_mul = self.experiment.calculate_referent_probability('MUL')
        actual_sum = self.experiment.calculate_referent_probability('SUM')

        # not testing the learning procedure here
        #words_seen = ['she:N', 'had:V', 'milk:N', 'drinks:V', 'buy:V', 'some:N', 'coffee:N', 'builds:V', 'snowman:N', 'gumdrop:N']
        total_word_frequency = np.sum([self.experiment.learner._wordsp.frequency(word) for word in self.experiment.learner._wordsp.all_words(0)])
        joint_probs = {}

        for feature in self.experiment.scene:
            for word in self.experiment.learner._wordsp.all_words(0):
                joint_probs[(feature, word)] = \
                    np.divide(np.float128(self.experiment.learner._learned_lexicon.prob(word, feature)) \
                    * self.experiment.learner._wordsp.frequency(word),
                    total_word_frequency)

        mul_ref_prob_snowman = np.divide(joint_probs[('snowman', 'gumdrop:N')],
            np.sum([joint_probs[('snowman', word)] for word in self.experiment.learner._wordsp.all_words(0)])) *\
            np.divide(joint_probs[('creation#2', 'gumdrop:N')],
            np.sum([joint_probs[('creation#2', word)] for word in self.experiment.learner._wordsp.all_words(0)])) *\
            np.divide(joint_probs[('entity#1', 'gumdrop:N')],
            np.sum([joint_probs[('entity#1', word)] for word in self.experiment.learner._wordsp.all_words(0)])) *\
            np.divide(joint_probs[('artifact#1', 'gumdrop:N')],
            np.sum([joint_probs[('artifact#1', word)] for word in self.experiment.learner._wordsp.all_words(0)])) *\
            np.divide(joint_probs[('model#4', 'gumdrop:N')],
            np.sum([joint_probs[('model#4', word)] for word in self.experiment.learner._wordsp.all_words(0)])) *\
            np.divide(joint_probs[('representation#2', 'gumdrop:N')],
            np.sum([joint_probs[('representation#2', word)] for word in self.experiment.learner._wordsp.all_words(0)])) *\
            np.divide(joint_probs[('physical entity#1', 'gumdrop:N')],
            np.sum([joint_probs[('physical entity#1', word)] for word in self.experiment.learner._wordsp.all_words(0)])) *\
            np.divide(joint_probs[('object#1', 'gumdrop:N')],
            np.sum([joint_probs[('object#1', word)] for word in self.experiment.learner._wordsp.all_words(0)])) *\
            np.divide(joint_probs[('matter#3', 'gumdrop:N')],
            np.sum([joint_probs[('matter#3', word)] for word in self.experiment.learner._wordsp.all_words(0)])) *\
            np.divide(joint_probs[('figure#4', 'gumdrop:N')],
            np.sum([joint_probs[('figure#4', word)] for word in self.experiment.learner._wordsp.all_words(0)]))

        mul_ref_prob_gumdrop = np.divide(joint_probs[('gumdrop', 'gumdrop:N')],
            np.sum([joint_probs[('gumdrop', word)] for word in self.experiment.learner._wordsp.all_words(0)])) *\
            np.divide(joint_probs[('sweet#3', 'gumdrop:N')],
            np.sum([joint_probs[('sweet#3', word)] for word in self.experiment.learner._wordsp.all_words(0)])) *\
            np.divide(joint_probs[('entity#1', 'gumdrop:N')],
            np.sum([joint_probs[('entity#1', word)] for word in self.experiment.learner._wordsp.all_words(0)])) *\
            np.divide(joint_probs[('food#1', 'gumdrop:N')],
            np.sum([joint_probs[('food#1', word)] for word in self.experiment.learner._wordsp.all_words(0)])) *\
            np.divide(joint_probs[('dainty#1', 'gumdrop:N')],
            np.sum([joint_probs[('dainty#1', word)] for word in self.experiment.learner._wordsp.all_words(0)])) *\
            np.divide(joint_probs[('substance#7', 'gumdrop:N')],
            np.sum([joint_probs[('substance#7', word)] for word in self.experiment.learner._wordsp.all_words(0)])) *\
            np.divide(joint_probs[('physical entity#1', 'gumdrop:N')],
            np.sum([joint_probs[('physical entity#1', word)] for word in  self.experiment.learner._wordsp.all_words(0)])) *\
            np.divide(joint_probs[('nutriment#2', 'gumdrop:N')],
            np.sum([joint_probs[('nutriment#2', word)] for word in self.experiment.learner._wordsp.all_words(0)])) *\
            np.divide(joint_probs[('candy#1', 'gumdrop:N')],
            np.sum([joint_probs[('candy#1', word)] for word in self.experiment.learner._wordsp.all_words(0)])) *\
            np.divide(joint_probs[('whole#2', 'gumdrop:N')],
            np.sum([joint_probs[('whole#2', word)] for word in self.experiment.learner._wordsp.all_words(0)]))

        sum_ref_prob_snowman = np.divide(np.sum([
            joint_probs[('snowman', 'gumdrop:N')],
            joint_probs[('creation#2', 'gumdrop:N')],
            joint_probs[('entity#1', 'gumdrop:N')],
            joint_probs[('artifact#1', 'gumdrop:N')],
            joint_probs[('model#4', 'gumdrop:N')],
            joint_probs[('representation#2', 'gumdrop:N')],
            joint_probs[('physical entity#1', 'gumdrop:N')],
            joint_probs[('object#1', 'gumdrop:N')],
            joint_probs[('matter#3', 'gumdrop:N')],
            joint_probs[('figure#4', 'gumdrop:N')]]),
            np.sum([
                np.sum([joint_probs[('snowman', word)] for word in self.experiment.learner._wordsp.all_words(0)]),
                np.sum([joint_probs[('creation#2', word)] for word in self.experiment.learner._wordsp.all_words(0)]),
                np.sum([joint_probs[('entity#1', word)] for word in self.experiment.learner._wordsp.all_words(0)]),
                np.sum([joint_probs[('artifact#1', word)] for word in self.experiment.learner._wordsp.all_words(0)]),
                np.sum([joint_probs[('model#4', word)] for word in self.experiment.learner._wordsp.all_words(0)]),
                np.sum([joint_probs[('representation#2', word)] for word in self.experiment.learner._wordsp.all_words(0)]),
                np.sum([joint_probs[('physical entity#1', word)] for word in self.experiment.learner._wordsp.all_words(0)]),
                np.sum([joint_probs[('object#1', word)] for word in self.experiment.learner._wordsp.all_words(0)]),
                np.sum([joint_probs[('matter#3', word)] for word in self.experiment.learner._wordsp.all_words(0)]),
                np.sum([joint_probs[('figure#4', word)] for word in self.experiment.learner._wordsp.all_words(0)])
            ]))

        sum_ref_prob_gumdrop = np.divide(np.sum([
            joint_probs[('gumdrop', 'gumdrop:N')],
            joint_probs[('sweet#3', 'gumdrop:N')],
            joint_probs[('entity#1', 'gumdrop:N')],
            joint_probs[('food#1', 'gumdrop:N')],
            joint_probs[('dainty#1', 'gumdrop:N')],
            joint_probs[('substance#7', 'gumdrop:N')],
            joint_probs[('physical entity#1', 'gumdrop:N')],
            joint_probs[('nutriment#2', 'gumdrop:N')],
            joint_probs[('candy#1', 'gumdrop:N')],
            joint_probs[('whole#2', 'gumdrop:N')]]),
            np.sum([
                np.sum([joint_probs[('gumdrop', word)] for word in self.experiment.learner._wordsp.all_words(0)]),
                np.sum([joint_probs[('sweet#3', word)] for word in self.experiment.learner._wordsp.all_words(0)]),
                np.sum([joint_probs[('entity#1', word)] for word in self.experiment.learner._wordsp.all_words(0)]),
                np.sum([joint_probs[('food#1', word)] for word in self.experiment.learner._wordsp.all_words(0)]),
                np.sum([joint_probs[('dainty#1', word)] for word in self.experiment.learner._wordsp.all_words(0)]),
                np.sum([joint_probs[('substance#7', word)] for word in self.experiment.learner._wordsp.all_words(0)]),
                np.sum([joint_probs[('physical entity#1', word)] for word in self.experiment.learner._wordsp.all_words(0)]),
                np.sum([joint_probs[('nutriment#2', word)] for word in self.experiment.learner._wordsp.all_words(0)]),
                np.sum([joint_probs[('candy#1', word)] for word in self.experiment.learner._wordsp.all_words(0)]),
                np.sum([joint_probs[('whole#2', word)] for word in self.experiment.learner._wordsp.all_words(0)])
            ]))

        expected_mul = {}
        expected_mul[('gumdrop:N', 'gumdrop:N')] = mul_ref_prob_gumdrop
        expected_mul[('gumdrop:N', 'snowman:N')] = mul_ref_prob_snowman

        expected_sum = {}
        expected_sum[('gumdrop:N', 'gumdrop:N')] = sum_ref_prob_gumdrop
        expected_sum[('gumdrop:N', 'snowman:N')] = sum_ref_prob_snowman

        self.assertDeepAlmostEqual(actual_mul, expected_mul)
        self.assertDeepAlmostEqual(actual_sum, expected_sum)

        os.remove('toy_lex')
        self.experiment.learner.reset()

class testExperimentalSetup(unittest.TestCase):

    def setUp(self):
        # create the experiment object
        self.experiment = referent_probability_experiments.NovelReferentExperiment()

        # create the config file
        config_string = """[Smoothing]
beta=10000
lambda=-1
power=1
epsilon=0.01
alpha=20

[Similarity]
simtype=COS
theta=0.7

[Features]
dummy=false
forget=false
forget-decay=0
novelty=false
novelty-decay=0
assoc-type=SUM
category=false
semantic-network=false
hub-type=hub-freq-degree
hub-num=75

[Statistics]
stats=true
context-stats=false
familiarity-smoothing=0.01
familiarity-measure=COUNT
age-of-exposure-norm=100
maxtime=1000"""

        with open('toy_config', 'w') as f:
            f.write(config_string)

        self.learner_config = learnconfig.LearnerConfig('toy_config')

        self.experiment.learner = learn.Learner('all_catf_prob_lexicon_cs.all', self.learner_config, [])

    def tearDown(self):
        os.remove('toy_config')

    def test_five_features_probabilistic_no_novel(self):
        self.experiment.learner.process_corpus('test_input_wn_fu_cs_scaled_categ.dev_without_gumdrop:N', './')
        self.experiment.familiar_features = set(
            list(itertools.chain.from_iterable(
                [self.experiment.learner._learned_lexicon.seen_features(word) \
                for word in self.experiment.learner._wordsp.all_words(0)])
            )
        )

        params = {}
        params['path'] = './'
        params['probabilistic'] = True
        params['n-features'] = 5
        params['prop-novel-features'] = 0.0
        params['prop-overlapping-features'] = 0.2

        self.experiment.familiar_objects = ['snowman:N']
        self.experiment.novel_word = 'gumdrop:N'
        self.experiment.utterance = ['gumdrop:N']
        results = self.experiment.iterate(params, 1, 1)

        self.assertEqual(len(results['overlapping feature(s)']) + len(results['scene']),
            params['n-features']*2)

        for feature in results['scene']:
            self.assertEqual(True, feature in self.experiment.familiar_features)

    def test_ten_features_probabilistic_no_novel(self):
        self.experiment.learner.process_corpus('test_input_wn_fu_cs_scaled_categ.dev_without_milk:N', './')
        self.experiment.familiar_features = set(
            list(itertools.chain.from_iterable(
                [self.experiment.learner._learned_lexicon.seen_features(word) \
                for word in self.experiment.learner._wordsp.all_words(0)])
            )
        )

        params = {}
        params['path'] = './'
        params['probabilistic'] = True
        params['n-features'] = 10
        params['prop-novel-features'] = 0.0
        params['prop-overlapping-features'] = 0.1

        self.experiment.familiar_objects = ['cat:N']
        self.experiment.novel_word = 'milk:N'
        self.experiment.utterance = ['milk:N']
        results = self.experiment.iterate(params, 1, 1)

        self.assertEqual(len(results['overlapping feature(s)']) + len(results['scene']),
            params['n-features']*2)

        for feature in results['scene']:
            self.assertEqual(True, feature in self.experiment.familiar_features)

    def test_five_features_probabilistic_one_novel(self):
        self.experiment.learner.process_corpus('test_input_wn_fu_cs_scaled_categ.dev_without_gumdrop:N', './')
        self.experiment.familiar_features = set(
            list(itertools.chain.from_iterable(
                [self.experiment.learner._learned_lexicon.seen_features(word) \
                for word in self.experiment.learner._wordsp.all_words(0)])
            )
        )

        params = {}
        params['path'] = './'
        params['probabilistic'] = True
        params['n-features'] = 5
        params['prop-novel-features'] = 0.2
        params['prop-overlapping-features'] = 0.2

        self.experiment.familiar_objects = ['snowman:N']
        self.experiment.novel_word = 'gumdrop:N'
        self.experiment.utterance = ['gumdrop:N']
        results = self.experiment.iterate(params, 1, 1)

        self.assertEqual(len(results['overlapping feature(s)']) + len(results['scene']),
            params['n-features']*2)

        familiar_count = 0
        novel_features = []
        for feature in results['scene']:
            if feature in self.experiment.familiar_features:
                familiar_count += 1
            else:
                novel_features.append(feature)
        self.assertEqual(10, len(results['overlapping feature(s)']) + familiar_count + params['n-features'] * params['prop-novel-features'])
        self.assertEqual(True, np.all([feature in self.experiment.learner._gold_lexicon.seen_features('gumdrop:N') or feature.startswith('novel') for feature in novel_features]))

    def test_ten_features_probabilistic_two_novel(self):
        self.experiment.learner.process_corpus('test_input_wn_fu_cs_scaled_categ.dev_without_milk:N', './')
        self.experiment.familiar_features = set(
            list(itertools.chain.from_iterable(
                [self.experiment.learner._learned_lexicon.seen_features(word) \
                for word in self.experiment.learner._wordsp.all_words(0)])
            )
        )

        params = {}
        params['path'] = './'
        params['probabilistic'] = True
        params['n-features'] = 10
        params['prop-novel-features'] = 0.2
        params['prop-overlapping-features'] = 0.1

        self.experiment.familiar_objects = ['cat:N']
        self.experiment.novel_word = 'milk:N'
        self.experiment.utterance = ['milk:N']
        results = self.experiment.iterate(params, 1, 1)

        self.assertEqual(len(results['overlapping feature(s)']) + len(results['scene']),
            params['n-features']*2)

        familiar_count = 0
        novel_features = []
        for feature in results['scene']:
            if feature in self.experiment.familiar_features:
                familiar_count += 1
            else:
                novel_features.append(feature)
        self.assertEqual(20, len(results['overlapping feature(s)']) + familiar_count + params['n-features'] * params['prop-novel-features'])
        self.assertEqual(True, np.all([feature in self.experiment.learner._gold_lexicon.seen_features('milk:N') or feature.startswith('novel') for feature in novel_features]))

    def test_five_features_not_probabilistic_no_novel(self):
        self.experiment.learner.process_corpus('test_input_wn_fu_cs_scaled_categ.dev_without_gumdrop:N', './')
        self.experiment.familiar_features = set(
            list(itertools.chain.from_iterable(
                [self.experiment.learner._learned_lexicon.seen_features(word) \
                for word in self.experiment.learner._wordsp.all_words(0)])
            )
        )

        params = {}
        params['path'] = './'
        params['probabilistic'] = False
        params['n-features'] = 5
        params['prop-novel-features'] = 0.0
        params['prop-overlapping-features'] = 0.2

        self.experiment.familiar_objects = ['snowman:N']
        self.experiment.novel_word = 'gumdrop:N'
        self.experiment.utterance = ['gumdrop:N']
        results = self.experiment.iterate(params, 1, 1)

        self.assertEqual(len(results['overlapping feature(s)']) + len(results['scene']),
            params['n-features']*2)

        for feature in results['scene']:
            self.assertEqual(True, feature in self.experiment.familiar_features)

    def test_ten_features_not_probabilistic_no_novel(self):
        self.experiment.learner.process_corpus('test_input_wn_fu_cs_scaled_categ.dev_without_milk:N', './')
        self.experiment.familiar_features = set(
            list(itertools.chain.from_iterable(
                [self.experiment.learner._learned_lexicon.seen_features(word) \
                for word in self.experiment.learner._wordsp.all_words(0)])
            )
        )

        params = {}
        params['path'] = './'
        params['probabilistic'] = False
        params['n-features'] = 10
        params['prop-novel-features'] = 0.0
        params['prop-overlapping-features'] = 0.1

        self.experiment.familiar_objects = ['cat:N']
        self.experiment.novel_word = 'milk:N'
        self.experiment.utterance = ['milk:N']
        results = self.experiment.iterate(params, 1, 1)

        self.assertEqual(len(results['overlapping feature(s)']) + len(results['scene']),
            params['n-features']*2)

        for feature in results['scene']:
            self.assertEqual(True, feature in self.experiment.familiar_features)

    def test_five_features_not_probabilistic_one_novel(self):
        self.experiment.learner.process_corpus('test_input_wn_fu_cs_scaled_categ.dev_without_gumdrop:N', './')
        self.experiment.familiar_features = set(
            list(itertools.chain.from_iterable(
                [self.experiment.learner._learned_lexicon.seen_features(word) \
                for word in self.experiment.learner._wordsp.all_words(0)])
            )
        )

        params = {}
        params['path'] = './'
        params['probabilistic'] = False
        params['n-features'] = 5
        params['prop-novel-features'] = 0.2
        params['prop-overlapping-features'] = 0.2

        self.experiment.familiar_objects = ['snowman:N']
        self.experiment.novel_word = 'gumdrop:N'
        self.experiment.utterance = ['gumdrop:N']
        results = self.experiment.iterate(params, 1, 1)

        self.assertEqual(len(results['overlapping feature(s)']) + len(results['scene']),
            params['n-features']*2)

        familiar_count = 0
        novel_features = []
        for feature in results['scene']:
            if feature in self.experiment.familiar_features:
               familiar_count += 1
            else:
                novel_features.append(feature)

        self.assertEqual(10, len(results['overlapping feature(s)']) + familiar_count + params['n-features'] * params['prop-novel-features'])
        self.assertEqual(True, np.all([feature in self.experiment.learner._gold_lexicon.seen_features('gumdrop:N') or feature.startswith('novel') for feature in novel_features]))

    def test_ten_features_not_probabilistic_two_novel(self):
        self.experiment.learner.process_corpus('test_input_wn_fu_cs_scaled_categ.dev_without_milk:N', './')
        self.experiment.familiar_features = set(
            list(itertools.chain.from_iterable(
                [self.experiment.learner._learned_lexicon.seen_features(word) \
                for word in self.experiment.learner._wordsp.all_words(0)])
            )
        )

        params = {}
        params['path'] = './'
        params['probabilistic'] = False
        params['n-features'] = 10
        params['prop-novel-features'] = 0.2
        params['prop-overlapping-features'] = 0.1

        self.experiment.familiar_objects = ['cat:N']
        self.experiment.novel_word = 'milk:N'
        self.experiment.utterance = ['milk:N']
        results = self.experiment.iterate(params, 1, 1)

        self.assertEqual(len(results['overlapping feature(s)']) + len(results['scene']),
            params['n-features']*2)

        familiar_count = 0
        novel_features = []
        for feature in results['scene']:
            if feature in self.experiment.familiar_features:
                familiar_count += 1
            else:
                novel_features.append(feature)

        self.assertEqual(20, len(results['overlapping feature(s)']) + familiar_count + params['n-features'] * params['prop-novel-features'])
        self.assertEqual(True, np.all([feature in self.experiment.learner._gold_lexicon.seen_features('milk:N') or feature.startswith('novel') for feature in novel_features]))

class testExperimentalResults(unittest.TestCase):

    def setUp(self):
        # create the experiment object
        self.experiment = referent_probability_experiments.NovelReferentExperiment()

        # create the config file
        config_string = """[Smoothing]
beta=10000
lambda=-1
power=1
epsilon=0.01
alpha=20

[Similarity]
simtype=COS
theta=0.7

[Features]
dummy=false
forget=false
forget-decay=0
novelty=false
novelty-decay=0
assoc-type=SUM
category=false
semantic-network=false
hub-type=hub-freq-degree
hub-num=75

[Statistics]
stats=true
context-stats=false
familiarity-smoothing=0.01
familiarity-measure=COUNT
age-of-exposure-norm=100
maxtime=1000"""

        with open('toy_config', 'w') as f:
            f.write(config_string)

        self.learner_config = learnconfig.LearnerConfig('toy_config')
        self.experiment.learner = learn.Learner('all_catf_prob_lexicon_cs.all', self.learner_config, [])
        self.experiment.learner.process_corpus('test_input_wn_fu_cs_scaled_categ.dev_without_gumdrop:N', './')

    def tearDown(self):
        os.remove('toy_config')

    def test_ratio_calculations(self):
        self.experiment.familiar_features = set(
            list(itertools.chain.from_iterable(
                [self.experiment.learner._learned_lexicon.seen_features(word) \
                for word in self.experiment.learner._wordsp.all_words(0)])
            )
        )

        params = {}
        params['path'] = './'
        params['probabilistic'] = False
        params['n-features'] = 2
        params['prop-novel-features'] = 0.2
        params['prop-overlapping-features'] = 0.0

        self.experiment.familiar_objects = ['milk:N']
        self.experiment.novel_word = 'gumdrop:N'
        self.experiment.utterance = ['gumdrop:N']
        results = self.experiment.iterate(params, 1, 1)

        self.assertEqual(results['ratio (SUM)'], results['novel referent (SUM)']/results['familiar referent (SUM)'])
        self.assertEqual(results['ratio (PROD)'], results['novel referent (MUL)']/results['familiar referent (MUL)'])

if __name__ == '__main__':
    unittest.main()
