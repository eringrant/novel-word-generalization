#!/usr/bin/python
from __future__ import print_function, division

import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter
import xml.etree.cElementTree as ET
import itertools
import pprint

import learn
import learnconfig
import wmmapping
import experiment
import experimental_materials

make_inputs = False

class GeneralisationExperiment(experiment.Experiment):

    def setup(self, params, rep):

        if params['forget'] is not False:
            self.forget = True
            self.forget_decay = params['forget']
        else:
            self.forget = False
            self.forget_decay = 0

        # novelty
        if params['novelty'] is not False:
            self.novelty = True
            self.novelty_decay = params['novelty']
        else:
            self.novelty = False
            self.novelty_decay = 0

        # create temporary config file
        config_filename = 'temp_config_'
        config_filename += '_'.join([str(value) for (param, value) \
                in sorted(params.items()) if len(str(value)) < 6])
        config_filename += '.ini'

        self.config_path = experimental_materials.write_config_file(
            config_filename,
            dummy=params['dummy'],
            forget=self.forget,
            forget_decay=self.forget_decay,
            novelty=self.novelty,
            novelty_decay=self.novelty_decay,
            L=params['lambda'],
            power=params['power'],
            maxtime=params['maxtime']
        )

        # create and teach the learner
        learner_config = learnconfig.LearnerConfig(self.config_path)
        stopwords = []
        self.learner = learn.Learner(params['lexname'], learner_config, stopwords)

        # access the hierarchy
        tree = ET.parse(params['hierarchy'])
        self.lexicon = self.create_lexicon_from_etree(tree)

        # get the corpus
        if params['corpus-path'] == 'generate':
            corpus = self.generate_corpus(tree, params)
        else:
            corpus = params['corpus-path']

        self.learner.process_corpus(corpus, params['path'])

        # training_sets is a dictionary of condition to a set of three
        # differnet training sets
        self.training_sets = {}

        self.training_sets['one example'] = [
            [experimental_materials.UtteranceScenePair(
                utterance='fep',
                objects=['green pepper'],
                lexicon=self.lexicon,
                probabilistic=False
            )],
            [experimental_materials.UtteranceScenePair(
                utterance='fep',
                objects=['tow-truck'],
                lexicon=self.lexicon,
                probabilistic=False
            )],
            [experimental_materials.UtteranceScenePair(
                utterance='fep',
                objects=['dalmatian'],
                lexicon=self.lexicon,
                probabilistic=False
            )]
        ]

        self.training_sets['three subordinate examples'] = [
            [experimental_materials.UtteranceScenePair(
                utterance='fep',
                objects=['green pepper'],
                lexicon=self.lexicon,
                probabilistic=False
            )] * 3,
            [experimental_materials.UtteranceScenePair(
                utterance='fep',
                objects=['tow-truck'],
                lexicon=self.lexicon,
                probabilistic=False
            )] * 3,
            [experimental_materials.UtteranceScenePair(
                utterance='fep',
                objects=['dalmatian'],
                lexicon=self.lexicon,
                probabilistic=False
            )] * 3
        ]

        self.training_sets['three basic-level examples'] = [
            [experimental_materials.UtteranceScenePair(
                utterance='fep',
                objects=[obj],
                lexicon=self.lexicon,
                probabilistic=False
            ) for obj in ['green pepper', 'yellow pepper', 'red pepper']],
            [experimental_materials.UtteranceScenePair(
                utterance='fep',
                objects=[obj],
                lexicon=self.lexicon,
                probabilistic=False
            ) for obj in ['tow-truck', 'fire-truck', 'semitrailer']],
            [experimental_materials.UtteranceScenePair(
                utterance='fep',
                objects=[obj],
                lexicon=self.lexicon,
                probabilistic=False
            ) for obj in ['dalmatian', 'poodle', 'pug']]
        ]

        self.training_sets['three superordinate examples'] = [
            [experimental_materials.UtteranceScenePair(
                utterance='fep',
                objects=[obj],
                lexicon=self.lexicon,
                probabilistic=False
            ) for obj in ['green pepper', 'potato', 'zucchini']],
            [experimental_materials.UtteranceScenePair(
                utterance='fep',
                objects=[obj],
                lexicon=self.lexicon,
                probabilistic=False
            ) for obj in ['tow-truck', 'airliner', 'sailboat']],
            [experimental_materials.UtteranceScenePair(
                utterance='fep',
                objects=[obj],
                lexicon=self.lexicon,
                probabilistic=False
            ) for obj in ['dalmatian', 'tabby', 'salmon']]
        ]

        #pprint.pprint(self.training_sets)

        # there are three relevant sets, correpsonding to the three
        # training sests for eahc confdition
        # i.e., we do not computer
        self.test_sets = [{}, {}, {}]
        self.test_sets[0]['subordinate matches'] = [
            'green pepper',
            'green pepper'
        ]
        self.test_sets[1]['subordinate matches'] = [
            'tow-truck',
            'tow-truck'
        ]
        self.test_sets[2]['subordinate matches'] = [
            'dalmatian',
            'dalmatian'
        ]
        self.test_sets[0]['basic-level matches'] = [
            'red pepper',
            'yellow pepper'
        ]
        self.test_sets[1]['basic-level matches'] = [
            'fire-truck',
            'semitrailer'
        ]
        self.test_sets[2]['basic-level matches'] = [
            'poodle',
            'pug'
        ]
        self.test_sets[0]['superordinate matches'] = [
            'potato',
            'zucchini'
        ]
        self.test_sets[1]['superordinate matches'] = [
            'airliner',
            'sailboat'
        ]
        self.test_sets[2]['superordinate matches'] = [
            'tabby',
            'salmon'
        ]

        return True

    def iterate(self, params, rep, n):

        results = {}

        for condition in self.teaching_sets:
            results[condition] = {}

            for i, teaching_set in enumerate(self.teaching_sets[condition]):
                learner = copy.deepcopy(self.learner)
                learner.process_pair(teaching_set.utterance(), teaching_set.scene(), params['path'], False)

                for cond in test_sets[i]:
                    for j in len(test_sets[i][cond]):
                        test_scene = test_sets[i][cond][j]
                        gen_prob = calculate_generalisation_probability(learner, 'fep', test_sets[i][cond][j])
                        try:
                            results[condition][cond].append(gen_prob)
                        except KeyError:
                            results[condition][cond] = []
                            results[condition][cond].append(gen_prob)

        pprint.pprint(results)

    def generate_corpus(self, tree, params):
        """
        @param tree An ElementTree instance.
        """
        corpus_path = 'temp_xt_corpus.dev'
        temp_corpus = open(corpus_path, 'w')

        root = tree.getroot()

        bag_of_words = []
        word_to_features_map = {}

        num_superordinate = params['num-superordinate'] // len(root.findall('.//superordinate'))
        num_basic = params['num-basic-level'] // len(root.findall('.//basic-level'))
        num_subordinate = params['num-subordinate'] // len(root.findall('.//subordinate'))

        for sup in root.findall('.//superordinate'):

            bag_of_words.extend([sup.get('label')] * num_superordinate)
            word_to_features_map[sup.get('label')] = sup.get('features').split(' ')

            for basic in sup.findall('.//basic-level'):

                bag_of_words.extend([basic.get('label')] * num_basic)
                word_to_features_map[basic.get('label')] = basic.get('features').split(' ')

                for sub in basic.findall('.//subordinate'):

                    bag_of_words.extend([sub.get('label')] * num_subordinate)
                    word_to_features_map[sub.get('label')] = sub.get('features').split(' ')

        np.random.shuffle(bag_of_words)

        for word in bag_of_words:
            feature_choices = word_to_features_map[word]

            if params['probabilistic'] is True:
                s = np.random.randint(1, len(feature_choices)+1)
                scene = list(np.random.choice(a=feature_choices, size=s, replace=False))
            else:
                scene = feature_choices[:]

            temp_corpus.write("1-----\nSENTENCE: ")
            temp_corpus.write(word)
            temp_corpus.write('\n')
            temp_corpus.write("SEM_REP:  ")
            for ft in scene:
                temp_corpus.write("," + ft)
            temp_corpus.write('\n')

        temp_corpus.close()
        return corpus_path

    def create_lexicon_from_etree(self, tree):
        root = tree.getroot()
        lexicon = wmmapping.Lexicon(self.learner._beta, [])

        for level in ['superordinate', 'basic-level', 'subordinate']:
            for node in root.findall('.//'+level):
                word = node.get('label')
                features = node.get('features').split(' ')
                for feature in features:
                    lexicon.set_prob(word, feature, 1/len(features))

        return lexicon

def bar_chart(scores, test_items):
    sorting = [
        'dalmatian0T',
        'dalmatian1T',
        'poodle0T',
        'pug0T',
        'tabby0T',
        'manx0T',
        'flounder0T',
        'motorboat0T',
        'fire-truck0T'
        ]

    subordinate = [
        'dalmatian0T',
        'dalmatian1T'
        ]

    basic = [
        'poodle0T',
        'pug0T'
        ]

    superordinate = [
        'tabby0T',
        'manx0T',
        'flounder0T'
        ]

    l0 = [scores[0][i] for i in sorting]
    l1 = [scores[1][i] for i in sorting]
    l2 = [scores[2][i] for i in sorting]
    l3 = [scores[3][i] for i in sorting]

    ind = np.array([2.5*n for n in range(len(test_items))])
    width = 0.5

    fig = plt.figure()
    ax = fig.add_subplot(111)

    p0 = ax.bar(ind,l0,width,color='r')
    p1 = ax.bar(ind+width,l1,width,color='g')
    p2 = ax.bar(ind+2*width,l2,width,color='b')
    p3 = ax.bar(ind+3*width,l3,width,color='y')
    print(l1)

    ax.set_ylabel("total score")
    ax.set_xlabel("test item")
    ax.set_xticks(ind + 2.5 * width)
    ax.set_xticklabels(sorting,rotation=45)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax.legend( (p0, p1, p2, p3), ('3 * 1 Dalmatian', '3 * subordinate', '3 * basic', '3 * superordinate') , loc='center left', bbox_to_anchor=(1, 0.5))

    title = "Generalization scores\n Beta=" + str(Beta) + " Lambda=" + str(Lambda)
    ax.set_title(title)

    plt.show()

    #l0 = [np.mean([scores[0][i] for i in subordinate]), np.mean([scores[0][i] for i in basic]), np.mean([scores[0][i] for i in superordinate])]
    #l1 = [np.mean([scores[1][i] for i in subordinate]), np.mean([scores[1][i] for i in basic]), np.mean([scores[1][i] for i in superordinate])]
    #l2 = [np.mean([scores[2][i] for i in subordinate]), np.mean([scores[2][i] for i in basic]), np.mean([scores[2][i] for i in superordinate])]
    #l3 = [np.mean([scores[3][i] for i in subordinate]), np.mean([scores[3][i] for i in basic]), np.mean([scores[3][i] for i in superordinate])]
    #print(l0)

    l0 = [np.mean([scores[0][i] for i in subordinate]), np.mean([scores[1][i] for i in subordinate]), np.mean([scores[2][i] for i in subordinate]), np.mean([scores[3][i] for i in subordinate])]
    l1 = [np.mean([scores[0][i] for i in basic]), np.mean([scores[1][i] for i in basic]), np.mean([scores[2][i] for i in basic]), np.mean([scores[3][i] for i in basic])]
    l2 = [np.mean([scores[0][i] for i in superordinate]), np.mean([scores[1][i] for i in superordinate]), np.mean([scores[2][i] for i in superordinate]), np.mean([scores[3][i] for i in superordinate])]

    ind = np.array([2.5*n for n in range(4)])
    width = 0.5

    fig = plt.figure()
    ax = fig.add_subplot(111)

    p0 = ax.bar(ind,l0,width,color='w')
    p1 = ax.bar(ind+width,l1,width,color='y')
    p2 = ax.bar(ind+2*width,l2,width,color='k')

    ax.set_ylabel("mean similarity measure")
    ax.set_xlabel("test condition")
    ax.set_xticks(ind + 2.5 * width)

    xlabels = ('3 * a single Dalmatian', '3 subordinate instances', '3 basic instances', '3 superordinate instances')
    ax.set_xticklabels(xlabels)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax.legend( (p0, p1, p2), ('generalisation to subordinate', 'generalisation to basic', 'generalisation to superordinate'), loc='center left', bbox_to_anchor=(1, 0.5))

    title = "Generalization scores\n Beta=" + str(Beta) + " Lambda=" + str(Lambda)
    ax.set_title(title)

    plt.show()

def calculate_generalisation_probability(learner, target_word, item):

    total = np.float128(0)
    for word in learner._wordsp.all_words(0):
        f_in_y = np.sum([learner._learned_lexicon(word, feature) for feature in item])
        f_in_word = np.sum([learner._learned_lexicon(word, feature) for feature in learner._learned_lexicon.seen_features(target_word)])
        word_freq = learner._wordsp.frequency(word)
        total += f_in_y * f_in_word * word_freq
    return total

def main(repkl=True, rescore=False):
    global tsn
    if repkl:
        # learn the meanings of words from input corpus, and update the learning curves
        learner = learn.Learner(Beta, Lambda, alpha, epsilon, simtype, theta, lexname, outdir, add_dummy, traceword, minfreq)
        (j1, j2, rfD) = learner.processCorpus(corpus, add_dummy, maxsents, 10000)
        f = open("learner.pkl",'w')
        pickle.dump(learner,f)
        f.close()

    #outfile = open("RESULT.TXT",'w')
    all_prims = []
    probs = {}
    all_scores = []

    log = open("out.log",'w')
    i = 0

    if rescore:
    # run experiments
        for t in teaching_sets:

            f = open("learner.pkl",'r')
            exp_learner = pickle.load(f)
            f.close()

            teach(exp_learner, t, log)
            all_scores.append(test(exp_learner, log))
            tsn += 1

        scorefile = open("scores.pkl",'w')
        pickle.dump(all_scores, scorefile)
        scorefile.close()

    scorefile = open("scores.pkl",'r')
    all_scores = pickle.load(scorefile)
    scorefile.close()

    log.close()
    bar_chart(all_scores,test_set)

if __name__ == "__main__":
    e = GeneralisationExperiment()
    e.start()
