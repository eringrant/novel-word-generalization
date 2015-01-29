import copy
import math
import numpy
import re
import sys
import os
import constants as CONST
from category import *
import input
import wmmapping
import statistics
import evaluate
import wgraph
"""
learn.py


"""

class Learner:
    """
    Encapsulate all learning and model updating functionality.

    Members:
        * See config.ini for a summary of the configuration members of Learner
        gold_lexicon -- gold standard Lexicon to compare the learned lexicon to
        lexicon -- learned Lexicon
        aligns -- learned Alignments
        wordsp -- WordPropsTable for recording word type statistics
        timesp -- TimePropsFile for recording time step statistics
        time -- record time steps based on the processing of input utterance-scene
            pairs
        vocab -- set of learned words (using a set to enforce uniqueness)
        features -- set of features seen so far (using a set to enforce uniqueness)
        acquisition_scores -- Dictionary of words to acquisition scores (used to
            avoid recalculating acquisition scores when possible)
        last_time -- Dictionary of words and the last time that they had been
            encountered. Used for novelty decaying.
        stopwords -- words to avoid for novelty-related computations

    """

    def __init__(self, lexicon_path, config, stopwords=[], k_sub=None,
            k_basic=None, k_sup=None, alpha_sub=None, alpha_basic=None,
            alpha_sup=None):
        """
        Initialize the Learner with all properties from LearnerConfig config,
        using lexicon_path to initialize the gold standard lexicon and
        stopwords_path to read the file of all stop words to ignore.

        """
        self.k_sub = k_sub
        self.k_basic = k_basic
        self.k_sup = k_sup

        self.alpha_sub = alpha_sub
        self.alpha_basic = alpha_basic
        self.alpha_sup = alpha_sup

        if not os.path.exists(lexicon_path):
            print "Initialization Error -- Lexicon does not exist : "+lexicon_path
            sys.exit(2)

        if config is None:
            print "Initialization Error -- Config required"
            sys.exit(2)

        # Begin configuration of the learner based on config

        # Smoothing
        self._beta = config.param_float("beta")
        if self._beta < 0:
            print "Config Error [beta] Must be non-zero positive : "+str(self._beta)
            sys.exit(2)

        self._lambda = config.param_float("lambda")
        self._power = config.param_float("power")
        if self._lambda > 1 and self._power <= 0:
            print "Config Error [lambda] [power]"
            print "\t lambda: " + self._lambda + ", power: " + self._power
            sys.exit(2)

        self._alpha = config.param_float("alpha")
        if self._alpha == 0:
            #print "Config Warning [alpha] No alpha smoothing"
            pass
        elif self._alpha < 0:
            #print "Config [alpha] Must be positive : "+str(self._alpha)
            #sys.exit(2)
            print "Using frequency-dependent alpha."

        self._epsilon = config.param_float("epsilon")
        if self._epsilon < 0:
            print "Config Error [epsilon] Must be non-zero positive : "+str(self._epsilon)
            sys.exit(2)

        # Similarity
        self._theta = config.param_float("theta")
        if self._theta < 0:
            print "Config Error [theta] Must be non-zero positive : "+str(self._theta)
            sys.exit(2)
        self._simtype = config.param("simtype")
        if self._simtype not in CONST.ALL_SIM_TYPES:
            print "Config Error [simtype] Invalid simtype : "+str(self._simtype)
            sys.exit(2)

        # Features
        self._dummy = config.param_bool("dummy")
        self._forget_flag = config.param_bool("forget")
        self._novelty_flag = config.param_bool("novelty")
        self._forget_decay = config.param_float("forget-decay")
        self._novelty_decay = config.param_float("novelty-decay")
        if self._forget_flag and self._forget_decay <= 0:
            x = str(self._forget_decay)
            print "Config Error [forget-decay] Must be non-zero positive : "+x
            sys.exit(2)
        if self._novelty_flag and self._novelty_decay <= 0:
            x = str(self._novelty_decay)
            print "Config Error [novelty-decay] Must be non-zero positive : "+x
            sys.exit(2)
        self._assoc_type = config.param("assoc-type")
        if self._assoc_type not in CONST.ALL_ASSOC_TYPES:
            print "Config Error [assoc-type] Invalid assoc-type : "+str(self._assoc_type)
            sys.exit(2)
        self._category_flag = config.param_bool("category")

        # Statistics
        self._stats_flag = config.param_bool("stats")
        self._context_stats_flag = config.param_bool("context-stats")

        self._postags = set()
        if config.has_param("tag1"):
            if config.param("tag1") not in CONST.ALL_TAGS:
                print "Config Error [tag1] Invalid : " + config.param("tag1")
                sys.exit(2)
            self._postags.add(config.param("tag1"))
        if config.has_param("tag2"):
            if config.param("tag2") not in CONST.ALL_TAGS:
                print "Config Error [tag2] Invalid : " + config.param("tag2")
                sys.exit(2)
            self._postags.add(config.param("tag2"))
        if config.has_param("tag3"):
            if config.param("tag3") not in CONST.ALL_TAGS:
                print "Config Error [tag3] Invalid : " + config.param("tag3")
                sys.exit(2)
            self._postags.add(config.param("tag3"))

        if self._stats_flag:
            self._wordsp = statistics.WordPropsTable(config.param("word-props-name"))
            self._timesp = statistics.TimePropsTable(config.param("time-props-name"))

        if self._context_stats_flag:
            smoothing = config.param_float("familiarity-smoothing")
            if smoothing < 0:
                print "Config Error [familiarity-smoothing] Must be positive : "+str(smoothing)
                sys.exit(2)

            fam_measure = config.param("familiarity-measure")
            if fam_measure not in CONST.ALL_FAM_MEASURES:
                print "Config Error [familiarity-measure] Invalid : "+str(fam_measure)
                sys.exit(2)

            f_name = config.param("context-props-name")
            self._contextsp = statistics.ContextPropsTable(f_name, smoothing,
                                                           fam_measure)
            self._contextsp._aoe_normalization = config.param_int("age-of-exposure-norm")

        # Other
        self._tasktype = config.param("tasktype")
        self._minfreq = config.param_int("minfreq")
        self._record_itrs = config.param_int("record-iterations")
        self._maxtime = config.param_int("maxtime")
        self._maxlearned = config.param_int("maxlearned")
        self._remove_singletons = config.param_bool("remove-singleton-utterances")

        # End configuration based on config

        self._gold_lexicon = input.read_gold_lexicon(lexicon_path, self._beta)
        self._all_features = input.all_features(lexicon_path)
        #print "number of Gold Features", len(self._all_features)

        #self._learned_lexicon = wmmapping.Lexicon(self._beta, self._gold_lexicon.words())
        self._learned_lexicon = wmmapping.Lexicon(self._beta, ['fep'],
                k_sub=k_sub, k_basic=k_basic, k_sup=k_sup,
                alpha_sub=self.alpha_sub, alpha_basic=self.alpha_basic,
                alpha_sup=self.alpha_sup)
        self._aligns = wmmapping.Alignments(self._alpha)

        self._time = 0
        self._vocab = set()
        self._features = set()
        self._feature_frequencies = {}
        self._acquisition_scores = {}
        self._last_time = {}

        self._stopwords = stopwords

        self._context = []

        # Growing a semantic network
        self._grow_graph_flag = config.param_bool("semantic-network")
        '''
        self._hub_type =  config.param("hub-type")
        self._hubs_num = config.param_int("hub-num")
        self._sim_threshold = config.param_int("sim-threshold")
        if self._grow_graph_flag:
            self._words_graph = wgraph.WordsGraph(self._hubs_num, self._sim_threshold,\
            self._hub_type) #TODO AIDA
        '''

        # Dictionary to avoid recalculating wordnet categories for words that
        # have already been seen
        #self._wordnet_categories = {}

    def alpha(self, feature):
        if self._alpha >= 0:
            return self._alpha
        else:
            alpha =  self._feature_frequencies[feature]
            #if feature.endswith('f0'):
                #print feature, alpha
            return alpha

    def init_words_graph(self, hubs_num, sim_threshold, hub_type, coupling, lambda0, a0, miu0, sigma0, sampling_method):
        self._words_graph = wgraph.WordsGraph(hubs_num, sim_threshold, hub_type, coupling, lambda0, a0, miu0, sigma0, sampling_method)


    def reset(self):
        """
        Reset the internal data structures so that the model can learn on a new
        corpus with the same mo/del settings.

        """
        if self._stats_flag:
            self._wordsp.reset()
            self._timesp.reset()

        if self._context_stats_flag:
            self._contextsp.reset()

        self._learned_lexicon = wmmapping.Lexicon(self._beta, self._gold_lexicon.words())
        self._aligns = wmmapping.Alignments(self._alpha)

        self._time = 0
        self._vocab = set()
        self._features = set()
        self._acquisition_scores = {}
        self._last_time = {}

        #self._wordnet_categories = {}

    #BM getLambdaFunction
    def get_lambda(self):
        """ Return a lambda smoothing factor. """
        if self._lambda < 1 and self._lambda > 0:
            return self._lambda

        return 1.0 / (1 + self._time**self._power)


    def learned_lexicon(self):
        """ Return a copy of the learned Lexicon. """
        return copy.deepcopy(self._learned_lexicon)

    #BM calculateAvgComp
    def avg_acquisition(self, words, key):
        """
        Return the average acquisition score of words in words that match the
        part-of-speech tag key.

        """
        total = 0.0
        vsize = 0
        for word in words:
            if postag_key_match(word, key):
                total += self.acquisition_score(word)
                vsize += 1

        if vsize == 0:
            return 0.0
        return total / float(vsize)

    #BM getComprehensionScore
    def acquisition_score(self, word):
        """
        Return the acquisition score of word. If "forgetting" is activated then
        this value is recalculated before returning.

        """
        if self._forget_flag or word not in self._acquisition_scores:
            # If it is forgetting we need to update p(f|w) by recalculating
            self.calculate_acquisition_score(word)

        return self._acquisition_scores[word]

    #BM calculateComprehensionScore
    def calculate_acquisition_score(self, word):
        """
        Calculate and return the acquisition score of word. If "forgetting" is
        activated then the meaning probabilities need to be recalculated.

        """
        if self._forget_flag:
            self.update_meaning_prob(word)

        true_m = self._gold_lexicon.meaning(word)
        lrnd_m = self._learned_lexicon.meaning(word)

        sim = evaluate.calculate_similarity(self._beta, lrnd_m, true_m, self._simtype)
        self._acquisition_scores[word] = sim
        return sim

    ##BM updateWordFProb
    #def update_meaning_prob(self, word):
    #    """
    #    Update the meaning probabilities of word in this learner's lexicon.
    #    This is done by calculating the association between this word and all
    #    encountered features - p(f|w) - then normalizing to produce a
    #    distribution.

    #    """
    #    Lambda = self.get_lambda()

    #    # Hash computed associations to avoid double calculating
    #    associations = {}

    #    sum_assoc = 0.0
    #    for feature in self._features:
    #        assoc = self.association(word, feature)
    #        associations[feature] = assoc
    #        sum_assoc += assoc

    #    sum_assoc += (self._beta * Lambda) # Smoothing

    #    for feature in self._features:
    #        meaning_prob = (associations[feature] + Lambda) / sum_assoc
    #        self._learned_lexicon.set_prob(word, feature, meaning_prob)

    #    prob_unseen = Lambda / sum_assoc
    #    self._learned_lexicon.set_unseen(word, prob_unseen)

    ##BM updateWordFProb
    #def update_meaning_prob(self, word):
    #    """
    #    Update the meaning probabilities of word in this learner's lexicon.
    #    This is done by calculating the association between this word and all
    #    encountered features - p(f|w) - then normalizing to produce a
    #    distribution.

    #    """
    #    Lambda = self.get_lambda()

    #    freq = self._wordsp.frequency(word) + 1
    #    denom = freq + self._beta * Lambda

    #    for feature in self._features:
    #        meaning_prob = (self.association(word, feature) + Lambda) / denom
    #        self._learned_lexicon.set_prob(word, feature, meaning_prob)

    #    prob_unseen = Lambda / denom
    #    self._learned_lexicon.set_unseen(word, (prob_unseen, prob_unseen, prob_unseen))

    #BM updateWordFProb
    def update_meaning_prob(self, word):
        """
        Update the meaning probabilities of word in this learner's lexicon.
        This is done by calculating the association between this word and all
        encountered features - p(f|w) - then normalizing to produce a
        distribution.

        """
        #TODO: change to calculating alignments
        sub_denom = 0.0
        basic_denom = 0.0
        sup_denom = 0.0

        for feature in self._features:

            if feature.startswith('sub'):
                sub_denom += self.association(word, feature)
            elif feature.startswith('bas'):
                basic_denom += self.association(word, feature)
            elif feature.startswith('sup'):
                sup_denom += self.association(word, feature)
            else:
                raise NotImplementedError

        sub_denom += self.k_sub * self.alpha_sub
        basic_denom += self.k_basic * self.alpha_basic
        sup_denom += self.k_sup * self.alpha_sup

        for feature in self._features:
            if feature.startswith('sub'):
                denom = sub_denom
                alpha = self.alpha_sub
            elif feature.startswith('bas'):
                denom = basic_denom
                alpha = self.alpha_basic
            elif feature.startswith('sup'):
                denom = sup_denom
                alpha = self.alpha_sup
            else:
                raise NotImplementedError

            meaning_prob = (self.association(word, feature) + alpha) / denom

            self._learned_lexicon.set_prob(word, feature, meaning_prob)

        self._learned_lexicon.set_unseen(word, (self.alpha_sub/sub_denom,
            self.alpha_basic/basic_denom, self.alpha_sup/sup_denom))

    def association(self, word, feature):
        """
        Return the association score between word and feature.
        If SUM is the association type then the total alignment probabilities
        over time, of word being aligned with feature, is the association.
        If ACT is the association type then an activation function using this
        learner's forget_decay value is used to calculate the association.

        """
        if self._assoc_type == CONST.SUM or self._assoc_type == CONST.DEC_SUM:
            return self._aligns.sum_alignments(word, feature)

        if self._assoc_type == CONST.ACT:
            # aligns is a dictionary of time--alignment_score pairs
            aligns = self._aligns.alignments(word, feature)
            t = self._time

            assoc_sum = 0.0
            for t_pr,val in aligns.items():
                align_decay = self._forget_decay / val
                assoc_sum += (val / math.pow(t-t_pr+1, align_decay))

            return math.log1p(assoc_sum)


    #BM noveltyWord
    def novelty(self, word):
        """
        Return the novelty decay coefficient of word based on the last time it
        has been encountered and the learner's novelty_decay base value.

        """
        last_time = 0
        if word in self._last_time:
            last_time = self._last_time[word]

        if last_time == 0:
            # No decay for novel words
            return 1
        else:
            delta_time = float(self._time - last_time) + 1
            denom = pow(delta_time, self._novelty_decay)
            return (1 - (1.0 / denom))


    #BM updateMappingTables
    def calculate_alignments(self, words, features, outdir, category_learner=None):
        """
        Update the alignments for each combination of word-feature pairs from
        the list words and set features.

        """

        # for each word, update p(f|w) distribution
        if self._forget_flag:
            for word in words:
                self.update_meaning_prob(word)


        #category_flag = self._category_flag
        category_flag = False
        #category_probs = {}
        #prob = 1.0 / float(self._beta)
        #if category_flag and category_learner is None:
        #    # In this case there is no category learner to use
        #    for word in words:
        #        if word not in category_probs:
        #            category_probs[word] = {}
        #        for feature in features:
        #            category_probs[word][feature] = prob
        #elif category_flag and category_learner is not None:
        #    # Can build categories
        #    category_probs = self.calculate_category_probs(words, features,
                                                          #category_learner)

        # Begin calculating the new alignment of a word given a feature, as:
        # alignment(w|f) = (p(f|w) + ep) / (sum(w' in words)p(f|w') + alpha*ep)
        for feature in features:
            # Normalization term, sum(w' in words) p(f|w')
            denom = 0.0
            category_denom = 0.0

            # Calculate the normalization terms
            for word in words:
                denom += self._learned_lexicon.prob(word,feature)
                if category_flag:
                    category_denom += category_probs[word][feature]

            denom +=  (self.alpha(feature) * self._epsilon)
            category_denom +=  (self.alpha(feature) * self._epsilon)

            # Calculate alignment of each word
            for word in words:
                # alignment(w|f) = (p(f|w) + ep) / normalization
                alignment = (self._learned_lexicon.prob(word,feature) + self._epsilon) / denom

                # The weight used in alignment calculation
                #weight = 0.5  #used in cogsci 2012 paper
                weight = self._wordsp.frequency(word) / (1.0 + self._wordsp.frequency(word))

                #print "word alignment", alignment

                if category_flag:
                    alignment = weight * alignment
                    category_prob = category_probs[word][feature]
                    factor = (category_prob + self._epsilon) / category_denom
                    alignment += (1 - weight) * factor
                    #print "category alignment", alignment

                if self._novelty_flag:
                    alignment *=  self.novelty(word)

                # Record the alignment at this time step and update association.
                if self._assoc_type == CONST.DEC_SUM:
                    self._aligns.add_decay_sum(word, feature, self._time,
                                               alignment, self._forget_decay)
                else:
                    self._aligns.add_alignment(word, feature, self._time, alignment)
            # End alignment calculation for each word
        # End alignment calculation for each feature

        #BM Added Jul 27 2012 for context statistics issue, FIND BETTER SOLUTION
        # for each word, update p(f|w) distribution
        for word in words:
            self.update_meaning_prob(word)

            # Add features to the list of seen features for each word
            self._learned_lexicon.add_seen_features(word, features)

            if self._novelty_flag or self._grow_graph_flag:
                # Update the last time this word has been seen
                self._last_time[word] = self._time

            if self._grow_graph_flag:
                # Add each word to the semantic network or words graph #TODO Aida
                if word.endswith(":N") and word not in self._stopwords:
                    word_acq_score = self.calculate_acquisition_score(word)
                    self._words_graph.add_word(self._context, word, word_acq_score, \
                    self._learned_lexicon, self._last_time, self._time, self._beta, self._simtype)

                    self._context.append(word)
                    if len(self._context) > 100:#self._words_graph._hubs_num: TODO
                        self._context = self._context[1:]

    def calculate_category_probs(self, words, features, category_learner):
        """
        Calculate all words in words then using the resulting category to
        determine the probability of each possible word-feature pair of the lists
        words and features. Return these probabilities as a 2D dictionary keyed
        by first words then features.

        """
        b = self._beta
        simtype = self._simtype

        # Determine word categories for getting the probabilities for each feature
        categories = {}
        for word in words:
            categories[word] = category_learner.word_category(word)

            if self._tasktype != CONST.NWL: continue #TODO do we need this?

            if categories[word] == -1:
                wn_category = wordnet_category(word)
                categories[word] = category_learner.categorize(word, simtype, b, None, wn_category) #features)


            '''
            if word.endswith(":N") and word not in self._stopwords:
                # For the novel word learning task, new words are categorized as
                # they are encountered
                if self._tasktype == CONST.NWL and categories[word] == -1:
                    # Word not properly categorized, do so by wordnet category
                    wn_category = wordnet_category(word)
                    categories[word] = category_learner.categorize(word, simtype,
                                                                   b, wn_category)
            '''


        # Calculate word-feature probabilities for the meaning of the category
        # that the word is in relative to the feature
        category_word_feature_probs = {}
        for feature in features:
            for word in words:
                prob = 1.0 / float(self._beta)
                if categories[word] != -1:
                    prob = category_learner.prob(categories[word], feature)
                if word not in category_word_feature_probs:
                    category_word_feature_probs[word] = {}
                category_word_feature_probs[word][feature] = prob

        return category_word_feature_probs


    #BM processPair
    def process_pair(self, words, features, outdir, category_learner=None):
        """
        Process the pair words-features, two lists of words and features,
        respectively, to be learned from.

        """
        # Time calculated w.r.t words-features pairings being processed
        self._time += 1

        # Add current features to the set of all seen features
        for feature in features:
            self._features.add(feature)
            try:
                self._feature_frequencies[feature] += 1
            except KeyError:
                self._feature_frequencies[feature] = 0
                self._feature_frequencies[feature] += 1

        if self._dummy:
            words.append("dummy")

        # Calculate the alignment probabilities and learned lexicon probabilities
        self.calculate_alignments(words, features, outdir, category_learner)

        if self._dummy:
            words.remove("dummy")

        if self._stats_flag:
            t = self._time

            for word in words:
                # Update word statistics
                if not self._wordsp.has_word(word):
                    learned_c = self.learned_count(self._postags)
                    self._wordsp.add_word(word, t, learned_c)
                else:
                    self._wordsp.inc_frequency(word)
                self._wordsp.update_last_time(word, t)

                # Get acquisition score to determine if word is now learned
                #acq = self.calculate_acquisition_score(word)
                #if word not in self._vocab and (acq >= self._theta):
                #    lrnd_count = self.learned_count(self._postags)
                #    frequency = self._wordsp.frequency(word)

                #    self._wordsp.update_lrnd_props(word, t, frequency, lrnd_count)

                #    if frequency > self._minfreq:
                #        self._vocab.add(word)

    #BM processCorpus
    def process_corpus(self, corpus_path, outdir, category_learner=None, corpus=None):
        """
        Process the corpus file located at corpus_path, saving any gathered
        statistics to the directory outdir. The file at corpus_path should
        contains sentences and their meanings. If a Corpus corpus is presented,
        the corpus_path is ignored and the corpus provided from is read instead.
        If a CategoryLearner category_learner is presented, categories are
        learned as dictated by the tasktype set on this Learner.
        Return the number of words learned and the number of time steps it
        required.

        """
        close_corpus = False
        if corpus is None:
            if not os.path.exists(corpus_path):
                print "Error -- Corpus does not exist : " + corpus_path
                sys.exit(2)
            corpus = input.Corpus(corpus_path)
            close_corpus = True;

        (words, features) = corpus.next_pair()

        learned = 0 # Number of words learned
        while words != []:

            #BM CHANGED
            if self._maxtime > 0 and self._time >= self._maxtime:
                break

            if self._remove_singletons and len(words) == 1:
                # Skip singleton utterances
                (words, features) = corpus.next_pair()
                continue

            # TODO add a parameter
            # Cluster words every X steps
            if self._time > 100  and self._time % 50 == 0\
                and self._category_flag and self._tasktype is not None:
                print "making categories"
                seen_words = self._wordsp._words
                lexicon = self._learned_lexicon
                stopwords = self._stopwords
                all_features = self._all_features # All features in gold lexicon


                seen_features = self._features # Seen Features

                clusters, labels, cwords = semantic_clustering_categories(self._beta,  seen_words, lexicon, all_features, stopwords, CONST.ALL, 40)
                category_learner = CategoryLearner(self._beta, clusters, lexicon, seen_features)


            # ==================================================
            # Keep the novel nouns
            if self._stats_flag:
                noun_count = 0
                novel_nouns = []
                for word in words:
                    if word.endswith(":N") and (not word in self._stopwords):
                        noun_count += 1
                        if not word in self._wordsp.all_words(0):
                            novel_nouns.append(word)
            # ====================================================

            self.process_pair(words, features, outdir, category_learner)

            learned = len(list(self._vocab))

            if self._maxlearned > 0 and learned > self._maxlearned:
                break

            if self._time % 1000 == 0:
                print self._time

            # Record statistics - average acquisition and similarity scores
            if self._stats_flag:
                self.record_statistics(corpus_path, words, novel_nouns,
                                       noun_count, outdir)
            # Record Context statistics
            if self._context_stats_flag:
                sims = self.calculate_similarity_scores(words)
                comps = self.calculate_comprehension_scores(words)
                self._contextsp.add_context(set(words), self._time, sims, comps)

            (words, features) = corpus.next_pair()

        # End processing words-sentences pairs from corpus

        if self._stats_flag:
            # Write statistics to files
            self._wordsp.write(corpus_path, outdir, str(self._time))
            self._timesp.write(corpus_path, outdir, str(self._time))
        if self._context_stats_flag:
            words = self._contextsp._words.keys()
            sims = self.calculate_similarity_scores(words)
            comps = self.calculate_comprehension_scores(words)
            for word in words:
                self._contextsp.add_similarity(word, sims[word])
                self._contextsp.add_comprehension(word, comps[word])
            # Write the statistics to files
            self._contextsp.write(corpus_path, outdir)


        if close_corpus:
            corpus.close()

        return learned, self._time


    def record_statistics(self, corpus, words, novel_nouns, noun_count, outdir):
        """
        Record statistics in this learner's timesp and wordsp regarding word
        types learned based on the learner's postag list. Also records statistics
        on novelty based on the noun words in novel_nouns.

        """
        # Write statistics information ever record_itrs iterations
        if self._record_itrs > 0 and self._time % self._record_itrs == 0:
            self._wordsp.write(corpus, outdir, str(self._time))
            self._timesp.write(corpus, outdir, str(self._time))

        # Dictionary to store statistics
        avg_acq = {}

        #BM begin novelty ==============================================
        avg_acq_nn = self.avg_acquisition(novel_nouns, CONST.N)

        if noun_count >= 2 and len(novel_nouns) >= 1:
            avg_acq[CONST.NOV_N_MIN1] = avg_acq_nn

        if noun_count >= 2 and len(novel_nouns) >= 2:
            avg_acq[CONST.NOV_N_MIN2] = avg_acq_nn
        #BM end novelty ================================================

        all_words = self._wordsp.all_words(self._minfreq)
        all_learned = list(self._vocab)

        # Average acquisition score for all pos tags (words) that are learned
        avg_acq[CONST.LRN] = self.avg_acquisition(all_learned, CONST.ALL)

        # Record statistics for words based on the configured pos tags
        if CONST.ALL in self._postags:
            avg_acq[CONST.ALL]  = self.avg_acquisition(all_words, CONST.ALL)
        if CONST.N in self._postags or CONST.ALL in self._postags:
            avg_acq[CONST.N] = self.avg_acquisition(all_words, CONST.N)
        if CONST.V in self._postags or CONST.ALL in self._postags:
            avg_acq[CONST.V] = self.avg_acquisition(all_words, CONST.V)
        if CONST.OTH in self._postags or CONST.ALL in self._postags:
            avg_acq[CONST.OTH] = self.avg_acquisition(all_words, CONST.OTH)

        heard = self.heard_count(self._minfreq, self._postags)
        learned = self.learned_count(self._postags)

        # Record all information at this time step
        self._timesp.add_time(self._time, heard, learned, avg_acq)

    def calculate_similarity_scores(self, words):
        """
        Calculate the similarity score for each word in the list words and return
        a dictionary mapping words to their similarity scores.
        This is recorded for context statistics purposes.

        """
        sims = {}
        beta = self._beta
        lexicon = self._learned_lexicon
        gold_lexicon = self._gold_lexicon
        for word in words:
            lrnd_m = lexicon.meaning(word)
            true_m = gold_lexicon.meaning(word)
            sim = evaluate.sim_cosine(beta, lrnd_m, true_m)
            sims[word] = sim
        return sims

    def calculate_comprehension_scores(self, words):
        """
        Calculate the comprehension score for each word in the list words and
        return a dictionary mapping words to their comprehension scores.
        This is recorded for context statistics purposes.

        """
        comps = {}
        lexicon = self._learned_lexicon
        gold_lexicon = self._gold_lexicon
        for word in words:
            lrnd_m = lexicon.meaning(word)
            true_features = gold_lexicon.meaning(word).seen_features()
            comp = 0.0
            for feature in true_features:
                comp += lrnd_m.prob(feature)
            comps[word] = comp
        return comps

    #BM getHeardCount
    def heard_count(self, minfreq, postags):
        """
        Return the number of different words that have been encountered and
        have occurred at least minfreq times and their parts of speech tags are
        in the list postags.

        """
        if CONST.ALL in postags:
            return self._wordsp.count(minfreq)

        words = self._wordsp.all_words(minfreq)
        return postag_count(words, postags)

    #BM getLearnedCount
    def learned_count(self, postags):
        """
        Return the number of words learned whose parts of speech tags are in the
        list postags.

        """
        if CONST.ALL in postags:
            return len(self._vocab)

        vocab_list = list(self._vocab)
        return postag_count(vocab_list, postags)



#====================================================================#
#            Helper Functions                                        #
#====================================================================#

def postag_key_match(word, key):
    """ Return whether the part-of-speech tag of word is consistent with key. """
    if key == CONST.ALL:
        return True

    tag = postag(word)
    # '' added to compare against CONST.OTH in case word does not have tag
    return (key==tag) or (key==CONST.OTH and not tag in [CONST.V, CONST.N, ''])


def postag(word):
    """ Return the part-of-speech tag of w, where w is of the form word:tag. """
    w = word + ":"
    word_pos = re.findall("([^:]+):", w)
    if len(word_pos) > 1:
        postag = word_pos[1]
        return postag
    else:
        return ''

def postag_count(words, postags):
    """ Return the number of words in list words whose postags are in postags. """
    count = 0
    for word in words:
        tag = postag(word)
        if tag in postags:
            count += 1
    return count
