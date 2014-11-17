import os
import re
import input
from nltk.corpus.reader import CorpusReader
from nltk.util import binary_search_file as _binary_search_file
from nltk.probability import FreqDist
import nltk
import matplotlib.pyplot as plt

class Frequency:

    def __init__(self, corpus):
        self.word_freqs = dict()
        self.feature_freqs = dict()
        self.total_words = 0
        self.total_features = 0
        self._remove_singletons = True
        self.process_corpus(corpus)

    def process_corpus(self, corpus_path, corpus=None):

        close_corpus = False
        if corpus is None:
            if not os.path.exists(corpus_path):
                print "Error -- Corpus does not exist : " + corpus_path
                sys.exit(2)
            corpus = input.Corpus(corpus_path)
            close_corpus = True;

        (words, features) = corpus.next_pair()

        for wd in words:
            self.word_count(wd)

        for ft in features:
            self.feature_count(ft)

        while words != []:

            if self._remove_singletons and len(words) == 1:
                # Skip singleton utterances
                (words, features) = corpus.next_pair()
                continue

            for wd in words:
                self.word_count(wd)

            for ft in features:
                self.feature_count(ft)

            (words, features) = corpus.next_pair()

    def word_count(self, wd):
        if wd in self.word_freqs:
            self.word_freqs[wd] = self.word_freqs[wd] + 1
        else:
            self.word_freqs[wd] = 1.0
        self.total_words += 1

    def feature_count(self, wd):
        if wd in self.feature_freqs:
            self.feature_freqs[wd] = self.feature_freqs[wd] + 1
        else:
            self.feature_freqs[wd] = 1.0
        self.total_features += 1

    def get_word_freq(self,wd):
        if wd in self.word_freqs:
            return self.word_freqs[wd]
        else:
            return 0

    def get_feature_freq(self,ft):
        if wd in self.feature_freqs:
            return self.feature_freqs[ft]
        else:
            return 0

def get_synsets(synset_strings):
        return [S(synset) for synset in synset_strings]

if __name__ == "__main__":
    freq = Frequency('input_wn_fu_cs_scaled_categ.dev')

    wn = nltk.corpus.WordNetCorpusReader(nltk.data.find('corpora/wordnet'))

    hierarchy = {}

    for word in freq.word_freqs:
        if word.split(':')[1] == 'N':

            try:
                s = wn.synset(word.split(':')[0] + '.n.01')
            except nltk.corpus.reader.wordnet.WordNetError:
                pass

            d = s.min_depth()

            if not d in hierarchy:
                hierarchy[d] = {}
            hierarchy[d][word] = freq.word_freqs[word]

    order = list(hierarchy.keys())
    order.sort()

    labels = []
    frequencies = []

    for number in order:

        for word in hierarchy[number]:
            labels.append(word)
            frequencies.append(hierarchy[number][word])

    print(labels, frequencies)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(frequencies, 'o', markersize=1)

    xtickNames = plt.setp(ax, xticklabels=[""] + labels + [""])
    plt.setp(xtickNames, rotation=25, fontsize=4)

    plt.xlabel('Word')
    plt.ylabel('Frequency')
    plt.legend()
    ax.set_xlim([0, len(frequencies)])
    ax.set_ylim([0,5000])
    #filename = outdir + '/'+ word + '_' + str(time) + '.png'
    #print "Plot: ", filename
    #plt.savefig(filename)
    plt.show()
