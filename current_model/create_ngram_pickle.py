#!/usr/bin/env python

import cProfile
import logging
import nltk
import pickle
import sys

from argparse import ArgumentParser
from google_ngram_downloader import readline_google_store
from itertools import groupby
from string import rstrip


def script(data_path, out_path, **kwargs):

    freq_dict = {}

    with open(data_path) as f:
        wordnet_leaf_nodes = map(rstrip, f)

    wn = nltk.corpus.WordNetCorpusReader(nltk.data.find('corpora/wordnet'), None)

    grams = []

    for leaf in wordnet_leaf_nodes:
        assert leaf not in grams

        features = []
        s = wn.synset(leaf)

        while s.hypernyms():
            features.append(str(s.name().split('.')[0].replace('_', ' ')))
            s = s.hypernyms()[0]
        features.append(str(s.name().split('.')[0].replace('_', ' ')))

        grams.extend(features)

    grams = list(set(grams))
    grams.sort()

    for letter, group in groupby(grams, key=lambda x: x[0]):
        group = list(group)

        # cannot deal with n-grams for n >= 3
        if any([len(gram.split(' ')) > 2 for gram in group]):
            raise NotImplementedError

        # deal with 1-grams
        for one_gram in [gram for gram in group if len(gram.split(' ')) == 1]:
            freq_dict[one_gram] = 0

        _, _, records = next(readline_google_store(ngram_len=1,indices=letter))

        for record in records:
            gram = record.ngram.encode('ascii', 'ignore')
            if gram in freq_dict:
                freq_dict[gram] += record.match_count

        # deal with 2-grams
        if any([len(gram.split(' ')) == 2 for gram in group]):
            for two_gram in [gram for gram in group if len(gram.split(' ')) == 2]:
                indices = two_gram.split(' ')[0][0] + two_gram.split(' ')[1][0]
                _, _, records = next(readline_google_store(ngram_len=2,indices=indices))

                freq_dict[two_gram] = 0

                for record in records:
                    if record.ngram.encode('ascii', 'ignore') == two_gram:
                        freq_dict[two_gram] += record.match_count

    with open(out_path, 'wb') as f:
        pickle.dump(freq_dict, f)

def parse_args(args):
    parser = ArgumentParser()

    parser.add_argument('--data_path', '-d', metavar='data_path', required=True,
            help='The data file of words, with one word on each line')
    parser.add_argument('--out_path', '-o', metavar='out_path', required=True,
            help='The name of the file to which to write the pickle')
    parser.add_argument('--logging', default='WARNING',
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
            help='Logging level')
    return parser.parse_args(args)

def main(args = sys.argv[1:]):
    args = parse_args(args)
    logging.basicConfig(level=args.logging)
    script(**vars(args))


if __name__ == '__main__':
    cProfile.run('sys.exit(main())')
