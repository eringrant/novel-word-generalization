#!/usr/bin/env python

import logging
import nltk
import pickle
import sys
import time

from argparse import ArgumentParser
from google_ngram_downloader import readline_google_store
from itertools import groupby
from multiprocessing import cpu_count, Pool
from string import rstrip


def script(data_path, out_path, num_cores, **kwargs):

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

    groupings = [(letter, list(group)) for letter, group in groupby(grams, key=lambda x: x[0])]

    if num_cores == 1:
        results = []
        for grouping in groupings:
            results.append(get_gram_dict(grouping))
    else:
        pool = Pool(processes=num_cores)
        results = pool.map(get_gram_dict, groupings)

    freq_dict = {}
    for result in results:
        freq_dict.update(result)

    with open(out_path, 'wb') as f:
        pickle.dump(freq_dict, f)

def get_gram_dict(letter_group):
    freq_dict = {}

    letter, group = letter_group
    group = list(group)

    # deal with 1-grams
    for one_gram in [gram for gram in group if len(gram.split(' ')) == 1]:
        freq_dict[one_gram] = 0

    connected = False
    while not connected:
        try:
            _, _, records = next(readline_google_store(ngram_len=1,indices=letter))
            connected = True
        except AssertionError:
            logging.info("Access to ngram database denied for index" + letter + "; sleeping for 600 seconds...")
            time.sleep(600)

    logging.info('Accessing Google one-gram data for letter ' + letter)
    for record in records:
        candidate = record.ngram.encode('ascii', 'ignore').lower()
        if candidate in freq_dict:
            freq_dict[candidate] += record.match_count

    # deal with 2-grams
    logging.info('Accessing Google two-gram data for letter ' + letter)
    if any([len(gram.split(' ')) == 2 for gram in group]):
        for two_gram in [gram for gram in group if len(gram.split(' ')) == 2]:
            indices = two_gram.split(' ')[0][:2]

            connected = False
            while not connected:
                try:
                    _, _, records = next(readline_google_store(ngram_len=2,indices=[indices]))
                    connected = True
                except AssertionError:
                    logging.info("Access to ngram database denied for indices" + indices + "; sleeping for 600 seconds...")
                    time.sleep(600)

            freq_dict[two_gram] = 0

            for record in records:
                candidate = record.ngram.encode('ascii', 'ignore').lower()
                if candidate == two_gram:
                    freq_dict[two_gram] += record.match_count

    # deal with n-grams for n > 2
    if any([len(gram.split(' ')) == 3 for gram in group]):
        logging.info('Encountered at least one n-gram with n >= 3; frequency set to 0')
        for gram in [gram for gram in group if len(gram.split(' ')) > 2]:
            freq_dict[gram] = 0

    return freq_dict

def parse_args(args):
    parser = ArgumentParser()

    parser.add_argument('--logging', type=str, default='INFO',
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
            help='Logging level')
    parser.add_argument('--data_path', '-d', metavar='data_path', type=str,
                        required=True,
            help='The data file of words, with one word on each line')
    parser.add_argument('--out_path', '-o', metavar='out_path', type=str,
                        required=True,
            help='The name of the file to which to write the pickle')
    parser.add_argument('--num_cores', '-n', metavar='num_cores',
                        type=int, choices=xrange(1, cpu_count()),
                        default=cpu_count(),
            help="Number of processes used; default is %i" % cpu_count())

    return parser.parse_args(args)

def main(args = sys.argv[1:]):
    args = parse_args(args)
    logging.basicConfig(level=args.logging)
    script(**vars(args))


if __name__ == '__main__':
    sys.exit(main())
