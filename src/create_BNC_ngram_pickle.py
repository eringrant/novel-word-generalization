#!/usr/bin/env python

import logging
import nltk
import pickle
import re
import sys
import pprint

from argparse import ArgumentParser
from multiprocessing import cpu_count, Pool
from string import rstrip


def script(data_path, out_path, num_cores, corpus_path, **kwargs):

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

    with open(corpus_path) as f:
        lines = f.readlines()

    if num_cores == 1:
        results = []
        results.append(compile_freq_dict((lines, grams)))
    else:
        q = int(len(lines) / num_cores)
        pool = Pool(processes=num_cores)
        results = pool.map(compile_freq_dict, [(lines[i*q:(i+1)*q], grams) for i in range(num_cores)])

    freq_dict = {}
    for result in results:
        for key in result:
            try:
                freq_dict[key] += result[key]
            except KeyError:
                freq_dict[key] = 0
                freq_dict[key] += result[key]

    with open(out_path, 'wb') as f:
        pickle.dump(freq_dict, f)

def compile_freq_dict((lines, grams)):
    freq_dict = {}

    length_to_gram_dict = {}

    for gram in grams:
        try:
            length_to_gram_dict[len(gram.split(' '))].append(gram)
        except KeyError:
            length_to_gram_dict[len(gram.split(' '))] = []
            length_to_gram_dict[len(gram.split(' '))].append(gram)

	word_pat = re.compile("<w ([^>]+)>([^ <]+)[ <]")

    for line in lines:
        line = line.lower()
        line = re.findall(word_pat, line)
        line = [word for (pos_tag, word) in line]

        for i in range(len(line)):
            for length in length_to_gram_dict:
                for gram in length_to_gram_dict[length]:
                    if gram == ' '.join(line[i:i+length]):
                        try:
                            freq_dict[gram] += 1
                        except KeyError:
                            freq_dict[gram] = 0
                            freq_dict[gram] += 1

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
    parser.add_argument('--corpus_path', '-c', metavar='corpus_path', type=str,
                        required=True,
            help='The name of the corpus file')
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
