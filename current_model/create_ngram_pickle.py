#!/usr/bin/env python

import logging
import pickle
import sys

from argparse import ArgumentParser
from google_ngram_downloader import readline_google_store
from itertools import groupby



def script(data_path, out_path, **kwargs):

    import pdb; pdb.set_trace()

    freq_dict = {}

    with open(data_path) as f:
        words = f.readlines()

    words.sort()

    for letter, group in groupby(words, key=lambda x: x[0]):
        group = list(group)

        # cannot deal with n-grams for n >= 3
        if any([len(gram.split(' ')) > 2 for gram in words]):
            raise NotImplementedError

        # deal with 1-grams
        for one_gram in [gram for gram in words if len(gram.split(' ')) == 1]:
            freq_dict[one_gram] = 0

        _, _, records = next(readline_google_store(ngram_len=1,indices=letter))
        for record in records:
            if str(record.ngram) in freq_dict:
                freq_dict[str(record.ngram)] += record.match_count

        # deal with 2-grams
        if any([len(gram.split(' ')) == 2 for gram in words]):
            for two_gram in [gram for gram in words if len(gram.split(' ')) == 2]:
                indices = two_gram.split(' ')[0][0] + two_gram.split(' ')[1][0]
                _, _, records = next(readline_google_store(ngram_len=2,indices=indices))

                freq_dict[two_gram] = 0

                for record in records:
                    if str(record.ngram) == two_gram:
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
    sys.exit(main())
