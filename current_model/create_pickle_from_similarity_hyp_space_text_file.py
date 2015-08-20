#!/usr/bin/python

import pickle

feature_map = {}

with open('similarity_hypothesis_space.txt', 'r') as f:
    for line in f:
        if not line == '\n':
            id = line.split(';')[0]
            features = [item.rstrip('\n') for item in (line.split(';')[1]).split(',')]
            feature_map[id] = features

with open('xt_hypothesis_space_feature_map.pkl', 'wb') as f:
    pickle.dump(feature_map, f)
