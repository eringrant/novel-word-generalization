import nltk
import pickle
import pprint

wn = nltk.corpus.WordNetCorpusReader(nltk.data.find('corpora/wordnet'), None)

with open('wordnet_leaf_nodes.pkl', 'rb') as f:
    wordnet_leaf_nodes = pickle.load(f)

leaf_to_features_map = {}

for leaf in wordnet_leaf_nodes:
    assert leaf not in leaf_to_features_map

    features = []
    s = wn.synset(leaf)

    while s.hypernyms():
        features.append(str(s.name()))
        s = s.hypernyms()[0]
    features.append(str(s.name()))

    leaf_to_features_map[leaf] = list(reversed(features))

with open('feature_map.pkl', 'wb') as f:
    pickle.dump(leaf_to_features_map, f)
