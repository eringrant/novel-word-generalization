import nltk

wn = nltk.corpus.WordNetCorpusReader(nltk.data.find('corpora/wordnet'), None)

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

