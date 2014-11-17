#!/bin/bash

# Generating the dev data


# Fixing the lexicon to contain tf-idf score for features,
# also removing the duplicate features for each word

PATH="/u/aida/wl/late_talkers/data/"
#The lexicon that contains all the words, and their features from wordnet
INLEX=$PATH"/wordnet_features/lexicon_cs.all"
echo $INLEX

#/bin/mkdir tmp

PLEX="not_norm/prob_lexicon_cs.all"
DEVDATA=$PATH"/raw_input/joel_john_liz_nic_ruth_warr.out"
TSTDATA=$PATH"/raw_input/anne_aran_becky_carl_domin_gail.out"

/usr/bin/python calc_tfidf_lexicon.py $INLEX $PLEX

echo "done"

OUTPATH=$PATH"prob_wordnet_features"

# Generating the corpus, and normalizing the lexicon
#./generateCorpus.py -i $DEVDATA -c input_wn_fu_cs_scaled_categ.dev -l $PLEX -o lexicon_wn_prob_dev_scaled_categ.all -d $OUTPATH -m 10000 -t FULL

./generateCorpus.py -i $TSTDATA -c input_wn_fu_cs_scaled_categ.tst -l $PLEX -o norm_prob_lexicon_cs.all -d $OUTPATH -m 10000 -t FULL
