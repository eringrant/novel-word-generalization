#!/bin/bash

#
# Aida's script for the novel word generalisation experiments
# ---

echo 'argis: [corpus_path] [lexicon_path] [stopwords_path] [config_path] [outdir]'

DATAPATH=$1

#CORPUS=$1
#CORPUS=$DATAPATH'/all_categfreq_prob_wordnet_features/input_wn_fu_cs_scaled_categ.tst' 
CORPUS=$DATAPATH'/all_categfreq_prob_wordnet_features/input_wn_fu_cs_scaled_categ.dev' 
#CORPUS=$DATAPATH'/prob_wordnet_features/input_wn_fu_cs_scaled_categ.tst'

#LEXICON=$2
LEXICON=$DATAPATH'/all_categfreq_prob_wordnet_features/all_catf_norm_prob_lexicon_cs.all' 
#LEXICON=$DATAPATH'/prob_wordnet_features/norm_prob_lexicon_cs.all'

# Stopwords can be entered as "" if none are to be used
#STOPWORDS=$3
#STOPWORDS='/u/aida/data/commonWords.txt'
STOPWORDS=$DATAPATH'/commonWords.txt'

if [ $STOPWORDS = "" ]; then
	STOPWORDS="."
fi

CONFIG=$2
OUT=$3


echo $CORPUS $LEXICON $CONFIG "$STOPWORDS" $OUT 
echo 'start'
time python xt_generalization.py $CORPUS $LEXICON $CONFIG "$STOPWORDS" $OUT  
echo 'done'
