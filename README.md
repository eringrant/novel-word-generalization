novel_word_generalization
=============

This code provides a framework for modeling the problem of novel word generalization, integrated within a cross-situational word learning framework.

The core algorithm implements the model of Fazly et al. (2010), which is an incremental and probabilistic word learner.
The extension of the Fazly et al. (2010) model to novel word generalization is described in Nematzadeh et al. (2015).


The original model of Fazly et al. is implemented
[here](https://github.com/aidanematzadeh/word_learning).)


References:

* Fazly, A., Alishahi, A., & Stevenson, S. (2010).  [A probabilistic computational model of cross-situational word learning](http://onlinelibrary.wiley.com/doi/10.1111/j.1551-6709.2010.01104.x/abstract).  *Cognitive Science*, 34(6), 1017-1063.

* Nematzadeh, A., Grant, E., and Stevenson, S. (2015). [A computational cognitive model of novel word generalization](http://www.emnlp2015.org/proceedings/EMNLP/pdf/EMNLP207.pdf). In *Proceedings of the 2015 Conference on Empirical Methods for Natural Language Processing*.


Starter code is provided in `starter/conduct_generalization_experiments.py`.


Requirements: `Python 2`, `numpy`, `scipy`
