XuTenenbaumExperimentReplicationForNIPS.m is the main script that iterates
through the 12 training conditions (1 dalmatian, 3 dalmatians, 3 dogs, 3
animals, 1 green pepper, 3 green peppers, etc...) and it calls
probGeneralization.m and computePosterior.m to calculate the generalization.

evaluateWnModels.m gets called after XuTenenbaumExperimentReplicationForNIPS to
evaluate how well it actually generalized.

hypothesis_space.mat has the prior (erlangPrior200), the hypothesis space
(hyps), and the associated labels for the rows (leaf_labels) and columns
(hyp_labels).

The hypothesis space, hyps, is a sparse binary matrix where (i,j)=1 means leaf
node i is a branch of internal node j (there is a directed path from j to i), or
they are the same node (as explained in the paper to distinguish leaf node
hypotheses).
