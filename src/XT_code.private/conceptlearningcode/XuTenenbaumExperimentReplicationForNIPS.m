% XuTenenbaumExperimentReplicationForNIPS.m

clear;
load hypothesis_space;
load replicationTrainingStimuli;


%---------------------
% RUN MODELS
%---------------------

% animals
% (A)
[ leafProbsA, hypProbsA ] = probGeneralization(Xtrain_singleSubAnimal, hyps, erlangPrior200);

% (B) 
[ leafProbsB, hypProbsB ] = probGeneralization(Xtrain_tripleSubAnimal, hyps, erlangPrior200);

% (C)
[ leafProbsC, hypProbsC ] = probGeneralization(Xtrain_tripleBasicAnimal, hyps, erlangPrior200);

% (D)
[ leafProbsD, hypProbsD ] = probGeneralization(Xtrain_tripleSuperAnimal, hyps, erlangPrior200);

% vehicles
% (E)
[ leafProbsE, hypProbsE ] = probGeneralization(Xtrain_singleSubVehicle, hyps, erlangPrior200);

% (F) 
[ leafProbsF, hypProbsF ] = probGeneralization(Xtrain_tripleSubVehicle, hyps, erlangPrior200);

% (G)
[ leafProbsG, hypProbsG ] = probGeneralization(Xtrain_tripleBasicVehicle, hyps, erlangPrior200);

% (H)
[ leafProbsH, hypProbsH ] = probGeneralization(Xtrain_tripleSuperVehicle, hyps, erlangPrior200);

% vegetables
% (I)
[ leafProbsI, hypProbsI ] = probGeneralization(Xtrain_singleSubVegetable, hyps, erlangPrior200);

% (J) 
[ leafProbsJ, hypProbsJ ] = probGeneralization(Xtrain_tripleSubVegetable, hyps, erlangPrior200);

% (K)
[ leafProbsK, hypProbsK ] = probGeneralization(Xtrain_tripleBasicVegetable, hyps, erlangPrior200);

% (L)
[ leafProbsL, hypProbsL ] = probGeneralization(Xtrain_tripleSuperVegetable, hyps, erlangPrior200);

saveFile = 'XuTenenbaumExperimentReplicationModelPredictions_bayes.mat';
save(saveFile,'leafProbsA','hypProbsA','leafProbsB','hypProbsB','leafProbsC','hypProbsC','leafProbsD','hypProbsD','leafProbsE','hypProbsE','leafProbsF','hypProbsF','leafProbsG','hypProbsG','leafProbsH','hypProbsH','leafProbsI','hypProbsI','leafProbsJ','hypProbsJ','leafProbsK','hypProbsK','leafProbsL','hypProbsL');

%---------------------
% EVALUATE TEST CASES
%---------------------
evaluateWnModels;
