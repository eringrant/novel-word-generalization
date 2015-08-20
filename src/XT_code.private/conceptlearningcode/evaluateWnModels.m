%evaluateWnModels

load XuTenenbaumExperimentReplicationModelPredictions_bayes.mat
load replicationTestingStimuli.mat;


%
% ------------
% animals (AA)
% ------------

AAbayesSubA = leafProbsA(Xtest_subAnimal);
AAbayesBaseA = mean(leafProbsA(Xtest_basicAnimal));
AAbayesSuperA = mean(leafProbsA(Xtest_superAnimal));
AAbayesNmA = mean(leafProbsA([Xtest_vehicles,Xtest_vegetables]));

AAbayesSubB = leafProbsB(Xtest_subAnimal);
AAbayesBaseB = mean(leafProbsB(Xtest_basicAnimal));
AAbayesSuperB = mean(leafProbsB(Xtest_superAnimal));
AAbayesNmB = mean(leafProbsB([Xtest_vehicles,Xtest_vegetables]));

AAbayesSubC = leafProbsC(Xtest_subAnimal);
AAbayesBaseC = mean(leafProbsC(Xtest_basicAnimal));
AAbayesSuperC = mean(leafProbsC(Xtest_superAnimal));
AAbayesNmC = mean(leafProbsC([Xtest_vehicles,Xtest_vegetables]));

AAbayesSubD = leafProbsD(Xtest_subAnimal);
AAbayesBaseD = mean(leafProbsD(Xtest_basicAnimal));
AAbayesSuperD = mean(leafProbsD(Xtest_superAnimal));
AAbayesNmD = mean(leafProbsD([Xtest_vehicles,Xtest_vegetables]));

%make graph
AAbayesMatrix = [AAbayesSubA, AAbayesBaseA, AAbayesSuperA, AAbayesNmA; AAbayesSubB, AAbayesBaseB, AAbayesSuperB, AAbayesNmB; AAbayesSubC, AAbayesBaseC, AAbayesSuperC, AAbayesNmC; AAbayesSubD, AAbayesBaseD, AAbayesSuperD, AAbayesNmD ];

ih = figure('Units', 'pixels', ...
     'Position', [100 100 800 300]);
bar(AAbayesMatrix);
ylim([0 1]);
colormap(gray);
set(gca,'XTickLabel',{'1','3 subordinate','3 basic','3 superordinate'});
legend('subordinate','basic','superordinate','non-matches','Location','NorthEastOutside');
title('Bayesian Model on Animals');
ylabel({'Probability of','Generalization'});

set(gcf, 'PaperPositionMode', 'auto');
    
saveFile = 'bayesAnimals.eps';
saveas(ih,saveFile);

save bayesAnimalsPredictions.mat AAbayesMatrix;




%
% ---------------
% vehicles (BB)
% ---------------

BBbayesSubE = leafProbsE(Xtest_subVehicle);
BBbayesBaseE = mean(leafProbsE(Xtest_basicVehicle));
BBbayesSuperE = mean(leafProbsE(Xtest_superVehicle));
BBbayesNmE = mean(leafProbsE([Xtest_animals,Xtest_vegetables]));

BBbayesSubF = leafProbsF(Xtest_subVehicle);
BBbayesBaseF = mean(leafProbsF(Xtest_basicVehicle));
BBbayesSuperF = mean(leafProbsF(Xtest_superVehicle));
BBbayesNmF = mean(leafProbsF([Xtest_animals,Xtest_vegetables]));

BBbayesSubG = leafProbsG(Xtest_subVehicle);
BBbayesBaseG = mean(leafProbsG(Xtest_basicVehicle));
BBbayesSuperG = mean(leafProbsG(Xtest_superVehicle));
BBbayesNmG = mean(leafProbsG([Xtest_animals,Xtest_vegetables]));

BBbayesSubH = leafProbsH(Xtest_subVehicle);
BBbayesBaseH = mean(leafProbsH(Xtest_basicVehicle));
BBbayesSuperH = mean(leafProbsH(Xtest_superVehicle));
BBbayesNmH = mean(leafProbsH([Xtest_animals,Xtest_vegetables]));

%make graph
BBbayesMatrix = [BBbayesSubE, BBbayesBaseE, BBbayesSuperE, BBbayesNmE; BBbayesSubF, BBbayesBaseF, BBbayesSuperF, BBbayesNmF; BBbayesSubG, BBbayesBaseG, BBbayesSuperG, BBbayesNmG; BBbayesSubH, BBbayesBaseH, BBbayesSuperH, BBbayesNmH ];

ih = figure('Units', 'pixels', ...
     'Position', [100 100 800 300]);
bar(BBbayesMatrix);
ylim([0 1]);
colormap(gray);
set(gca,'XTickLabel',{'1','3 subordinate','3 basic','3 superordinate'});
legend('subordinate','basic','superordinate','non-matches','Location','NorthEastOutside');
title('Bayesian Model on Vehicles');
ylabel({'Probability of','Generalization'});

set(gcf, 'PaperPositionMode', 'auto');
    
saveFile = 'bayesVehicles.eps';
saveas(ih,saveFile);

save bayesVehiclesPredictions.mat BBbayesMatrix;



%
% ---------------
% vegetables (CC)
% ---------------

CCbayesSubI = leafProbsI(Xtest_subVegetable);
CCbayesBaseI = mean(leafProbsI(Xtest_basicVegetable));
CCbayesSuperI = mean(leafProbsI(Xtest_superVegetable));
CCbayesNmI = mean(leafProbsI([Xtest_animals,Xtest_vehicles]));

CCbayesSubJ = leafProbsJ(Xtest_subVegetable);
CCbayesBaseJ = mean(leafProbsJ(Xtest_basicVegetable));
CCbayesSuperJ = mean(leafProbsJ(Xtest_superVegetable));
CCbayesNmJ = mean(leafProbsJ([Xtest_animals,Xtest_vehicles]));

CCbayesSubK = leafProbsK(Xtest_subVegetable);
CCbayesBaseK = mean(leafProbsK(Xtest_basicVegetable));
CCbayesSuperK = mean(leafProbsK(Xtest_superVegetable));
CCbayesNmK = mean(leafProbsK([Xtest_animals,Xtest_vehicles]));

CCbayesSubL = leafProbsL(Xtest_subVegetable);
CCbayesBaseL = mean(leafProbsL(Xtest_basicVegetable));
CCbayesSuperL = mean(leafProbsL(Xtest_superVegetable));
CCbayesNmL = mean(leafProbsL([Xtest_animals,Xtest_vehicles]));

%make graph
CCbayesMatrix = [CCbayesSubI, CCbayesBaseI, CCbayesSuperI, CCbayesNmI; CCbayesSubJ, CCbayesBaseJ, CCbayesSuperJ, CCbayesNmJ; CCbayesSubK, CCbayesBaseK, CCbayesSuperK, CCbayesNmK; CCbayesSubL, CCbayesBaseL, CCbayesSuperL, CCbayesNmL ];

ih = figure('Units', 'pixels', ...
     'Position', [100 100 800 300]);
bar(CCbayesMatrix);
ylim([0 1]);
colormap(gray);
set(gca,'XTickLabel',{'1','3 subordinate','3 basic','3 superordinate'});
legend('subordinate','basic','superordinate','non-matches','Location','NorthEastOutside');
title('Bayesian Model on Vegetables');
ylabel({'Probability of','Generalization'});

set(gcf, 'PaperPositionMode', 'auto');
    
saveFile = 'bayesVegetables.eps';
saveas(ih,saveFile);

save bayesVegetablesPredictions.mat CCbayesMatrix;





%
% Aggregate scores for the models
% 
% bayes 

%(A) 1 subordinate
bayesAvgSubA = mean([AAbayesSubA BBbayesSubE CCbayesSubI]);
bayesAvgBaseA = mean([AAbayesBaseA BBbayesBaseE CCbayesBaseI]);
bayesAvgSuperA = mean([AAbayesSuperA BBbayesSuperE CCbayesSuperI]);
bayesAvgNmA = mean([AAbayesNmA BBbayesNmE CCbayesNmI]);


%(B) 3 subordinate
bayesAvgSubB = mean([AAbayesSubB BBbayesSubF CCbayesSubJ]);
bayesAvgBaseB = mean([AAbayesBaseB BBbayesBaseF CCbayesBaseJ]);
bayesAvgSuperB = mean([AAbayesSuperB BBbayesSuperF CCbayesSuperJ]);
bayesAvgNmB = mean([AAbayesNmB BBbayesNmF CCbayesNmJ]);

%(C) 3 basic
bayesAvgSubC = mean([AAbayesSubC BBbayesSubG CCbayesSubK]);
bayesAvgBaseC = mean([AAbayesBaseC BBbayesBaseG CCbayesBaseK]);
bayesAvgSuperC = mean([AAbayesSuperC BBbayesSuperG CCbayesSuperK]);
bayesAvgNmC = mean([AAbayesNmC BBbayesNmG CCbayesNmK]);

%(D) 3 superordinate
bayesAvgSubD = mean([AAbayesSubD BBbayesSubH CCbayesSubL]);
bayesAvgBaseD = mean([AAbayesBaseD BBbayesBaseH CCbayesBaseL]);
bayesAvgSuperD = mean([AAbayesSuperD BBbayesSuperH CCbayesSuperL]);
bayesAvgNmD = mean([AAbayesNmD BBbayesNmH CCbayesNmL]);

%make graph
AvgBayesMatrix = [bayesAvgSubA, bayesAvgBaseA, bayesAvgSuperA, bayesAvgNmA; 
    bayesAvgSubB, bayesAvgBaseB, bayesAvgSuperB, bayesAvgNmB; 
    bayesAvgSubC, bayesAvgBaseC, bayesAvgSuperC, bayesAvgNmC; 
    bayesAvgSubD, bayesAvgBaseD, bayesAvgSuperD, bayesAvgNmD ];

ih = figure('Units', 'pixels', ...
     'Position', [100 100 800 300]);
bar(AvgBayesMatrix);
ylim([0 1]);
colormap(gray);
set(gca,'XTickLabel',{'1','3 subordinate','3 basic','3 superordinate'});
legend('subordinate','basic','superordinate','non-matches','Location','NorthEastOutside');
title('Bayesian Model averaged across Stimulus Categories');
ylabel({'Probability of','Generalization'});

set(gcf, 'PaperPositionMode', 'auto');
    
saveFile = 'bayesAvgReplication.eps';
saveas(ih,saveFile);

save bayesReplicationAvg.mat AvgBayesMatrix;




