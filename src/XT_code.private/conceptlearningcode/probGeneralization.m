% bayesian generalization yo
function [ leafProbs, hypProbs ] = probGeneralization(X, hyps, prior)
    t0 = cputime;
    
    hypProbs = computePosterior(X, hyps, prior);
    
    numLeaves=size(hyps,1);
    leafProbs = zeros(numLeaves,1);
    possInds = 1:size(hyps,2);
    
    for i = 1:numLeaves
        if (mod(i,10000) == 0)
            disp(i);
        end
            
        leafProbs(i) = sum(hypProbs(setdiff(hyps(i,:).*possInds,0)));
        
    end
   
 %   leafProbs = sum(hyps.*repmat(hypProbs,numLeaves,1),2);
   

    
    tN = cputime - t0;
    disp(sprintf('probGeneralization runtime: %d\n',tN));
end