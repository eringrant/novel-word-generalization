% function: computePosterior.m 
% input: 
%    X: the observed numbers in a row vector
%    hyps: the matrix of hypotheses
%    prior: the prior distribution over hypotheses 
%
% output:
%    hypothesisProbs: a row vector containing the posterior probability distribution p(h|X) for all h
%           i.e., how well the number concept hypotheses explain the observed data X

function [ hypothesisProbs ] = computePosterior(X, hyps, prior)
    t0 = cputime;

hypothesisProbs = zeros(1,size(hyps,2));
likelihood = zeros(1,size(hyps,2));


for i=1:size(hyps,2),
    if all(ismember(X,find(hyps(:,i))))
        likelihood(i) = (1/sum(hyps(:,i)))^(length(X));
    end
end
    
hypothesisProbs = (likelihood .* prior)/ sum(likelihood .* prior); 

tN = cputime - t0;
disp(sprintf('computePosterior runtime: %d\n',tN));

end