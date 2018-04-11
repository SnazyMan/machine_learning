function [w,b] = naiveBayes(trainingdata, labels)
%naiveBayes supervised naive Bayes 

% extract dimensonality
[m,n] = size(trainingdata);

nOnes = find(labels == 1);
nZeros = find(labels == -1);

% calculate priors
prior1 = length(nOnes)/m;
prior0 = length(nZeros)/m;

% calculate conditional probabilities
s = 0;
for j = 1:n
    % calculate theta1
    tmp = length( find(trainingdata(nOnes,j) == 1) );
    theta(j) = tmp/length(nOnes);
    
    % calculate theta0
    tmp = length( find(trainingdata(nZeros,j) == 1) );    
    theta0(j) = tmp/length(nZeros);    
    
    % calculate jth weight parameter
    w(j) = log( theta(j)/theta0(j) ) - log( (1-theta(j))/(1-theta0(j)) );
    
    % sum this term to be used in bias calculation
    s = s + log( (1-theta(j))/(1-theta0(j)) );
end

b = s - log(prior0/prior1);

end

