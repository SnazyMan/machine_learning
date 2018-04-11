function [w,b] = semiSupervisedNB(labeleddata,labels,unlabeleddata)
%semiSupervisedNB Semi-Supervised Naive Bayes

% extract dimensionality
[m,n] = size(labeleddata);
[um,un] = size(unlabeleddata);

% first compute regular naive bayes for parameter initializations
[prior0,prior1,theta0,theta1] = naive_Bayes(labeleddata,labels);

% compute initial log liklihood
%llh = logliklihood(theta0,theta1,prior0,prior1,q0,q1,data,undata,labels);
llh_old = 0;
iter = 0;
while(1)
    
    % E step
    q1 = Estep(unlabeleddata,prior0,prior1,theta0,theta1);
    q0 = 1 - q1;
    
    % M step
    [prior0,prior1,theta0,theta1] = Mstep(labeleddata,labels,unlabeleddata,q1,q0);
    
    % log liklihood
    llh = logliklihood(theta0,theta1,prior0,prior1,q0,q1,labeleddata,unlabeleddata,labels);
    
    % Stopping criterion   
    if (abs(((llh - llh_old)/llh)) < .01)
        break;
    end
    
    llh_old = llh;
    iter = iter + 1;
end

% compute w and b from parameters
s = 0;
for j = 1:n
    % calculate jth weight parameter
    w(j) = log( theta1(j)/theta0(j) ) - log( (1-theta1(j))/(1-theta0(j)) );
    
    % sum this term to be used in bias calculation
    s = s + log( (1-theta1(j))/(1-theta0(j)) );
end

b = s - log(prior0/prior1);

end

function q = Estep(ul_data,prior0,prior1,theta0,theta1)
% calculate the postier probability of all unlabeled data using bayes
% theorem

    [m,n] = size(ul_data);
    
    for i = 1:m
        % calculate liklihood p(x|y=1), p(x|y=-1)
        x_given_y = 1;
        x_given_y0 = 1;
        for j = 1:n
            if (ul_data(i,j) == 1)
                x_given_y = x_given_y * theta1(j);
                x_given_y0 = x_given_y0 * theta0(j);
            else
                x_given_y = x_given_y * (1 - theta1(j));
                x_given_y0 = x_given_y0 * (1 - theta0(j));
            end
        end
        % use bayes theorem to calculate postier p(y=1|x)
        q(i) = (x_given_y * prior1) / (x_given_y * prior1 + x_given_y0 * prior0);
    end
end

function [prior0,prior1,theta0,theta1] = Mstep(labeleddata,labels,unlabeleddata,q1,q0)
% maximazation step, update the parameters

    [m,n] = size(labeleddata);
    [um,un] = size(unlabeleddata);
    
    % compute updated priors
    y1 = find(labels == 1);
    y0 = find(labels == 0);
    prior0 = (1/(m+um))*(length(y1) + sum(q1));
    prior1 = 1 - prior0;
    
    % compute updated conditional probabilities
    for j = 1:n
        x1y1 = find(labeleddata(y1,j) == 1);
        x1 = find(unlabeleddata(:,j) == 1);        
        theta1(j) = (length(x1y1) + sum(q1(x1))) / (length(y1) + sum(q1));
        
        x1y0 = find(labeleddata(y0,j) == 1);
        theta0(j) = (length(x1y0) + sum(q0(x1))) / (length(y0) + sum(q0));                
    end
end

function llh = logliklihood(theta0,theta1,prior0,prior1,q0,q1,data,undata,labels)
% compute log liklihood
    
    [m,n] = size(data);
    [um,un] = size(undata);
    
    % labeled section 
    for i = 1:m
        % calculate p(x|y=1)
        x_given_y(i) = 1;
        for j = 1:n
            if (data(i,j) == 1)
                x_given_y(i) = x_given_y(i) * theta1(j);
            else
                x_given_y(i) = x_given_y(i) * (1 - theta1(j));
            end
        end
        % calculate log p(x,y) = log p(x|y=1)p(y=1)
        joint(i) = log(x_given_y(i) * prior1);
    end
    llh_label = sum(joint);
    
    % unlabeled section
    llh_unlabel = 0;
    for i = 1:um
        % calculate p(x|y=1)
        x_given_y_un(i) = 1;
        for j = 1:n
            if (undata(i,j) == 1)
                x_given_y_un(i) = x_given_y_un(i) * theta1(j);
            else
                x_given_y_un(i) = x_given_y_un(i) * (1 - theta1(j));
            end
        end
        
        % q * (p(x,y)/q)
        llh_unlabel = llh_unlabel + q1(i) * log((x_given_y_un(i) * prior1)/q1(i));
        
        % calculate p(x|y=-1)
        x_given_y0_un(i) = 1;
        for j = 1:n
            if (undata(i,j) == 1)
                x_given_y0_un(i) = x_given_y0_un(i) * theta0(j);
            else
                x_given_y0_un(i) = x_given_y0_un(i) * (1 - theta0(j));
            end
        end
    
        % q * (p(x,y)/q)
         llh_unlabel = llh_unlabel + q0(i) * log((x_given_y0_un(i) * prior0)/q0(i));
    end 
    
    % find total log liklihood
    llh = llh_label + llh_unlabel;
end

function [prior0,prior1,theta0,theta] = naive_Bayes(trainingdata,labels)

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
        
    end
end