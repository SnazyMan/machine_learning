function [pi,mu,cov] = EM_MoG(traindata, K, mu_init)
%EM_MoG EM algorithm for Mixture of Gaussians
%  traindata - input data
%  K -  Number of Gaussians
%  pi - vector of mixining coefficients, dimension K
%  mu - matrix of mean vectors, dimension kxd
%  cov - array of covariance matrices, dimension kxdxd

mu = mu_init;

% extract dimensionality
[m,n] = size(traindata);

% initialize mixing coefficients to a uniform distribution of dimension k
pi = ones([K,1]);
pi = pi./length(pi);

% initialize covariance matrix to identity dxd, there should be k cov
% matrices
for i = 1:K
    cov{i} = eye(n);
end

% EM algorithm 
% Stopping criterion - 1000 iterations or change in log liklihood less than
% 10^-6
liklihood = logliklihood(traindata,K,pi,mu,cov);
for q = 1:1000
    % E step - evaluate responsibilities using the current parameters
    gamma = Estep(traindata,K,mu,cov,pi);

    % M step - Re-estimate parameters using the current responsibilities
    [pi,mu,cov] = Mstep(traindata,K,gamma);

    % Evaluate log liklihood
    liklihood_new = logliklihood(traindata,K,pi,mu,cov);

    % check for convergence
    diff = liklihood_new - liklihood;
    if(abs(diff) < 10^-6)
        break;
    end    
    liklihood = liklihood_new;
end
end

function gamma = Estep(traindata,K,mu,cov,pi)
     % E step - evaluate responsibilities using the current parameters
     for i = 1:K
        temp(:,i) = pi(i)*gaussian_pdf(traindata(:,:),mu(i,:),cov{i});
     end
     sumTemp = sum(temp,2);
     for i = 1:K
        gamma(:,i) = pi(i)*gaussian_pdf(traindata(:,:),mu(i,:),cov{i})./sumTemp;
     end
end

function [pi,mu,cov] = Mstep(traindata,K,gamma)
    % M step - Re-estimate parameters using the current responsibilities
    [m,n] = size(traindata);
    mk = sum(gamma,1);
    mu = (1./mk)'.*(gamma'*traindata);
    for i = 1:K
        cov{i} =  (1/mk(i))*((gamma(:,i).*(traindata(:,:) - mu(i,:)))'*(traindata(:,:) - mu(i,:)));
    end 
    pi = mk./m;
end

function liklihood = logliklihood(traindata,K,pi,mu,cov)
    % Evaluate log liklihood
    for i = 1:K
        temp_newlike(:,i) = pi(i)*gaussian_pdf(traindata(:,:),mu(i,:),cov{i});
    end
    liklihood = sum(log(sum(temp_newlike,2)));    
end