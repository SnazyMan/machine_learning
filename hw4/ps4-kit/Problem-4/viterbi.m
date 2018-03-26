function [z, max_prob] = viterbi(x, init, trans, emit)
%viterbi algorithm implementation
%   input - x: observed state sequence x
%   input - init: initial state probabilities 
%   input - trans: transition probabilities
%   input - emit: emission probabilities
%   output - z: most likely hidden state sequence 

% extract dimensionality
T = length(x); % length of input sequence 
[K,KK] = size(trans); % number of hidden states

% initialize probability matrix and first time step
delta = zeros([K T]);
for i = 1:K
    delta(i,1) = exp( log(init(i)) + log(emit(i,x(1))) ) ;
end

% Compute maximal joint probability over partial hidden state sequence
% maintain back-tracking variables
% avoid underflow by adding exp(logs) instead of multiplying
for t = 1:(T-1)
    for k = 1:K
        % find maximum probability at time t+1 for each k
        delta(k,t+1) = compute_delta(delta, x, trans, emit, k, K, t);
        
        % find index which gives maximum probability
        psi(t+1,k) = compute_psi(delta, trans, t, k, K);
    end
end

% backtrack to compute most likely hidden state sequence
[value, z(T)] = max(delta(:,T));
for t = (T-1):-1:1
    z(t) = psi(t+1,z(t+1));
end

% find highest joint probability
max_prob = max(delta(:,T));

end

function delta_out = compute_delta(delta, x, trans, emit, k, K, t)
    for j = 1:K
        temp(j) = exp( log(delta(j,t)) + log(trans(j,k)) ); 
    end
    maxtemp = max(temp);
    
    delta_out = exp( log(maxtemp) + log(emit(k,x(t+1))) );     
end

function psi_out = compute_psi(delta, trans, t, k, K)
    for j = 1:K
        temp(j) = exp( log(delta(j,t)) + log(trans(j,k)) );
    end
    
    [dummy,psi_out] = max(temp);    
end
