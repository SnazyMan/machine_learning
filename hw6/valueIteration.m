function [v,q,pi] = valueIteration(s,a,p,r,gamma)
%valueIteration - implementation of the value iteration algorithm
%   input - s: number of states
%   input - a: number of actions
%   input - p: transition probabilty matrix
%   input - r: reward function matrix
%   input - gamma: discount factor
%   output - v: optimal value function

% initalize V(s) = 0 for all states
v = zeros(1,s);
% initialize policy = 0 for all states
pi = zeros(1,s);

% repeat until max_s(V_t+1(s) - V_t(s)) < .001
v_old = zeros(1,s);
while (1)    
    % for every state, update V*(s)
    for i = 1:s
        for j = 1:a
            q(i,j) = update_state(i,j,r,gamma,p,v);   
        end
        [v(i),pi(i)] = max(q(i,:));
    end
    
    % check convergence condition
    converge = max(abs(v - v_old));
    if (converge < .001)
        break;
    end
    v_old = v;
end
end

function q = update_state(s,a,r,gamma,p,v)
    q = sum(p(s,:,a) .* ( r(s,:,a) + (gamma*v)));
end