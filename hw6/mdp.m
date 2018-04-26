%% Tyler Olivieri 
% CIS520 HW6

clc;clear;close all;

% construct mdp
s = 5; % number of states
a = 2; % number of actions
p = zeros([s s a]); % transition probabilty matrix
p(:,1,1) = .3; p(:,1,2) = .7; p(1,2,1) = .7; p(2,3,1) = .7; p(3,4,1) = .7;
p(4,5,1) = .7; p(5,5,1) = .7; p(1,2,2) = .3; p(2,3,2) = .3; p(3,4,2) = .3; 
p(4,5,2) = .3; p(5,5,2) = .3;
r = zeros([s s a]); % reward function matrix
r(:,1,1) = 3; r(:,1,2) = 3; r(5,5,1) = 10; r(5,5,2) = 10;

gamma = .9; % discount factor

% find optimal value function
[v,q,pi] = valueIteration(s,a,p,r,gamma)