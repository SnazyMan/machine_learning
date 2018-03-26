%% Tyler Olivieri 
% CIS520 Machine learning - Hidden Markov Models

clc;clear;close all;

% read in data
[init,trans,emit,vocab,pos,test_sents,test_sents2,test_sents3] = read_data();

% extract emission probability indexes for a given sequence of words
for i = 1:length(test_sents)
    emit_seq(i) = find(strcmp(test_sents{i},vocab));
end

% compute viterbi algorithm to find most likely hidden state sequence
[z1,max_prob1] = viterbi(emit_seq, init, trans, emit);

% extract tags from most likely hidden state sequence
for i = 1:length(z1)
    postags1{i} = pos{z1(i)};
end
%---------------------------------------------------------------------
% extract emission probability indexes for a given sequence of words
for i = 1:length(test_sents2)
    emit_seq2(i) = find(strcmp(test_sents2{i},vocab));
end

% compute viterbi algorithm to find most likely hidden state sequence
[z2,max_prob2] = viterbi(emit_seq2, init, trans, emit);

% extract tags from most likely hidden state sequence
for i = 1:length(z2)
    postags2{i} = pos{z2(i)};
end
%----------------------------------------------------------------------
% extract emission probability indexes for a given sequence of words
for i = 1:length(test_sents3)
    emit_seq3(i) = find(strcmp(test_sents3{i},vocab));
end

% compute viterbi algorithm to find most likely hidden state sequence
[z3,max_prob3] = viterbi(emit_seq3, init, trans, emit);

% extract tags from most likely hidden state sequence
for i = 1:length(z3)
    postags3{i} = pos{z3(i)};
end


function [init, trans, emit, vocab, pos, test_sents, test_sents2, ...
    test_sents3] =  read_data()

    %fd = fopen("./init.txt");
    %C = textscan(fd, '%f', 218);
    %fclose(fd);
    %init = cell2mat(C);
    load('./init.txt')
    
    %fd = fopen("./trans.txt");
    %testSamples = 218;
    %for i = 1:testSamples
    %    C(i) = textscan(fd, '%f', 218);
    %end
    %fclose(fd);
    %trans = cell2mat(C);
    load('./trans.txt')
    
    %fd = fopen("./emit.txt");
    %testSamples = 14394;
    %for i = 1:testSamples
    %    C(i) = textscan(fd, '%f', 218);
    %end
    %fclose(fd);
    %emit = cell2mat(C);
    load('./emit.txt');
    
    fd = fopen("./vocabulary.txt");
    vocab =  textscan(fd, '%s', 14394);
    fclose(fd);
    vocab = [vocab{:}];
    %load('./vocabulary.txt');

    fd = fopen("./pos.txt");
    pos =  textscan(fd, '%s', 218);
    fclose(fd);
    pos = [pos{:}];
    %load('./pos.txt');

    fd = fopen("./test_sents.txt");
    testSamples = 14;
    for i = 1:testSamples
        test_sents(i) = textscan(fd,"%s",1);
    end
    testSamples = 10;
    for i = 1:testSamples
        test_sents2(i) = textscan(fd,"%s",1);
    end
    testSamples = 8;
    for i = 1:testSamples
        test_sents3(i) = textscan(fd,"%s",1);
    end
end