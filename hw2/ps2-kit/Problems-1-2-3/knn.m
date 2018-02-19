function [nlabel] = knn(traindata, trainlabel, testdata, k)
% k nearest neighbor algorithm     
% traindata - data in which we have known labels
% trainlabel - labels for our data
% query - new example without a label
% k - number of nearest neighbors to consider
% nlabel - predicted output, label of nearest neighbor(s)

    % extract dimensionality 
    [m,n] = size(traindata);
    [m_t n_t] = size(testdata);
    
    % for each testdata point, find nearest in traindata
    for i=1:m_t
        % replicate the current testdata traindata times
        % find distance from current testdata point to all traindata
        distance = sqrt(sum((repmat(testdata(i,:), m, 1) - traindata).^2, 2));
        
        % sort distances in ascending order
        % retain positions to keep track of where we are in the traindata
        % value = distance(idx)
        [value idx] = sort(distance, 'ascend');
        
        % collect the k labels of the smallest distances
        nearest(i,:) = trainlabel(idx(1:k));
        
        % take a majority vote of the labels for the prediction
        nlabel(i) = mode(nearest(i,:));
    end
end

