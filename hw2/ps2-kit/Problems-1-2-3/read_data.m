function [data, labels] =  read_data(path, numExamples)
% Read data from data sets using textscan - will become more generic and
% parameterized when different data sets need to be read

    fd = fopen(path);
    for v = 1:numExamples
        C(v) = textscan(fd, '%f', 3, 'Delimiter', ',');
    end
    fclose(fd);

    % convert from cell array to data array
    datalabels = cell2mat(C);
    datalabels = datalabels';

    % extract labels and remove labels from traindata matrix
    labels = datalabels(:, end);
    data = datalabels(:, 1:end - 1);

end