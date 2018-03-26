function sample_plot(X_test, mu,sigma,subset)

%%%You will need to edit lines 22-24%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Plot learned contours 
fig = figure;

% Display a scatter plot of the three distributions.
hold off;

plot(X_test(:,1),X_test(:,2),'ko');
hold on;

set(gcf,'color','white') % White background for the figure.


gridSize = 100;
u1 = linspace(-6, 8, gridSize);
u2 = linspace(-10,8,gridSize);
[A B] = meshgrid(u1, u2);
gridX = [A(:), B(:)];

% Calculate the Gaussian response for every value in the grid.
z1 = gaussian_pdf(gridX, mu(1,:), sigma{1});
z2 = gaussian_pdf(gridX, mu(2,:), sigma{2});
z3 = gaussian_pdf(gridX, mu(3,:), sigma{3});
% Obviously you will have to replace 
% mu() and sigma{} in lines 22-24 with what you use to define the mean and
% covariance of each Gaussian. You will need to use this function
%through your implementation of the EM algorithm so make sure you get this
%right.

% Reshape the responses back into a 2D grid to be plotted with contour.
Z1 = reshape(z1, gridSize, gridSize);
Z2 = reshape(z2, gridSize, gridSize);
Z3 = reshape(z3, gridSize, gridSize);

% Plot the contour lines to show the pdf over the data.
[C, h] = contour(u1, u2, Z1,'blue');
[C, h] = contour(u1, u2, Z2,'blue');
[C, h] = contour(u1, u2, Z3,'blue');

axis([-6 8 -10 8])
str = sprintf('Test Data and Estimated Contours for %d', subset);
title(str);
str = sprintf("g%d", subset);
end