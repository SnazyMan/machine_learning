This folder contains three subfolders, two .mat files and three .m file to help you with the problem on Gaussian Mixture Models. 

Files:
X.mat - Full training data (480 x 2)
X_test.mat - Design matrix for the test data (120 x 2)
sample_plot.m - MATLAB code to help with plotting tasks
gaussian_pdf.m - MATLAB code to compute the P(x_i|z_i)
compute_nllh.m - MATLAB code to compute normalized log-likelihood 

Folders:
CrossValidation - contains the train and test data for each fold of the 5-fold cross-validation procedure on the training set.

TrainSubsets - contains different sized subsets of the training set in train.txt (10%, 20%, ..., 100%). Use X_0.1.mat for 10% data, X_0.2.mat for 20% data and so on.

MeanInitialization - Initial Gaussian means to be set prior to running the EM algorithm. 
For Part (a) of the problem, use the files in MeanInitialization/Part_a. (mu_0.1.mat for 10% data, mu_0.2.mat for 20% data and so on).   
For Part (b) of the problem, use the files in MeanInitialization/Part_b. (mu_k_1.mat for k=1 case, mu_k__2.mat for k=2 case and so on). 

	
	
