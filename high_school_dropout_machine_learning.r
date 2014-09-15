###########################################################################
# Author:       Adam Weiss
# Description:  This is an experiment to see if machine learning can 
#               detect when a student will drop out of highschool using
#               the 2009 Educational Longitudinal Survey from the National
#               Center for Education Statistics (NCES).  Results are in the
#               aweiss_predicting_high_school_dropouts.pdf and .tex files
############################################################################

require(doParallel)
require(caret)
require(scatterplot3d)
require(xtable) # for generating latex tables
registerDoParallel(4) # run training using 4 threads.

#load full data set
load("hsls_09_student.rdata")
hsls09 <- hsls_09_student_v2_0

# convert data to numeric
for(i in 1:ncol(hsls09))
     if(class(hsls09[,i]) != "numeric" || class(hsls09[,i]) != "integer")
         hsls09[,i] <- as.numeric(hsls09[,i])

# Select only variables from the file cols.csv
cols <- read.csv("cols.csv", header = FALSE)
cols <- as.vector(t(cols))

subset <- hsls09[,cols] 

#select all variables that are not x2everdrop (whether a student has dropped classes)
var <- names(hsls09) != "x2everdrop"

# perform PCA analysis on the data
fit <- prcomp(hsls09[,var], scale=TRUE)

# comment out the following to save the results - this could take a while
#save(fit, file="pca_fit_subset.rdata")
# optionally load the saved subset data
#load("pca_fit_subset.rdata")

#convert classes to factors
classes <- factor(ifelse(hsls09[,!var] == 1, "Y", "N"))

#create normal classes
classes <- hsls09[,!var]

#split data into testing and training set
inTrain <- createDataPartition(y=hsls09[,!var], p=.75, list=FALSE)

# summary(fit) shows cumulative variance of < .9 before PC112 analysis.  This is the typical variance used for PCA
inputData <- fit$x[,1:112]

# perform neural network training with 1,3,5,and 7 hidden perceptrons and decay values of 0, 1e-04, 0.1
nntrain <- train(inputData[inTrain,], classes[inTrain], "nnet", trControl = trainControl(method="repeatedcv",repeats=1,number=10,classProbs = TRUE, summaryFunction = twoClassSummary),  preProc = c("center", "scale"), maxit=500, metric = "ROC", tuneGrid = expand.grid(.size = c(1,3,5,7), .decay=c(0, 1e-04,0.1)))

# optionally save the results for later use
# save(nntrain, file="nntrain.rdata")

# optionally load the results to speed things up
#load("nntrain.rdata")

# K-NN training, tuning with 4-12 neighbors
knntrain <- train(inputData[inTrain,], classes[inTrain], "knn", trControl = trainControl(method="repeatedcv",repeats=1,number=10,classProbs = TRUE, summaryFunction = twoClassSummary),  preProc = c("center", "scale"), metric = "ROC", tuneGrid = expand.grid(.k = c(4,5,6,7,8,9,10,11,12)))

# optionally save/load results for later use
#save(knntrain, file="~/Documents/Dropbox/Graduate School/JHU/machine_learning/project code/knntrain.rdata")
#load("nntrain.rdata")

# SVM Radial Basis Training
svmRadialTrain <- train(inputData[inTrain,], classes[inTrain], "svmRadialCost", trControl = trainControl(method="repeatedcv",repeats=1,number=10,classProbs = TRUE, summaryFunction = twoClassSummary),  preProc = c("center", "scale"), metric = "ROC")

# optionally save/load results for later use
#save(svmRadialTrain, file="~/Documents/Dropbox/Graduate School/JHU/machine_learning/project code/svmRadialTrain.rdata")
#load("~/Documents/Dropbox/Graduate School/JHU/machine_learning/project code/svmRadialTrain.rdata")

# SVM Polynomial Kernel Training
svmPolyTrain <- train(inputData[inTrain,], classes[inTrain], "svmPoly", trControl = trainControl(method="repeatedcv",repeats=1,number=10,classProbs = TRUE, summaryFunction = twoClassSummary),  preProc = c("center", "scale"), metric = "ROC")

# optionally save/load results for later use
#save(svmPolyTrain, file="~/Documents/Dropbox/Graduate School/JHU/machine_learning/project code/svmPolyTrain.rdata")
#load("~/Documents/Dropbox/Graduate School/JHU/machine_learning/project code/svmPolyTrain.rdata")

# compare the results of the resampling
resamps <- resamples(list(NN = nntrain, KNN = knntrain, SVMR = svmRadialTrain, SVMP = svmPolyTrain))

# get p-values and related info from all of the training
xtable(summary(diff(resamps, confLevel = 0.95, confInt = 0.95))$table$ROC)
