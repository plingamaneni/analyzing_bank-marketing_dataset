title: "Pranavi Lingamaneni final data analysis project"

library(MASS)
library(glmnet)
library(randomForest)
library(gbm)
library(caret)

Bank <- na.omit(read.csv("R/win-library/4.1/bank.csv", header = TRUE, sep = ";",
                na.strings = "?"))
sample_size <- round(nrow(Bank)*0.05)
dsobject <- Bank[sample(nrow(Bank), sample_size, replace = FALSE),]
summary(dsobject)

dsobject$job <- factor(dsobject$job)
dsobject$marital <- factor(dsobject$marital)
dsobject$education <- factor(dsobject$education)
dsobject$default <- factor(dsobject$default)
dsobject$housing <- factor(dsobject$housing)
dsobject$loan <- factor(dsobject$loan) 
dsobject$contact <- factor(dsobject$contact)
dsobject$month <- factor(dsobject$month)
dsobject$poutcome <- factor(dsobject$poutcome)
dsobject$y <- factor(dsobject$y)

summary(dsobject)

## identify any predictors that have zero or near-zero variance
nzv <- nearZeroVar(dsobject, saveMetrics= TRUE)
nzv ## run and see if there are any non-varying variables

## none of the variables has zero or near-zero variance, which means that we can use
## all of the predictors in our modeling processing.

## next, let's split the data into two chunks - one used for training+testing; 
## the other to serve as a hold-out set that we use to assess model performance (scoring)
## we will use a simple splitting approach.
set.seed(1984)
trainIndices <- createDataPartition(dsobject$y, ## indicate which var. is outcome
                                    p = 0.8, # indicate proportion to use in training-testing
                                    list = FALSE, 
                                    times = 1)

training <- dsobject[trainIndices,]
holdout <- dsobject[-trainIndices,]

## centering and scaling as part of the pre-processing step
preProcValues <- preProcess(training, method = c("center", "scale"))

## Next, create the scaled+centered of the training+testing subset of the dataset
trainTransformed <- predict(preProcValues, training) 
## apply the same scaling and centering on the holdout set, too
holdoutTransformed <- predict(preProcValues, holdout)

## create settings for cross-validation to be used
## we will use repeated k-fold CV. For the sake of time
## we will use 5-fold CV with 3 repetitions
fitControl <- trainControl(
  method = "repeatedcv", ## perform repeated k-fold CV
  number = 5,
  repeats = 3,
  classProbs = TRUE)

## random forest model
grid <- expand.grid(mtry = 1:(ncol(trainTransformed)-1)) 

forestfit <- train(y ~ .,
                   data = trainTransformed, 
                   method = "rf",
                   trControl = fitControl,
                   verbose = FALSE,
                   tuneGrid = grid)

## check what information is available for the model fit
names(forestfit)

## some plots
trellis.par.set(caretTheme())
plot(forestfit)

## make predictions on the hold-out set
predvals <- predict(forestfit, holdoutTransformed)

## create the confusion matrix and view the results
confusionMatrix(predvals, holdoutTransformed$y, positive = "yes")

## Rank the variables in terms of their importance
varImp(forestfit)

## boosted-tree model
grid <- expand.grid(interaction.depth = seq(1:3),
                    shrinkage = seq(from = 0.01, to = 0.2, by = 0.01),
                    n.trees = seq(from = 100, to = 500, by = 100),
                    n.minobsinnode = seq(from = 5, to = 15, by = 5))

boostedfit <- train(y ~ .,
                    data = trainTransformed, 
                    method = "gbm",
                    trControl = fitControl,
                    verbose = FALSE,
                    tuneGrid = grid)

## check what information is available for the model fit
names(boostedfit)

## some plots
trellis.par.set(caretTheme())
plot(boostedfit)

## make predictions on the hold-out set
predvals <- predict(boostedfit, holdoutTransformed)

## compute the performance metrics
confusionMatrix(predvals, holdoutTransformed$y)

## Rank the variables in terms of their importance
varImp(boostedfit)

##bagged-tree model
fitControl <- trainControl(
  method = "repeatedcv", ## perform repeated k-fold CV
  number = 5,
  repeats = 3,
  classProbs = TRUE)

grid <- expand.grid(mtry = ncol(trainTransformed))

baggedfit <- train(y ~ .,
                   data = trainTransformed, 
                   method = "rf",
                   trControl = fitControl,
                   verbose = FALSE,
                   tuneGrid = grid)

## check what information is available for the model fit
names(baggedfit)

## some plots
trellis.par.set(caretTheme())
plot(baggedfit)

## make predictions on the hold-out set
predvals <- predict(baggedfit, holdoutTransformed)

## create the confusion matrix and view the results
confusionMatrix(predvals, holdoutTransformed$y, positive = "yes")

## Rank the variables in terms of their importance
varImp(baggedfit)

#CLUSTERING
dp <- dsobject[, c("age","balance","day","duration","campaign","pdays","previous")]

set.seed(1984)
km.obj <- kmeans(dp, 3, nstart=25)

set.seed(1984)
dp.reduced <- scale(dp[, -c(1,2)])
km.obj <- kmeans(scale(dp.reduced), 3, nstart=25)

sum(km.obj$withinss)/km.obj$tot.withinss
km.obj$tot.withinss/sum(km.obj$betweenss)

maxnumclusters <- 20
vct.dissimilarity.ratios <- numeric(maxnumclusters)

for(numclusters in 2:maxnumclusters){
  set.seed(1984)
  km.obj <- kmeans(dp.reduced, numclusters, nstart=25)
  vct.dissimilarity.ratios[numclusters] <- km.obj$tot.withinss/sum(km.obj$betweenss)
}
ratio.changes <- sapply(2:length(vct.dissimilarity.ratios),
                        function(ind){vct.dissimilarity.ratios[ind+1] / vct.dissimilarity.ratios[ind]})

which(ratio.changes == min(ratio.changes, na.rm=T))

##SVM models
#SVM Linear
grid <- expand.grid(C = c(0.01, 0.1, 10, 100, 1000))#,
#sigma = c(0.5, 1, 2, 3, 4))

svmlinearfit <- train(y ~ .,
                      data = trainTransformed,
                      method = "svmLinear",
                      trControl = fitControl,
                      verbose = FALSE,
                      tuneGrid = grid)

## check what information is available for the model fit
names(svmlinearfit)

## some plots
trellis.par.set(caretTheme())
plot(svmlinearfit)

## make predictions on the hold-out set
predvals <- predict(svmlinearfit, holdoutTransformed)

## compute the performance metrics
confusionMatrix(predvals, holdoutTransformed$y, positive = "yes")

## Rank the variables in terms of their importance
varImp(svmlinearfit)

#SVM Radial
grid <- expand.grid(C = c(0.01, 0.1, 10, 100, 1000),
                    sigma = c(0.5, 1, 2, 3, 4))

svmradialfit <- train(y ~ .,
                      data = trainTransformed,
                      method = "svmRadial",
                      trControl = fitControl,
                      verbose = FALSE,
                      tuneGrid = grid)

## check what information is available for the model fit
names(svmradialfit)

## some plots
trellis.par.set(caretTheme())
plot(svmradialfit)

## make predictions on the hold-out set
predvals <- predict(svmradialfit, holdoutTransformed)

## compute the performance metrics
confusionMatrix(predvals, holdoutTransformed$y, positive = "yes")

## Rank the variables in terms of their importance
varImp(svmradialfit)

##logistic regression
logisticfit <- train(y ~ .,
                     data = trainTransformed, 
                     method = "glm",
                     #family = "family", ## specifying this seems to cause errors/warnings
                     trControl = fitControl)

# some plots
trellis.par.set(caretTheme())
plot(logisticfit)

## make predictions on the hold-out set
predvals <- predict(logisticfit, holdoutTransformed)

## compute the performance metrics
confusionMatrix(predvals, holdoutTransformed$y, positive = "yes")

## Rank the variables in terms of their importance
varImp(logisticfit)
