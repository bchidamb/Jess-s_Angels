

train = read.csv(file="data/mu_probe.csv", header=TRUE, sep=",")
test = read.csv(file="data/mu_val.csv", header=TRUE, sep=",")

head(train)

y = train[, 5]
ytest = test[, 5]

x = data.frame(train[, 1:4])
xtest = data.frame(test[, 1:4])

RMSE = function(true, pred) {
  res =  sqrt(sum((true - pred)**2) / length(true))
  return(res)
}

require(SuperLearner)
#require(gam)
#require(caret)
require(randomForest)
require(glmnet)
require(xgboost)
#require(kernelKnn)
#require(RCurl)
#require(MASS)
#require(tmle)
#require(ggplot2)
#require(gbm)

# Use 2 of those cores for parallel SuperLearner.
# Replace "2" with "num_cores" (without quotes) to use all cores.
options(mc.cores = 1)

# Check how many parallel workers we are using (on macOS/Linux).
getOption("mc.cores")

# We need to set a different type of seed that works across cores.
# Otherwise the other cores will go rogue and we won't get repeatable results.
# This version is for the "multicore" parallel system in R.
set.seed(1, "L'Ecuyer-CMRG")



#sl_lib = c("SL.xgboost", "SL.randomForest", "SL.nnet", "SL.ksvm",
#           "SL.bartMachine", "SL.KernelKnn", "SL.rpartPrune", "SL.lm", "SL.mean")

# xgboost > mean and glmnet 
# KernelKNN > xgboost
# xgboost > RandomForest
sl_lib = c("SL.mean", "SL.xgboost")


system.time({
  sl = SuperLearner(Y = y, X = x, SL.library = sl_lib)
})
sl


pred = predict(sl, xtest, onlySL = T)

realPred = as.vector(pred$pred)
ytest = as.vector(ytest)


RMSE(ytest, realPred)



# Fit XGBoost, RF, Lasso, Neural Net, SVM, BART, K-nearest neighbors, Decision Tree, 
# OLS, and simple mean; create automatic ensemble.
system.time({
  # This will take about 3x as long as the previous SuperLearner.
  cv_sl = CV.SuperLearner(Y = y, X = x, V = 3,
                          parallel="multicore",
                          SL.library = sl_lib)
})
summary(cv_sl)

# Review performance of each algorithm and ensemble weights.
cv_sl

