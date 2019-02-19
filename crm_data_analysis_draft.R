

# load required libraries
library(caret)
library(corrplot)
library(plyr)



setwd("C:/Users/User/Dropbox/MSc Introduction to Statistical Learning/")




training <- read.table('train_X.csv', header = TRUE,
                       sep = '\t', na.strings=T)

testing <- read.table('test_X.csv', header = TRUE,
                      sep = '\t', na.strings=T)
results <- read.table('train_Y.csv', header = T,
                      sep = '\t')




results = results[-1,]                   #delete one row to match the remain dataset



# Set seed
set.seed(123456)

# Remove all the variables(columns names) having high missing values(50%)
train_x <- training[, colMeans(is.na(training)) <= .5]
dim(train_x )

# Remove Zero and Near Zero-Variance Predictors available in the caret package
nzv <- nearZeroVar(train_x)
train_x2<- train_x [, -nzv]
dim(train_x2)


train_x2$churn<-  as.factor(results$churn ) #Add a new variable, churn
train_x2$appetency <- as.factor(results$appetency)   #Add a new variable, appetency
train_x2$upselling <- as.factor(results$upselling)  #Add a new variable, upselling


tail(train_x2)



# Identifying numeric variables within the dataset
numericData <- train_x2[sapply(train_x2, is.numeric)]


# Identifying categorical variables within the dataset
categoricalData <- train_x2[sapply(train_x2, is.factor)]


# Compute the correlation matrix descending 
descrCor <- cor(numericData)

# Print correlation matrix and look at max correlation
print(descrCor)

summary(descrCor[upper.tri(descrCor)])



# find attributes that are highly corrected
highlyCorrelated <- findCorrelation(descrCor, cutoff=0.7)



# print indexes of highly correlated attributes
print(highlyCorrelated)



# Check Correlation Plot
corrplot(descrCor)


# Indentifying Variable Names of Highly Correlated Variables
highlyCorCol <- colnames(numericData)[highlyCorrelated]

# Print highly correlated attributes
highlyCorCol



sapply(train_x2, class)


write.table(train_x2, "cleanedtrainingData.csv",row.names=FALSE, sep=",") #save the cleaned dataset


dfEvaluate <- cbind(as.data.frame(sapply(train_x2, as.numeric)),
                    churn=train_x2$churn)


EvaluateAUC <- function(dfEvaluate) {
  require(xgboost)
  require(Metrics)
  CVs <- 10
  cvDivider <- floor(nrow(dfEvaluate) / (CVs+1))
  indexCount <- 1
  outcomeName <- c('churn')
  predictors <- names(dfEvaluate)[!names(dfEvaluate) %in% outcomeName]
  lsqErr  <- c()
  lsqAUC  <- c()
  for (cv in seq(1:CVs)) {
    print(paste('crossval',cv))
    dataTestIndex <- c((cv * cvDivider):(cv * cvDivider + cvDivider))
    dataTest <- dfEvaluate[dataTestIndex,]
    dataTrain <- dfEvaluate[-dataTestIndex,]
    
    bst <- xgboost(data = as.matrix(dataTrain[,predictors]),
                   label = dataTrain[,outcomeName],
                   max.depth=6, eta = 1, verbose=0,
                   nround=5, nthread=4, 
                   objective = "reg:linear")
    
    predictions <- predict(bst, as.matrix(dataTest[,predictors]), outputmargin=TRUE)
    err <- rmse(dataTest[,outcomeName], predictions)
    auc <- auc(dataTest[,outcomeName],predictions)
    
    lsqErr <- c(lsqErr, err)
    lsqAUC <- c(lsqAUC, auc)
    gc()
  }
  print(paste('AVG Error:',mean(lsqErr)))
  print(paste('AVG AUC:',mean(lsqAUC)))
}


EvaluateAUC(dfEvaluate)

#///////////////////////using appetency /////////////////////////////


dfEvaluate <- cbind(as.data.frame(sapply(train_x2, as.numeric)),
                    churn=train_x2$appetency)


EvaluateAUC <- function(dfEvaluate) {
  require(xgboost)
  require(Metrics)
  CVs <- 20
  cvDivider <- floor(nrow(dfEvaluate) / (CVs+1))
  indexCount <- 1
  outcomeName <- c('appetency')
  predictors <- names(dfEvaluate)[!names(dfEvaluate) %in% outcomeName]
  lsqErr  <- c()
  lsqAUC  <- c()
  for (cv in seq(1:CVs)) {
    print(paste('crossval',cv))
    dataTestIndex <- c((cv * cvDivider):(cv * cvDivider + cvDivider))
    dataTest <- dfEvaluate[dataTestIndex,]
    dataTrain <- dfEvaluate[-dataTestIndex,]
    
    bst <- xgboost(data = as.matrix(dataTrain[,predictors]),
                   label = dataTrain[,outcomeName],
                   max.depth=6, eta = 1, verbose=0,
                   nround=5, nthread=4, 
                   objective = "reg:linear")
    
    predictions <- predict(bst, as.matrix(dataTest[,predictors]), outputmargin=TRUE)
    err <- rmse(dataTest[,outcomeName], predictions)
    auc <- auc(dataTest[,outcomeName],predictions)
    
    lsqErr <- c(lsqErr, err)
    lsqAUC <- c(lsqAUC, auc)
    gc()
  }
  print(paste('AVG Error:',mean(lsqErr)))
  print(paste('AVG AUC:',mean(lsqAUC)))
}


EvaluateAUC(dfEvaluate)

#//////////////////// evaluation using upselling ////////////////////////////////////

dfEvaluate <- cbind(as.data.frame(sapply(train_x2, as.numeric)),
                    churn=train_x2$upselling)

EvaluateAUC <- function(dfEvaluate) {
  require(xgboost)
  require(Metrics)
  CVs <- 20
  cvDivider <- floor(nrow(dfEvaluate) / (CVs+1))
  indexCount <- 1
  outcomeName <- c('upselling')
  predictors <- names(dfEvaluate)[!names(dfEvaluate) %in% outcomeName]
  lsqErr  <- c()
  lsqAUC  <- c()
  for (cv in seq(1:CVs)) {
    print(paste('crossval',cv))
    dataTestIndex <- c((cv * cvDivider):(cv * cvDivider + cvDivider))
    dataTest <- dfEvaluate[dataTestIndex,]
    dataTrain <- dfEvaluate[-dataTestIndex,]
    
    bst <- xgboost(data = as.matrix(dataTrain[,predictors]),
                   label = dataTrain[,outcomeName],
                   max.depth=6, eta = 1, verbose=0,
                   nround=5, nthread=4, 
                   objective = "reg:linear")
    
    predictions <- predict(bst, as.matrix(dataTest[,predictors]), outputmargin=TRUE)
    err <- rmse(dataTest[,outcomeName], predictions)
    auc <- auc(dataTest[,outcomeName],predictions)
    
    lsqErr <- c(lsqErr, err)
    lsqAUC <- c(lsqAUC, auc)
    gc()
  }
  print(paste('AVG Error:',mean(lsqErr)))
  print(paste('AVG AUC:',mean(lsqAUC)))
}


EvaluateAUC(dfEvaluate)





library(randomForest)

#//////////////////////////////////////////////////////////////////////////////////////////////
#
#                                      Train Random Forest#                                   / 
#                                                                                             /
#/////////////////////////////////////////////////////////////////////////////////////////////

sqtmtry<- round(sqrt(ncol(dfEvaluate) - 1))
rfGrid <- expand.grid(mtry = c(round(sqtmtry / 2), sqtmtry, 2 * sqtmtry))

ctrl <- trainControl(method = "cv", classProbs = TRUE, summaryFunction = twoClassSummary, number = 3) 


#//////////////////////////////////////////////////////////////////////////////////////////////
#
#                                      Train Random Forest#                                   

# create new dataset to be used for Random Forest 
dfEvaluate1 <- dfEvaluate                       # new dataset 

factor(dfEvaluate1$churn,levels=c(-1, 1),labels=c('no', 'yes'))     # discretise churn variable
factor(dfEvaluate1$appetency,levels=c(-1, 1),labels=c('no', 'yes')) # discretise appetency variable
factor(dfEvaluate1$upselling,levels=c(-1, 1),labels=c('no', 'yes')) # discretise upselling variable




set.seed(15689)         #set seed
churn.rf <- randomForest(churn ~ ., data = dfEvaluate1,na.action=na.exclude) #train a random forest using churn
table(predict(churn.rf),dfEvaluate1$churn)                                   # cross tabulate the prediction

print(churn.rf)          #print the random forest

graphics.off() # graphics parameters
par("mar")      # graphics parameters
par(mar=c(1,1,1,1))  # graphics parameters


# Variable Importance for the churn attribute
varImpPlot(diabetes.rf,  
           sort = T,
           n.var=10,
           main="Top 10 - Churn Most important Variable")

#Random Forest
set.seed(15689)

optional.mod <- tuneRF(dfEvaluate1[-as.numeric(ncol(dfEvaluate1))],dfEvaluate1$churn,ntreeTry = 150, 
                       stepFactor = 2, improve = 0.05,trace = T, plot = T, doBest = F,na.action=na.exclude)

tun.mtry <- optional.mod[as.numeric(which.min(optional.mod[,"OOBError"])),"mtry"]

tun.rf <- randomForest(churn~.,data=dfEvaluate1, mtry= tun.mtry, ntree=101, 
                       keep.forest=TRUE, proximity=TRUE, importance=TRUE,test=test)

pred.test <- predict(tun.rf, newdata = test)                                        # predict using the testing dataset
confusionMatrix(test$diagnosis,pred.test)                                           # confusion matrix using the testing dataset



#//////////////////////////////////////////////////////////////////////////////////////////////
#
#                                       Appetency Random Forest


# create new dataset to be used for Random Forest 


factor(dfEvaluate1$churn,levels=c(-1, 1),labels=c('no', 'yes'))     # discretise churn variable
factor(dfEvaluate1$appetency,levels=c(-1, 1),labels=c('no', 'yes')) # discretise appetency variable
factor(dfEvaluate1$upselling,levels=c(-1, 1),labels=c('no', 'yes')) # discretise upselling variable

dfEvaluate2 <- dfEvaluate                       # new dataset 

set.seed(15689)         #set seed
appetency.rf <- randomForest(appetency ~ ., data = dfEvaluate2,na.action=na.exclude) #train a random forest using appetency
table(predict(appetency.r),dfEvaluate2$appetency)                                   # cross tabulate the prediction

print(churn.rf)          #print the random forest

graphics.off() # graphics parameters
par("mar")      # graphics parameters
par(mar=c(1,1,1,1))  # graphics parameters


# Variable Importance for the appetency attribute
varImpPlot(appetency.r,  
           sort = T,
           n.var=10,
           main="Top 10 - Appetency Most important Variable")

#//////////////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////////////
#
#                                       Appetency Random Forest



set.seed(15689)         #set seed
appetency.rf <- randomForest(churn ~ ., data = dfEvaluate1,na.action=na.exclude) #train a random forest using appetency
table(predict(appetency.rf),dfEvaluate1$appetency)                                   # cross tabulate the prediction

print(appetency.rf)          #print the random forest

graphics.off() # graphics parameters
par("mar")      # graphics parameters
par(mar=c(1,1,1,1))  # graphics parameters


# Variable Importance for the appetency attribute
varImpPlot(appetency.rf,  
           sort = T,
           n.var=10,
           main="Top 10 - Appetency Most important Variable")


#//////////////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////////////
#
#                                       Upselling Random Forest



set.seed(15689)         #set seed
upselling.rf <- randomForest(upselling ~ ., data = dfEvaluate1,na.action=na.exclude) #train a random forest using appetency
table(predict(upselling.rf),dfEvaluate1$upselling)                                   # cross tabulate the prediction

print(upselling.rff)          #print the random forest

graphics.off() # graphics parameters
par("mar")      # graphics parameters
par(mar=c(1,1,1,1))  # graphics parameters


# Variable Importance for the appetency attribute
varImpPlot(upselling.rf,  
           sort = T,
           n.var=10,
           main="Top 10 - upselling Most important Variable")

