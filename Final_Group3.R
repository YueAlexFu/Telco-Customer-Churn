## FINAL PROJECT - GROUP 3
etwd("C:/Users/even4/OneDrive/Desktop/STAT642 Data Mining/group project")

library(caret)
library(corrplot)
library(e1071)
library(factoextra)
library(rpart)
library(rpart.plot)
library(randomForest)

data <- read.csv("telco_churn.csv")

## The data consists of 7043 observations of 21 variables.
summary(data)
str(data)
head(data)

## Dependent variable
## Churn is the dependent variable.
## The dataset is imbalanced with 1869 records of churned 
## customers and 5174 records of non-churned customers.

table(data$Churn)
barplot(table(data$Churn))



## Pre-processing

## Step 1: Remove variables that provide no information.

## The output of the str function shows us that we have 
## a 'customerID' column. This column is unique for each
## row and provides no information that can be used to 
## model churn.

data2 <- data[-1]

## Step 2: Change dummy variables for categorical variables.

## We have 15 categorical variables apart from the dependent
## variable. We use the dummyVars to create one dummy variable
## for each level of these variables. 

dummy <- dummyVars(~gender + Partner + Dependents + PhoneService
                 + MultipleLines + InternetService + OnlineSecurity
                 + OnlineBackup + DeviceProtection + TechSupport
                 + StreamingTV + StreamingMovies + Contract
                 + PaperlessBilling + PaymentMethod,
                 data = data2, sep="_", fullRank = TRUE,
                 drop2nd = TRUE)

data3 <- predict(dummy, data2)

data4 <- data.frame(data2[, names(data2) %in% 
                        c("SeniorCitizen", "tenure",
                        "MonthlyCharges", "TotalCharges", "Churn") ],
                    data3)

## Step 3: Missing data

## Identify records with missing values

data4[!complete.cases(data4), ]

## We have 11 rows where one variable has missing data.
## Since this is an insignficant portion of our data,
## we will delete the corresponding rows.

data5 <- data4[complete.cases(data4), ]


## Feature Selection

## Step 1: Correlation between dependent variables - Identifying redundant vars
## Identify variable pairs with high correlation (mag c > 0.70)

data5_corr <- cor(data5[-5])
## write.csv(round(data5_corr,2),"cor.csv")
## corrplot(data5_corr, method = "number")

high_corrs <- findCorrelation(data5_corr, cutoff=.70, 
                              names=TRUE)
high_corrs

data5_corr2 <- as.data.frame(as.table(data5_corr))
nrow(subset(data5_corr2, abs(Freq) > 0.7 & abs(Freq) < 1))

## There are nine variable pairs where the magnitude of correlation
## is greater than 0.7. We eliminate one variable from each of
## these pairs.


data6 <- data5[, !names(data5) %in% high_corrs]

data6_corr <- cor(data6[-3])
data6_corr <- as.data.frame(as.table(data6_corr))
nrow(subset(data6_corr, abs(Freq) > 0.7 & abs(Freq) < 1))
# 0 pairs above 0.70.


## Split data into training and testing

set.seed(2020)
samp <- createDataPartition(data6$Churn, p=.80, list=FALSE)
train = data6[samp, ] 
test = data6[-samp, ]

## Feature Selection 2 - Random Forest

set.seed(2020)
churn.rf <- randomForest(Churn~.,
                      data=train,
                      importance=TRUE, 
                      proximity=TRUE, 
                      ntree=500)
churn.rf
churn.rf$importance

varImp(churn.rf)

varImpPlot(churn.rf, lcolor = "blue", 
           main = "Variable Importance Plot",
           type = 2)




## Model 1: Support Vector Machiine

## We've set class weights as 2.7 for the positive class
## and 1 for the negative class to counteract the impact of 
## class imbalance on the model.


## a: Linear Kernel with 21 variables

set.seed(2020)

svm_mod_lin <- svm(Churn~.,
                   data = train,
                   method = "C-classification",
                   kernel = "linear",
                   scale = TRUE,
                   class.weights = c('Yes' = 2.7, 'No' = 1))

svm.train_lin <- predict(svm_mod_lin, 
                     train[, -3], type="class")

svm.train.acc_lin <- confusionMatrix(svm.train_lin, 
                                 train$Churn, 
                                 mode="prec_recall",
                                 positive = "Yes")
svm.train.acc_lin

## Precision = 0.46
## Recall = 0.83
## F1 = 0.59

svm.test_lin <- predict(svm_mod_lin, 
                    test[, -3], 
                    type="class")

svm.test.acc_lin <- confusionMatrix(svm.test_lin, 
                                test$Churn, 
                                mode="prec_recall",
                                positive = "Yes")
svm.test.acc_lin


## Precision = 0.47
## Recall = 0.84
## F1 = 0.60

## b: Radial Kernel with 21 variables

set.seed(2020)
svm_mod_rad <- svm(Churn~., 
                   data=train, 
                   method="C-classification", 
                   kernel="radial", 
                   scale=TRUE,
                   class.weights = c('Yes' = 2.7, 'No' = 1))


svm.train_rad <- predict(svm_mod_rad, 
                       train[, -3], type="class")

svm.train.acc_rad <- confusionMatrix(svm.train_rad, 
                                   train$Churn, 
                                   mode="prec_recall",
                                   positive = "Yes")
svm.train.acc_rad

## Precision = 0.58
## Recall = 0.86
## F1 = 0.69

svm.test_rad <- predict(svm_mod_rad, 
                      test[, -3], 
                      type="class")

svm.test.acc_rad <- confusionMatrix(svm.test_rad, 
                                  test$Churn, 
                                  mode="prec_recall",
                                  positive = "Yes")
svm.test.acc_rad


## Precision = 0.53
## Recall = 0.78
## F1 = 0.63


## There isn't too much difference between the performace of the linear
## and radial kernel. We picked the radial kernel because it is more
## flexible to variation in the data. 

## c: Radial Kernel with 4 variables


## Pre-processing: PCA

pp=prcomp((data6[, -3]), scale = TRUE)
fviz_eig(pp)

fviz_pca_var(pp,col.var = "contrib",gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),repel = TRUE)    


## First 2 PCs explain 33% of the variation in the overall data.
## However, all the variation is not needed to predict churn.

## We built a model using the 4 variables identified by RF as the most
## important predictors aftering transforming them into PCs.

train2 <- data.frame(train[, names(train) %in% 
                                  c("tenure",
                                    "InternetService_Fiber.optic", 
                                    "PaymentMethod_Electronic.check", 
                                    "Contract_Two.year", "Churn") ])

pp2=prcomp((train2[, -2]), scale = TRUE)

fviz_eig(pp2)
## The first two PCs explain over 75% of the variation 
## in the top 4 variables.

fviz_pca_var(pp2,col.var = "contrib",gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),repel = TRUE)    

## The prcomp object stores the transformed values of the training data.
## We will add the churn column back into the data.

## Traing data in terms of Principal Components
train_pc2 <- data.frame(pp2$x, train2[, 2])
colnames(train_pc2) <- c("PC1", "PC2", "PC3", "PC4", "Churn")

## We then transformed the testing data using the same Principal Components.

test2 <- data.frame(test[, names(test) %in% 
                                 c("tenure",
                                   "InternetService_Fiber.optic", 
                                   "PaymentMethod_Electronic.check", 
                                   "Contract_Two.year", "Churn") ])


test2_pred = predict(pp2, test2[, -2])

## Testing data in terms of Principal Components
test_pc2 = data.frame(test2_pred, test2[, 2])
colnames(test_pc2) <- c("PC1", "PC2", "PC3", "PC4","Churn")

## SVM model

set.seed(2020)
svm_mod_rad_pc <- svm(Churn ~., 
               data=train_pc2, 
               method="C-classification", 
               kernel="radial", 
               scale=FALSE,
               class.weights = c('Yes' = 2.7, 'No' = 1))



svm.train_rad_pc <- predict(svm_mod_rad_pc, train_pc2)
svm.train.acc_pc <-confusionMatrix(svm.train_rad_pc, train_pc2$Churn, 
                                mode="prec_recall", positive = "Yes")
svm.train.acc_pc

## Precision = 0.48
## Recall = 0.83
## F1 = 0.61

svm.test_rad_pc <- predict(svm_mod_rad_pc, test_pc2)
svm.test.acc_pc <- confusionMatrix(svm.test_rad_pc, test_pc2$Churn, 
                                mode="prec_recall", positive = "Yes")
svm.test.acc_pc

## Precision = 0.48
## Recall = 0.82
## F1 = 0.61


## SVM Plot

plot(svm_mod_rad_pc, train_pc2, PC1~PC2, 
     slice = list(PC3 = 1, PC4 = -1))


## final results comparison - testing for both models

round(cbind(all_var = svm.test.acc_rad$byClass, top4_var = svm.test.acc_pc$byClass), 2)



## Model 2: Decision Tree

##OPTION 1: No hyperparameter tuning

# DT classification to our training dataset.

set.seed(2020)
data.rpart <- rpart(formula = Churn ~ ., 
                    data = train, 
                    method = "class")

data.rpart

# Tree plot
rpart.plot(data.rpart, 
           extra = 2, 
           under = TRUE,  
           varlen=0, 
           faclen=0)
##OR
rpart.plot(data.rpart, 
           extra = "auto", 
           under = TRUE,  
           varlen=0, 
           faclen=0)

# Evaluating In-Sample Performance
# for our training and test data sets

#TRAINING
inpred_DT = predict(object=data.rpart,
                    newdata=train, type = "class")


DT.train.acc <-confusionMatrix(data=inpred_DT, 
                               reference=train$Churn, mode="prec_recall", positive = "Yes")

DT.train.acc

#TEST
outpred_DT = predict(object=data.rpart,
                     newdata=test, type = "class")


DT.test.acc <-confusionMatrix(data=outpred_DT, 
                              reference=test$Churn, mode="prec_recall", positive = "Yes")
DT.test.acc 

## Results from Training|Test
#Accuracy 0.79 | 0.79
#Kappa 0.37 | 0.37
#Recall 0.39 | 0.37
#F1 0.50 | 0.48

## After setting seed
## Precision = 0.68 | 0.71
## Recall = 0.38 | 0.39
## F1 = 0.49 | 0.51


##OPTION 2: Hyperparameter Tuning/ Model Pruning for DT Models

## 1. Grid Search
grids <- expand.grid(cp=seq(from=0,to=.4,by=.02))
grids

ctrl_grid <- trainControl(method="repeatedcv",
                          number = 10,
                          repeats = 3,
                          search="grid",
                          classProbs = TRUE,
                          summaryFunction = twoClassSummary)

set.seed(2020)
DTFit <- train(form=Churn ~ ., 
               data = train, 
               method = "rpart",
               trControl = ctrl_grid,
               tuneGrid=grids,
               metric = "ROC")

DTFit
plot(DTFit)

## Results from DTFit
## ROC 0.81 / 0.82
## Sens 0.87 / 0.87
## Spec 0.49 / 0.53
## cp value 0


# Variable importance information from our model
varImp(DTFit)
plot(varImp(DTFit))

# Variable importance: PaymentMethod_Electronic.check, tenure, InternetService_Fiber.optic
# Contract_Two.year and StreamingMovies_No.internet.service (Above 50.0)

# Averaged confusion matrix across our model
confusionMatrix(DTFit)

# Accuracy 0.78

## 2. Random Search (chosen)
ctrl_random <- trainControl(method="repeatedcv",
                            number = 10,
                            repeats = 3,
                            search ="random",
                            classProbs = TRUE,
                            summaryFunction = twoClassSummary)

## MODEL 1: Using all variables 
set.seed(2020)
DT2Fit <- train(form = Churn ~ ., 
                data = train, 
                method = "rpart",
                trControl = ctrl_random, 
                tuneLength=10,
                metric = "ROC")

DT2Fit
plot(DT2Fit)

## Results from DT2Fit
## ROC 0.80 / 0.82
## Sens 0.89 / 0.87
## Spec 0.48 / 0.53
## cp value 0.00055 / 0

# Variable importance 
varImp(DT2Fit)
plot(varImp(DT2Fit))

# Variable importance: PaymentMethod_Electronic.check, InternetService_Fiber.optic, tenure,
# Contract_Two.year and StreamingMovies_No.internet.service (Above 50.0)

# Averaged confusion matrix 
confusionMatrix(DT2Fit)

# Accuracy 0.78

## we choose our best model --> DTF2it (random search/hyperparameter)

DT2Fit$finalModel

# We can plot the tree for our best fitting model.
rpart.plot(DT2Fit$finalModel) ## Plot with all variables

# Apply our best fitting model to our TRAINING DATA
inpreds_DT <- predict(object=DT2Fit, newdata=train)

confusionMatrix(data=inpreds_DT, reference=train$Churn, 
                mode="prec_recall",positive = 'Yes')

trainDT_perf <- confusionMatrix(data=inpreds_DT, 
                                reference=train$Churn, 
                                mode="prec_recall")

# Apply our best fitting model to our TESTING DATA
outpreds_DT <- predict(object=DT2Fit, newdata=test)

confusionMatrix(data=outpreds_DT, reference=test$Churn,
                mode="prec_recall", positive = 'Yes')

testDT_perf <- confusionMatrix(data=outpreds_DT, 
                               reference=test$Churn,
                               mode="prec_recall")

## Results from Training|Test
#Accuracy 0.84 | 0.79
#Kappa 0.55 | 0.40
#Recall 0.59 | 0.46
#F1 0.66 | 0.54

## Comparing Performance

# Comparing Overall Model Performance
round(cbind(train=trainDT_perf$overall, test=testDT_perf$overall), 2)

# Comparing Class-Level Model Performance
round(cbind(train=trainDT_perf$byClass, test=testDT_perf$byClass), 2)

## MODEL 2: Using only 5 variables

## Random Search 
ctrl_random <- trainControl(method="repeatedcv",
                            number = 10,
                            repeats = 3,
                            search ="random",
                            classProbs = TRUE,
                            summaryFunction = twoClassSummary)

set.seed(2020)
DT3Fit <- train(form = Churn ~ ., 
                data = train2, 
                method = "rpart",
                trControl = ctrl_random, 
                tuneLength=10,
                metric = "ROC")

DT3Fit
plot(DT3Fit)

## Results from DT3Fit
## ROC 0.81
## Sens 0.89
## Spec 0.47
## cp value 0.000133

confusionMatrix(DT3Fit)

DT3Fit$finalModel

# Accuracy: 0.78

# Tree plot
rpart.plot(DT3Fit$finalModel)

# Apply model to our NEW TRAINING DATA
inpreds_DT2 <- predict(object=DT3Fit, newdata=train2)

confusionMatrix(data=inpreds_DT2, reference=train2$Churn,
                mode="prec_recall", positive = 'Yes')

trainDT_perf2 <- confusionMatrix(data=inpreds_DT2, 
                                 reference=train2$Churn, 
                                 mode="prec_recall")

# Apply model to our NEW TESTING DATA
outpreds_DT2 <- predict(object=DT3Fit, newdata=test2)

confusionMatrix(data=outpreds_DT2, reference=test2$Churn,
                mode="prec_recall", positive = 'Yes')

testDT_perf2 <- confusionMatrix(data=outpreds_DT2, 
                                reference=test2$Churn,
                                mode="prec_recall")

## Results from Training2|Test2
#Accuracy 0.80 | 0.79
#Kappa 0.44 | 0.41
#Recall 0.49  | 0.46
#F1 0.57 | 0.54

## Comparing Performance

# Comparing Overall Model Performance
round(cbind(train=trainDT_perf2$overall, test=testDT_perf2$overall), 2)

# Comparing Class-Level Model Performance
round(cbind(train=trainDT_perf2$byClass, test=testDT_perf2$byClass), 2)


save.image("Final_Group3.RData")
