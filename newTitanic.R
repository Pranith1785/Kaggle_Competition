###**************************** Titanic ***************************************************
rm(list = ls(all=TRUE))
##****************************** Libraries *****************************************
library(plyr)
library(Hmisc)
library(knitr)
library(sqldf)
library(caret)
library(DMwR)
library(glmnet)
library(dplyr)
library(nnet)
library(e1071)
library(rpart)
library(rpart.plot)
library(C50)
library(class)
library(e1071)
library(kernlab)
library(randomForest)
library(ipred)
library(ada)
library(dummies)
library(ggplot2)
library(corrplot)
library(ROCR)

getwd()
setwd("C:\\Users\\gouripra\\Documents\\Pranith Kumar G\\Kaggle\\Titanic")

titanicData <- read.csv("train.csv",sep = ",",na.strings = c(" ","","?","NA"))

pdata <- titanicData

testData <- read.csv("test.csv")

sum(is.na(titanicData))
sum(is.na(testData))

summary(titanicData)
str(titanicData)
colnames(titanicData)

##********************************************************************************************
## Get NA columns
sapply(titanicData,function(x){sum(is.na(x))})
colnames(titanicData)[colSums(is.na(titanicData)) > 0]

sapply(testData,function(x){sum(is.na(x))})

nearZeroVar(titanicData,freqCut = 90/10)

##***********************************************************************************************
###### Type connversion
titanicData$Survived <- as.factor(titanicData$Survived)
titanicData$Pclass <- as.factor(titanicData$Pclass)
titanicData$PassengerId <- as.factor(titanicData$PassengerId)

testData$Pclass <- as.factor(testData$Pclass)
testData$PassengerId <- as.factor(testData$PassengerId)

##********************************************************************************************
##### Visualisation

ggplot(titanicData,aes(x = Sex, fill = Survived)) + geom_bar(stat = 'count',position = 'dodge') 
   

sqldf("select count(Survived) as from titanicData where Survived = 1 union select count(Survived) from titanicData where Survived = 0")

sqldf("select Survived,Sex,count(*) from titanicData group by Sex, Survived")

sqldf("select Survived, Sex, PClass,Count(*) from titanicData group by Sex, Survived, Pclass")


ggplot(titanicData[!is.na(titanicData$Survived),],aes(x = Pclass, fill = Survived)) + 
      geom_bar(stat = 'count',position = "fill") + 
      labs(x= "Survived or not", y = 'count') + facet_grid(.~Sex)

##********************************************************************************************
##### Feature engineering

######### Title and Surname

titanicData$Name <- as.character(titanicData$Name)
titanicData$Title <- sapply(titanicData$Name,function(x){trimws(strsplit(x,split = "[,.]")[[1]][2])})
titanicData$Surname <- sapply(titanicData$Name,function(x){trimws(strsplit(x,split = "[,.]")[[1]][1])})

testData$Name <- as.character(testData$Name)
testData$Title <- sapply(testData$Name,function(x){trimws(strsplit(x,split = "[,.]")[[1]][2])})
testData$Surname <- sapply(testData$Name,function(x){trimws(strsplit(x,split = "[,.]")[[1]][1])})

sqldf("select Title,Sex, count(*) from titanicData Group by Title , Sex")

table(titanicData$Sex,titanicData$Title, titanicData$Survived)

titanicData$Title[titanicData$Title == "Mme"] <- "Mrs"
titanicData$Title[titanicData$Title %in% c("Ms","Mlle")] <- "Miss"
titanicData$Title[!(titanicData$Title %in% c("Mrs","Miss","Mr","Master"))] <- "other"

testData$Title[testData$Title == "Mme"] <- "Mrs"
testData$Title[testData$Title %in% c("Ms","Mlle")] <- "Miss"
testData$Title[!(testData$Title %in% c("Mrs","Miss","Mr","Master"))] <- "other"


table(titanicData$Sex,titanicData$Title)

titanicData$Title <- as.factor(titanicData$Title)
titanicData$Surname <-  as.factor(titanicData$Surname)

testData$Title <- as.factor(testData$Title)
testData$Surname <-  as.factor(testData$Surname)

ggplot(titanicData,aes(x = Title, fill = Survived)) +
   geom_bar(stat = 'Count', position = 'dodge') +  
      labs(x = "Title", y = "Passenger count") +
     geom_label(stat = 'count', aes(label = ..count..)) + 
        theme_grey()

titanicData$Name <- NULL
testData$Name <- NULL

##********************************************************************************************
##### Feature engineering   -  Ticket2 - replacing last 2 charcters
  
titanicData$Ticket2 <- sub(pattern = "..$",replacement = "xx",x = titanicData$Ticket)

testData$Ticket2 <- sub(pattern = "..$",replacement = "xx",x = testData$Ticket)

##********************************************************************************************
##### Feature engineering   -  FamilySize

titanicData$Fsize <- titanicData$SibSp + titanicData$Parch + 1

ggplot(titanicData,aes(x= Fsize, fill = Survived)) +
  geom_bar(stat = 'count',position = 'Dodge') +
    labs(x="Family Size") + theme_grey()

table(titanicData$Fsize,titanicData$Survived,titanicData$Sex)

rest <- titanicData %>%
          select(PassengerId, Title, Age, Ticket, Ticket2, Surname, Fsize) %>%
          filter(Fsize=='1') %>%
          group_by(Ticket2, Surname) %>%
           dplyr::summarise(count=n())
rest <- rest[rest$count>1,]
rest1 <- titanicData[(titanicData$Ticket2 %in% rest$Ticket2 & titanicData$Surname %in% rest$Surname & titanicData$Fsize=='1'), c('PassengerId', 'Surname', 'Title', 'Age', 'Ticket', 'Ticket2', 'Fsize', 'SibSp', 'Parch')]
rest1 <- left_join(rest1, rest, by = c("Surname", "Ticket2"))
rest1 <- rest1[!is.na(rest1$count),]
rest1 <- rest1 %>%
            arrange(Surname, Ticket2)
kable(rest1[1:12,])

titanicData <- left_join(titanicData, rest1,by = c("PassengerId", "Surname", "Title","Age","SibSp", "Parch","Ticket","Ticket2","Fsize"))

for (i in 1:nrow(titanicData)){
  if (!is.na(titanicData$count[i])){
      titanicData$Fsize[i] <- titanicData$count[i]
  }
}

ggplot(titanicData,aes(x= Fsize,fill = Survived)) +
        geom_bar(stat = 'count',position = 'dodge') +
          scale_x_continuous(breaks = c(1:11)) + theme_classic()

###creating group from Fsize
titanicData$Group[titanicData$Fsize == 1] <- "Solo"
titanicData$Group[titanicData$Fsize == 2] <- "duo"
titanicData$Group[titanicData$Fsize == 3] <- "Medium"
titanicData$Group[titanicData$Fsize == 4] <- "Medium"
titanicData$Group[titanicData$Fsize >= 5] <- "Large"

titanicData$count <- NULL

#####
testData$Fsize <- testData$SibSp + testData$Parch + 1

rest <- testData %>%
  select(PassengerId, Title, Age, Ticket, Ticket2, Surname, Fsize) %>%
  filter(Fsize=='1') %>%
  group_by(Ticket2, Surname) %>%
  dplyr::summarise(count=n())
rest <- rest[rest$count>1,]
rest1 <- testData[(testData$Ticket2 %in% rest$Ticket2 & testData$Surname %in% rest$Surname & testData$Fsize=='1'), c('PassengerId', 'Surname', 'Title', 'Age', 'Ticket', 'Ticket2', 'Fsize', 'SibSp', 'Parch')]
rest1 <- left_join(rest1, rest, by = c("Surname", "Ticket2"))
rest1 <- rest1[!is.na(rest1$count),]
rest1 <- rest1 %>%
  arrange(Surname, Ticket2)

testData <- left_join(testData, rest1,by = c("PassengerId", "Surname", "Title","Age","SibSp", "Parch","Ticket","Ticket2","Fsize"))

for (i in 1:nrow(testData)){
  if (!is.na(testData$count[i])){
    testData$Fsize[i] <- testData$count[i]
  }
}

##creating group from Fsize
testData$Group[testData$Fsize == 1] <- "Solo"
testData$Group[testData$Fsize == 2] <- "duo"
testData$Group[testData$Fsize == 3] <- "Medium"
testData$Group[testData$Fsize == 4] <- "Medium"
testData$Group[testData$Fsize >= 5] <- "Large"

testData$count <- NULL

##******************************************************************************************************
## Embarked - location 

head(titanicData[is.na(titanicData$Embarked),])

##Calculate the Fare amount based on PClass and embarked

Fare_table <-titanicData[(!is.na(titanicData$Embarked)& !is.na(titanicData$Fare)),] %>%
                   group_by(Pclass,Embarked) %>%
                     dplyr::summarise(median_Fare = median(Fare),Mean_Fare = mean(Fare))
    
kable(Fare_table)

titanicData$Embarked[is.na(titanicData$Embarked)] <- 'C'

##******************************************************************************************************
## Fare

titanicData <- left_join(titanicData,Fare_table,by = c("Pclass","Embarked"))
titanicData$Fare[titanicData$Fare == 0] <- titanicData$median_Fare[titanicData$Fare == 0]

testData <- left_join(testData,Fare_table,by = c("Pclass","Embarked"))
testData$Fare[is.na(testData$Fare)] <- 0
testData$Fare[testData$Fare == 0] <- testData$median_Fare[testData$Fare == 0]

summary(titanicData$Fare)

ggplot(all, aes(x=Fare)) +
  geom_histogram(binwidth = 5, fill='blue') + theme_grey() +
  scale_x_continuous(breaks= seq(0, 520, by=30))


titanicData$FareBins <- cut2(titanicData$Fare, g=5)

ggplot(titanicData[!is.na(titanicData$Survived),], aes(x=FareBins, fill=Survived))+
  geom_bar(stat='count') + theme_grey() + facet_grid(.~Pclass)+
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

summary(titanicData$FareBins)

titanicData$FareGroup[titanicData$Fare < 7.92] <- 1
titanicData$FareGroup[titanicData$Fare >= 7.92 & titanicData$Fare < 11.50] <- 2
titanicData$FareGroup[titanicData$Fare >= 11.50 & titanicData$Fare < 23.25] <- 3
titanicData$FareGroup[titanicData$Fare >= 23.25 & titanicData$Fare < 42.40] <- 4
titanicData$FareGroup[titanicData$Fare >= 42.40] <- 5

titanicData$FareBins <- NULL

testData$FareGroup[testData$Fare < 7.92] <- 1
testData$FareGroup[testData$Fare >= 7.92 & testData$Fare < 11.50] <- 2
testData$FareGroup[testData$Fare >= 11.50 & testData$Fare < 23.25] <- 3
testData$FareGroup[testData$Fare >= 23.25 & testData$Fare < 42.40] <- 4
testData$FareGroup[testData$Fare >= 42.40] <- 5

##******************************************************************************************************
## Age

ggplot(titanicData[!is.na(titanicData$Age),], aes(x = Age, fill = Survived)) +
  geom_density(alpha=0.5, aes(fill=Survived)) + labs(title="Survival density and Age") +
  scale_x_continuous(breaks = scales::pretty_breaks(n = 10)) + theme_grey()

ggplot(titanicData[!is.na(titanicData$Age),],aes(x= Title,y = Age, fill = Survived)) +
        geom_boxplot()

ggplot(titanicData[!is.na(titanicData$Age),],aes(x= Title,y = Age, fill = Pclass)) +
  geom_boxplot()


AgeLm <- lm(Age~ Pclass + Sex + SibSp + Parch + Embarked + Title + Fsize, 
                        data = titanicData[!is.na(titanicData$Age),])

summary(AgeLm)

titanicData$Age_pred <- predict(AgeLm,titanicData)

par(mfrow = c(1,2))
p1 <- hist(titanicData$Age[!is.na(titanicData$Age)],main = "Actual Age values",xlab ="Age",col = 'green')
p2 <- hist(titanicData$Age_pred[!is.na(titanicData$Age_pred)],main = "Actual Age values",xlab="Age",col= 'yellow')

titanicData$Age[is.na(titanicData$Age)] <- titanicData$Age_pred[is.na(titanicData$Age)]

testData$Age_pred <- predict(AgeLm,testData)
testData$Age[is.na(testData$Age)] <- testData$Age_pred[is.na(testData$Age)]

####***********************************************************************************************************************
### cabin
titanicData$Cabin <- as.character(titanicData$Cabin)
titanicData$Cabin[is.na(titanicData$Cabin)] <- 'Z'
titanicData$Cabin <- substring(titanicData$Cabin,1,1)

levels(as.factor(titanicData$Cabin))

ggplot(titanicData,aes(x= Cabin, fill = Survived)) + 
   geom_bar(stat = 'count',position = 'stack') + 
    facet_grid(.~Pclass) + theme_classic()

c1 <- round(prop.table(table(titanicData$Survived,titanicData$Cabin),2)*100)
kable(c1)


testData$Cabin <- as.character(testData$Cabin)
testData$Cabin[is.na(testData$Cabin) | testData$Cabin == ""] <- 'Z'
testData$Cabin <- substring(testData$Cabin,1,1)

###*********************************************************************************************************
### 

sum(is.na(titanicData))
sum(is.na(testData))

##*****************************************************************************************************
## Removing the columns which are not needed

RemCols <- setdiff(colnames(titanicData),c("PassengerId","median_Fare","SibSp","Parch","Cabin","Mean_Fare","Age_pred","Ticket","Surname","Ticket2"))
titanicData <- titanicData[,RemCols]

testRemCols <- setdiff(colnames(testData),c("PassengerId","median_Fare","SibSp","Parch","Cabin","Mean_Fare","Age_pred","Ticket","Surname","Ticket2"))
testData <- testData[,testRemCols]

##*******************************************************************************************************
## type conversion

str(titanicData)
cat_cols = c("Pclass","Sex","Embarked","Title","Group","FareGroup")
num_cols = c("Age","Fare","Fsize")

titanicData[,cat_cols] <- lapply(titanicData[,cat_cols],function(x){as.factor(x)})
titanicData[,num_cols] <- lapply(titanicData[,num_cols],function(x){as.numeric(x)})

testData[,cat_cols] <- lapply(testData[,cat_cols],function(x){as.factor(x)})
testData[,num_cols] <- lapply(testData[,num_cols],function(x){as.numeric(x)})

##*******************************************************************************************************
## Finding Importance -  correlation , Chi-square
corr = cor(titanicData[,num_cols],method = "pearson")
corrplot(corr,method = "number",type = "full")

chisq.test(titanicData$Pclass,titanicData$Sex)

chisq.test(titanicData$Sex,titanicData$Pclass)
##is p is less than 0.05, reject null hypothesis i.e., both are dependent)

##*******************************************************************************************************
## Split the Data
set.seed(1785)
rowNum <- createDataPartition(titanicData$Survived,p=0.7,list = F)
trainData <- titanicData[rowNum,]
valData <- titanicData[-rowNum,]

##*****************************************************************************************************

sample <- read.csv("gender_submission.csv")

##*****************************************************************************************************
## logistic model

LR_Model <- glm(Survived~.,trainData,family = "binomial")
summary(LR_Model)

train_LR <- predict(LR_Model,trainData,type = "response")
val_LR <- predict(LR_Model,valData,type = "response")
test_LR <- predict(LR_Model,testData,type = "response")

train_LR <- as.factor(ifelse(train_LR >0.5,1,0))
val_LR <- as.factor(ifelse(val_LR>0.5,1,0))
test_LR <- as.factor(ifelse(test_LR>0.5,1,0))

confusionMatrix(trainData$Survived,train_LR)
confusionMatrix(valData$Survived,val_LR)
summary(test_LR)

sample$Survived <- test_LR
write.csv(sample,"test_output_LR.csv",row.names = F)


##*******************************************************************************************************
## Naive Bayes

NB_model <- naiveBayes(Survived~.,trainData)
summary(NB_model)

train_NB <- predict(NB_model,trainData) 
val_NB <- predict(NB_model,valData)
test_NB <- predict(NB_model,testData)

confusionMatrix(trainData$Survived,train_NB)
confusionMatrix(valData$Survived,val_NB)
summary(test_NB)

sample$Survived <- test_NB
write.csv(sample,"test_output_NB.csv",row.names = F)

###********************************************************************************************
###  Decision Trees 

trCtrl <- trainControl(method = "cv",number = 5)
set.seed(1234)
rpart_grid <- expand.grid(cp=seq(0,1.5))
DT_rpart_Model <- train(Survived~.,trainData,method = "rpart",trControl = trCtrl,tuneGrid = rpart_grid)

DT_rpart_Model
plot(DT_rpart_Model)

train_DT_rpart <- predict(DT_rpart_Model,trainData)
val_DT_rpart <- predict(DT_rpart_Model,valData)
test_DT_rpart <- predict(DT_rpart_Model,testData,type = "raw")

confusionMatrix(trainData$Survived,train_DT_rpart)
confusionMatrix(valData$Survived,val_DT_rpart)
summary(test_DT_rpart)

sample$Survived <- test_DT_rpart
write.csv(sample,"test_output_DT_rpart.csv",row.names = F)

##***************** SVM linear *********************

trCtrl <- trainControl(method = "cv", number = 5)
grid <- expand.grid(cost = c(0.05,2.5,5.5))
set.seed(3233)
model_svm_linear <- train(Survived~.,trainData, method = "svmLinear2",
                          trControl=trCtrl,
                          tuneGrid = grid)

model_svm_linear
plot(model_svm_linear)

train_svm_linear <- predict(model_svm_linear,trainData)
val_svm_linear <- predict(model_svm_linear,valData)
test_svm_linear <- predict(model_svm_linear,testData)

confusionMatrix(trainData$Survived,train_svm_linear)
confusionMatrix(valData$Survived,val_svm_linear)
summary(test_svm_linear)

sample$Survived <- test_svm_linear
write.csv(sample,"test_output_svmL.csv",row.names = F)

##****************** SVM Radial *****************************

trCtrl <- trainControl(method = "cv", number = 5)
grid <- expand.grid(C = c(0.05,0.5,1.5),sigma = c(0.01,0.5))
set.seed(3233)

model_svm_radial <- train(Survived~.,trainData, method = "svmRadial",
                          trControl=trCtrl,
                          tuneGrid = grid)

model_svm_radial
plot(model_svm_radial)

train_svm_radial <- predict(model_svm_radial,trainData)
val_svm_radial <- predict(model_svm_radial,valData)
test_svm_radial <- predict(model_svm_radial,testData)

confusionMatrix(trainData$Survived,train_svm_radial)
confusionMatrix(valData$Survived,val_svm_radial)
summary(test_svm_radial)

sample$Survived <- test_svm_radial
write.csv(sample,"test_output_svmR.csv",row.names = F)


###****************** Random Forest *******************************

#trCtrl <- trainControl(method="cv", number=10)
#set.seed(1234)
#rf_grid <- expand.grid(mtry=c(4,7))
#model_rf <- train(Survived~.,data = trainData, method = "rf",
#                       trControl=trCtrl,
#                       tuneGrid = rf_grid)

model_rf <- randomForest(Survived ~ .,data = trainData,ntree = 30,mtry = 5)

# We can also look at variable importance from the built model using the importance() function and visualise it using the varImpPlot() funcion
importance(model_rf)
varImpPlot(model_rf)
model_rf
summary(model_rf)

train_rf <- predict(model_rf,trainData)
val_rf <- predict(model_rf,valData)
test_rf <- predict(model_rf,testData)

confusionMatrix(trainData$Survived,train_rf)
confusionMatrix(valData$Survived,val_rf)
summary(test_rf)

sample$Survived <- test_rf
write.csv(sample,"test_output_rf.csv",row.names = F)

###***************** XGBoost ********************
xgb_trcontrol = trainControl(method = "cv",number = 5)

xgb_grid = expand.grid(nrounds = 10,eta = c(0.01, 0.001),max_depth = c(2, 4),
                       gamma = 1, colsample_bytree=0.7, min_child_weight=2,subsample=0.9)
set.seed(1234)
model_xgb = train(Survived~., data=trainData,method = "xgbTree",
                  trControl = xgb_trcontrol,tuneGrid = xgb_grid)

model_xgb
summary(model_xgb)

train_xgb <- predict(model_xgb,trainData)
val_xgb <- predict(model_xgb,valData)
test_xgb <- predict(model_xgb,testData)

confusionMatrix(trainData$Survived,train_xgb)
confusionMatrix(valData$Survived,val_xgb)
summary(test_xgb)

sample$Survived <- test_xgb
write.csv(sample,"test_output_xgb.csv",row.names = F)

#*********************** gbm ************************************

gbmControl <- trainControl(method="cv", number=10)
gbm_grid <- expand.grid(interaction.depth=c(1, 3, 5), n.trees = c(5,10,25,50),
                        shrinkage=c(0.01, 0.001),
                        n.minobsinnode=10)
set.seed(1234)

model_gbm <- train(Survived~., data=trainData,method = "gbm",
                   trControl=gbmControl,
                   tuneGrid=gbm_grid)

model_gbm
summary(model_gbm)

train_gbm <- predict(model_gbm,trainData)
val_gbm <- predict(model_gbm,valData)
test_gbm <- predict(model_gbm,testData)

confusionMatrix(trainData$Survived,train_gbm)
confusionMatrix(valData$Survived,val_gbm)
summary(test_gbm)

sample$Survived <- test_gbm
write.csv(sample,"test_output_gbm.csv",row.names = F)

