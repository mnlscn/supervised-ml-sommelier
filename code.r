This is an [R Markdown](http://rmarkdown.rstudio.com) Notebook used to evaluate various machine learning models used to predict white wine quality.

Load the needed packages and libraries
```{r message=FALSE, warning=FALSE}
install.packages("xgboost")
install.packages("caret")
install.packages('mlbench')
install.packages('ROCR')
library(visdat)
library(skimr)
library(DataExplorer)
library(corrplot)
library(ggplot2)
library(reshape)
library(GGally)
library(bestglm)
library(MASS)
library(randomForest)
library(caret)
require(xgboost)
library(reshape)
library(tree)
library(gbm)
library(keras)
library(ROCR)
```

Set the working directory and load the training (wwtrd) data
```{r message=FALSE, warning=FALSE}
setwd("C:/Users/dpaga/OneDrive/Documents/0.UNICT_DSP/2. Data Analysis II Sem/Final Report/FS Draft v3")
wwtrd<-read.csv('winequality-white_train.csv')
wwtrd<-subset(wwtrd, select=-index)
vis_dat(wwtrd)
```

```{r}
skim(wwtrd)
```

```{r}
plot_intro(wwtrd)
```

```{r}
plot_bar(as.factor(wwtrd$quality))
```

```{r}
plot_histogram(wwtrd[-length(wwtrd)])
```

```{r}
plot_density(wwtrd[-length(wwtrd)])
```

```{r}
plot_qq(wwtrd[-length(wwtrd)])
```

```{r}
corrplot.mixed(cor(wwtrd[-length(wwtrd)]), number.cex=0.7, tl.cex=0.7)
```
#### Initial Draft Data Analysis Summary

1. Luckily, there is no missing data to deal with.
2. There are 11 numeric predictor variables and 1 outcome ordinal variable.
3. The density variable appears to most closely resemble a normal distribution.
4. Since alcohol is weight less than water there is no surprise that the alcohol variable and density are very strongly negatively correlated with a -0.76 negative correlation.
5. Since the fermentation of sugar is what produces the alcohol it is also not surprising that residual.sugar and alcohol have a negative correlation of -0.45.
6. Residual.sugar and density have the strongest correlation of 0.85 which is due to the fact that sugar is heavier than water so where there is more residual.sugar in the wine the wine is more dense or heavy.
7. Looking at the variable scatter plots as compared to quality it looks like higher quality wines are generally higher in alcohol, lower in residual sugar and density that the lower quality wines.

```{r}
str(wwtrd)
wwtrd.class<-wwtrd
wwtrd.class[wwtrd.class$quality<6, 'quality']<-'low'
wwtrd.class[wwtrd.class$quality!='low' & wwtrd.class$quality>6, 'quality']<-'high'
wwtrd.class[wwtrd.class$quality!='low' & wwtrd.class$quality!='high', 'quality']<-'medium'
wwtrd.class$quality<-factor(wwtrd.class$quality, levels=c('low', 'medium', 'high'))
summary(wwtrd.class$quality)
```

```{r}
ggplot(wwtrd.class, aes(x=quality))+geom_bar(aes(fill=quality))
```

```{r message=FALSE, warning=FALSE}
meltData <- melt(wwtrd.class)
meltData$variable<-factor(meltData$variable)
ggplot(meltData, aes(variable, value))+ geom_boxplot() + facet_wrap(~variable,scale="free")
```

```{r warning= FALSE, message=FALSE}
ggp = ggpairs(wwtrd, progress=TRUE, upper = list(continuous = wrap("cor", size = 2.5)),
        lower = list(continuous = wrap("points", alpha = 0.2, size=0.1),
              combo = wrap("dot", alpha = 1, size=0.2) ))+theme_grey(base_size = 5.3)
print(ggp, progress = FALSE)
```

In order to get a deeper understanding of the predictors' interactions, corrplot() functions perfomrs hierarchical clustering in two groups, based on the correlation values. This can help us in finding similar trend for variables belonging to the same clusters.

```{r}
corrplot(cor(wwtrd[-length(wwtrd)]), method="number", order="hclust", number.cex=0.7, addrect=2)
```

```{r warning= FALSE, message=FALSE}}
ggp2 <- ggpairs(wwtrd.class, aes(color=quality),
        upper = list(continuous = wrap("cor", size = 2)),
        lower = list(continuous = wrap("points", alpha = 0.2, size=0.05),
              combo = wrap("dot", alpha = 1, size=0.2) ))+theme_grey(base_size = 5.3)
print(ggp2, progress = FALSE)
```

```{r}
wwtrd.mean.aggr<-aggregate(meltData$value, by=list(meltData$quality,meltData$variable), mean)
colnames(wwtrd.mean.aggr)<-c('quality', 'variable', 'mean')

wwtrd.sd.aggr<-aggregate(meltData$value, by=list(meltData$quality,meltData$variable), sd)
colnames(wwtrd.sd.aggr)<-c('quality', 'variable', 'sd')
```

```{r}
ggplot(meltData[meltData$variable %in% c('fixed.acidity', 'citric.acid'),], aes(value, color=quality))+
geom_density()+facet_wrap(~variable,scales="free")
```

```{r}
ggplot(meltData[meltData$variable %in% c('fixed.acidity', 'citric.acid'),], aes(variable, value, fill=quality))+ geom_boxplot(notch=T) + facet_wrap(~variable,scale="free")
```

```{r}
subset(wwtrd.mean.aggr, variable %in% c('fixed.acidity', 'citric.acid'))
subset(wwtrd.sd.aggr, variable %in% c('fixed.acidity', 'citric.acid'))
```

```{r}
ggplot(meltData[meltData$variable %in% c('chlorides','residual.sugar', 'density','free.sulfur.dioxide','total.sulfur.dioxide'),], aes(value, color=quality))+
geom_density()+facet_wrap(~variable,scales="free")
```

```{r}
ggplot(meltData[meltData$variable %in% c('chlorides','residual.sugar', 'density','free.sulfur.dioxide','total.sulfur.dioxide'),], aes(variable, value, fill=quality))+ geom_boxplot(notch=T) + facet_wrap(~variable,scale="free")
```

```{r}
subset(wwtrd.mean.aggr, variable %in% c('chlorides','residual.sugar', 'density','free.sulfur.dioxide','total.sulfur.dioxide'))

subset(wwtrd.sd.aggr, variable %in% c('chlorides','residual.sugar', 'density','free.sulfur.dioxide','total.sulfur.dioxide'))
```

```{r}
ggplot(meltData[meltData$variable %in% c('pH', 'sulphates'),], aes(value, color=quality))+geom_density() + facet_wrap(~variable,scales="free")
```

```{r}
ggplot(meltData[meltData$variable %in% c('pH', 'sulphates'),], aes(variable, value, fill=quality))+ geom_boxplot(notch=T) + facet_wrap(~variable,scale="free")
```

```{r}
subset(wwtrd.mean.aggr, variable %in% c('pH', 'sulphates'))
subset(wwtrd.sd.aggr, variable %in% c('pH', 'sulphates'))
```

```{r}
ggplot(meltData[meltData$variable %in% c('volatile.acidity', 'alcohol'),], aes(value, color=quality))+geom_density() + facet_wrap(~variable,scales="free")
```

```{r}
ggplot(meltData[meltData$variable %in% c('density', 'alcohol', 'residual.sugar'),], aes(variable, value, fill=quality))+ geom_boxplot(notch=T) + facet_wrap(~variable,scale="free")
```

```{r}
subset(wwtrd.mean.aggr, variable %in% c('volatile.acidity', 'alcohol'))
subset(wwtrd.sd.aggr, variable %in% c('volatile.acidity', 'alcohol'))
```

Removing all the 6-ranked wines ("the medium" class), for which is rather difficult detecting characteristic features - they are average wines, after all - the scatterplot shows that the points are quite well separated for each couple of variables, reflecting the opposite qualities of the wines.

```{r}
ggp3 <- ggpairs(wwtrd.class[wwtrd.class$quality!='medium',], aes(color=quality),
        upper = list(continuous = wrap("cor", size = 2)),
        lower = list(continuous = wrap("points", alpha = 0.2, size=0.1),
              combo = wrap("dot", alpha = 1, size=0.2) ))+theme_grey(base_size = 5.3)
print(ggp3, progress = FALSE)
```

Thanks to the library _bestglm_ the best logistic regressor is found according to different criteria.

```{r}
# Find the best logistic regressor - Data preparation
wwtrd.class[wwtrd.class$quality=='medium', 'quality']<-'high'
wwtrd.class$quality<-factor(wwtrd.class$quality, levels=c('low', 'high'))
wwtrd.class.glm <- within(wwtrd.class, {
    y    <- quality         # bwt into y
    quality  <- NULL        # Delete bwt
})

wwtrd.class.glm$y<-as.character(wwtrd.class.glm$y)
wwtrd.class.glm[wwtrd.class.glm$y=='low',]$y<-0
wwtrd.class.glm[wwtrd.class.glm$y=='high','y']<-1
wwtrd.class.glm$y<-as.integer(wwtrd.class.glm$y)
```

Most of the predictors present very low collinearity. Density especially, but also alcohol and residual sugar have an important level of collinearity. This addresses to not include all of them in the selected regressor.

```{r}
#Collinearity of the predictors
vifx(wwtrd.class.glm[-12])
```

```{r}
# Best regressor according to AIC
best.reg.aic<-bestglm(wwtrd.class.glm, family=binomial, IC='AIC')
best.reg.aic
```

Even though this regressor minimizes AIC, Std.Error for density is really too big with respect to its estimate. There's a lot of uncertainty on the intercept too and also other predictors show consistent std.errors in comparison to their estimates.

```{r}
#Validation - data preparation
wwvd<-read.csv('winequality-white_validation.csv')
wwvd<-subset(wwvd, select=-X)
wwvd[wwvd$quality<6, 'quality']<-0
wwvd[wwvd$quality !=0, 'quality']<-1

#Validation
glm.probs.aic<-predict(best.reg.aic$BestModel, wwvd, type='response')

#probability of being high-quality wine
head(glm.probs.aic)
```

```{r}
glm.pred.aic<-glm.probs.aic
glm.pred.aic[glm.probs.aic<.5]<-"low"
glm.pred.aic[glm.probs.aic>.5]<-"high"

#predicted class
head(glm.pred.aic)
```

```{r}
#Confusion matrix
table(glm.pred.aic,wwvd$quality)
confMat.aic<-addmargins(table(glm.pred.aic,wwvd$quality))
confMat.aic
```

```{r}
#Accuracy and error rate
accuracy.aic<-(confMat.aic[1,1]+confMat.aic[2,2])/nrow(wwvd)*100 
Err.aic<-100-accuracy.aic   # misclassification error
accuracy.aic
Err.aic
```

```{r}
# Best regressor according to BIC
best.reg.bic<-bestglm(wwtrd.class.glm, family=binomial, IC='BIC')
best.reg.bic
```

The proportion of the standard error with respect to previous method has generally reduced.

```{r}
glm.probs.bic<-predict(best.reg.bic$BestModel, wwvd, type='response')
head(glm.probs.bic)
```

```{r}
glm.pred.bic<-glm.probs.bic
glm.pred.bic[glm.probs.bic<.5]="low"
glm.pred.bic[glm.probs.bic>.5]="high"
#predicted class
head(glm.pred.bic)
```

```{r}
#Confusion matrix
table(glm.pred.bic,wwvd$quality)
confMat.bic<-addmargins(table(glm.pred.bic,wwvd$quality))
confMat.bic
```

```{r}
#Accuracy and error rate
accuracy.bic=(confMat.bic[1,1]+confMat.bic[2,2])/nrow(wwvd)*100 
err.bic=100-accuracy.bic
accuracy.bic
err.bic
```

```{r}
# Best regressor according to cross-validation
best.reg.cv<-bestglm(wwtrd.class.glm, family=binomial, IC='CV', CVArgs = list(Method='HTF', K=11, REP=5))
best.reg.cv
```

Simpler method with best p-values and std. errors. As you can see from the following chunk, the percentual loss in terms of AIC and BIC from the best methods is really small.

```{r}
AIC(best.reg.cv$BestModel)
BIC(best.reg.cv$BestModel)

((AIC(best.reg.aic$BestModel)-AIC(best.reg.cv$BestModel))/AIC(best.reg.aic$BestModel))*100
((BIC(best.reg.bic$BestModel)-BIC(best.reg.cv$BestModel))/BIC(best.reg.aic$BestModel))*100
```

```{r}
#probability of being high-quality wine
glm.probs.cv<-predict(best.reg.cv$BestModel, wwvd, type='response')
head(glm.probs.cv)
```

```{r}
glm.pred.cv<-glm.probs.cv
glm.pred.cv[glm.probs.cv<.5]<-"low"
glm.pred.cv[glm.probs.cv>.5]<-"high"

#predicted class
head(glm.pred.cv)
```
```{r}
#Accuracy and error rate
accuracy.cv<-(confMat.cv[1,1]+confMat.cv[2,2])/nrow(wwvd)*100 
Err.cv<-100-accuracy.cv   # misclassification error
accuracy.cv
Err.cv
```

```{r}
#Confusion matrix
table(glm.pred.cv,wwvd$quality)
confMat.cv<-addmargins(table(glm.pred.cv,wwvd$quality))
confMat.cv
```
```{r}
#TPR  and FPR
TPR.cv<-confMat.cv[2,2]/(confMat.cv[2,2]+confMat.cv[1,2])*100
FPR.cv<-confMat.cv[2,1]/(confMat.cv[2,1]+confMat.cv[1,1])*100


TPR.cv
FPR.cv
```



```{r}
pred.t.LR_T=prediction(glm.probs.cv, wwvd$quality)
perf.t.LR_T=performance(pred.t.LR_T, measure = "tpr", x.measure = "fpr")

plot(perf.t.LR_T,colorize=TRUE,lwd=2) 
plot(perf.t.LR_T,colorize=TRUE,lwd=2, print.cutoffs.at=c(0.2,0.5,0.8)) 
abline(a=0,b=1, lty=2)

LR_T.auc=performance(pred.t.LR_T, measure = "auc") 
LR_T.auc@y.values[[1]]
```
```{r}
vifx(wwtrd.class.glm[c('alcohol', 'residual.sugar', 'volatile.acidity')])
```

```{r}
wwtrd.class.lda<-wwtrd.class
wwtrd.class.lda$quality<-as.character(wwtrd.class.lda$quality)
wwtrd.class.lda[wwtrd.class.lda$quality=='medium' | wwtrd.class.lda$quality=='high', 'quality']<-1
wwtrd.class.lda[wwtrd.class.lda$quality!=1, 'quality']<-0
wwtrd.class.lda
lda.fit<-lda(quality~volatile.acidity+residual.sugar+alcohol, family=binomial, data=wwtrd.class.lda)
lda.fit
```

```{r}
lda.predict=predict(lda.fit, wwvd, type='response')
lda.pred.class<-lda.predict$class
head(lda.pred.class)
```

```{r}
confMat.LDA<-addmargins(table(lda.pred.class,wwvd$quality))
confMat.LDA
```

```{r}
accuracy.LDA=(confMat.LDA[1,1]+confMat.LDA[2,2])/nrow(wwvd)*100 # accuracy
Err.LDA=100-accuracy.LDA # misclassification error

accuracy.LDA
Err.LDA
```

```{r}
TPR.LDA<-confMat.LDA[2,2]/(confMat.LDA[2,2]+confMat.LDA[1,2])*100
FPR.LDA<-confMat.LDA[2,1]/(confMat.LDA[2,1]+confMat.LDA[1,1])*100

TPR.LDA
FPR.LDA

```


```{r}
pred.t.LDA_T=prediction(lda.predict$posterior[,2], wwvd$quality)
perf.t.LDA_T=performance(pred.t.LDA_T, measure = "tpr", x.measure = "fpr")
plot(perf.t.LDA_T,colorize=TRUE,lwd=2, print.cutoffs.at=c(0.2,0.5,0.8))
abline(a=0,b=1, lty=2)
```

```{r}
LDA_T.auc=performance(pred.t.LDA_T, measure = "auc") 
LDA_T.auc@y.values[[1]]
```

All the methods performed give similar and disappointing performances. Choosing cross-validation as optimal criterion to choose the regressor, we obtain slightly better results with just 3 predictors. This is enough to choose this model as the best. Alcohol, residual sugar are always selected in each subset, so they should explain most of the data variability for sure.

## Decision Tree Analysis

```{r}
wwtrd$qlevels <- factor(ifelse(wwtrd$quality <= 5, "No", "Yes"))
table(wwtrd$qlevels)
table(wwtrd$quality)
wwvald<-read.csv('winequality-white_validation.csv')
wwvald<-subset(wwvald, select=-index)
wwvald$qlevels <- factor(ifelse(wwvald$quality <= 5, "No", "Yes"))
table(wwvald$qlevels)
table(wwvald$quality)
```

```{r}
base.tree.fit = tree(qlevels ~ . -quality, data=wwtrd) 
```
```{r}
summary(base.tree.fit)
```

```{r}
base.tree.fit
```

```{r}
set.seed(99)
base.tree.pred <- predict(base.tree.fit, wwvald, type = "class")
print(addmargins(table(base.tree.pred , wwvald$qlevels)))
```

```{r}
print((126+578)/983) #shows the accuracy rate on the validation data
```

```{r}
summary(base.tree.pred)
```
```{r} 
#Now let us prun the base tree to see if we get a better result on the validation data.
cv.wwtrd.base.tree <- cv.tree(base.tree.fit , FUN = prune.misclass)
```

```{r}
cv.wwtrd.base.tree
```

```{r}
par(mfrow = c(1, 2))
plot(cv.wwtrd.base.tree$size, cv.wwtrd.base.tree$dev, type = "b")
plot(cv.wwtrd.base.tree$k, cv.wwtrd.base.tree$dev, type = "b")
```

```{r}
prune.wwtrd.basetree <- prune.misclass(base.tree.fit , best = 6)
plot(prune.wwtrd.basetree)
text(prune.wwtrd.basetree , pretty = 0)
```
```{r}
prun.tree.pred <- predict(prune.wwtrd.basetree , wwvald, type = "class")
table(prun.tree.pred , wwvald$qlevels)
```
```{r}
print(addmargins(table(prun.tree.pred , wwvald$qlevels)))
```
```{r}
print((126+578)/983) #shows the accuracy rate on the test data  #It is interesting how the optimal prunned model at 71.6% was the same as the basic model at 71.6%.
```

#Boosting - Now let us see of a boosing model does on the data.
#SINCE GBM REQUIRES the response to be 0 or 1, I will modify wwtrd & wwvald.  Instructions to run model found at: https://rpubs.com/omicsdata/gbm
```{r}
xgb.train<-read.csv('winequality-white_train.csv')
xgb.train<-subset(xgb.train, select=-index)
xgb.vald<-read.csv('winequality-white_validation.csv')
xgb.vald<-subset(xgb.vald, select=-index)
```
```{r}
xgb.train.f<-xgb.train
xgb.train.n<-xgb.train
xgb.vald.f<-xgb.vald
xgb.vald.n<-xgb.vald
xgb.train.f$quality <- factor(ifelse(xgb.train$quality <= 5, "0", "1"))
xgb.train.n$quality <- as.numeric(ifelse(xgb.train$quality <= 5, "0", "1"))
xgb.vald.f$quality <- factor(ifelse(xgb.vald$quality <= 5, "0", "1"))
xgb.vald.n$quality <- as.numeric(ifelse(xgb.vald$quality <= 5, "0", "1"))
table(xgb.train$quality)
table(xgb.vald$quality)
```

```{r}
gbm.model = gbm(quality~., data=xgb.train.n, shrinkage=0.01, distribution = 'bernoulli', cv.folds=5, n.trees=3000, verbose=F)
```
```{r}
best.iter = gbm.perf(gbm.model, method="cv")
```
```{r}
best.iter
```
```{r}
summary(gbm.model)
```
```{r}
plot.gbm(gbm.model, 1, best.iter)
plot.gbm(gbm.model, 2, best.iter)
plot.gbm(gbm.model, 3, best.iter)
plot.gbm(gbm.model, 4, best.iter)
plot.gbm(gbm.model, 5, best.iter)
plot.gbm(gbm.model, 6, best.iter)
plot.gbm(gbm.model, 7, best.iter)
plot.gbm(gbm.model, 8, best.iter)
plot.gbm(gbm.model, 9, best.iter)
plot.gbm(gbm.model, 10, best.iter)
plot.gbm(gbm.model, 11, best.iter)
```

```{r}
mydata=xgb.train.f

set.seed(123)
fitControl = trainControl(method="cv", number=5, returnResamp = "all")

model2 = train(quality~., data=mydata, method="gbm",distribution="bernoulli", trControl=fitControl, verbose=F, tuneGrid=data.frame(.n.trees=best.iter, .shrinkage=0.01, .interaction.depth=1, .n.minobsinnode=1))
```
```{r}
model2
```

```{r}
summary(model2)
```

```{r}
confusionMatrix(model2)
```
```{r}
mPred = predict(model2, xgb.vald.f, na.action = na.pass)
postResample(mPred, xgb.vald.f$quality)
```

```{r}
confusionMatrix(mPred, xgb.vald.f$quality)
```

```{r}
#Now let us try Bagging and RandomForest Trees
set.seed(99)
bag.wwtrd<-randomForest(qlevels ~. - quality, data = wwtrd, mtry = 11, importance = TRUE, type = class)
```

```{r}
bag.wwtrd
```
```{r}
pred.bag <- predict(bag.wwtrd , wwvald, type="class")
print(addmargins(table(pred.bag , wwvald$qlevels)))
```
```{r}
print((225+577)/983)
#The accuracy results of the bagging tree that uses all variables when branching is 81.6%. 
```
#### Now we will look at the random forrest model in detail to find the optimal random forrest model for the data.  First I will run a default random forrest model then I will hypertune the random forrest model to find the optimal model perameters using a grid search and then I will use the tuneRF algorithim.  Based on the results we will choose the optimal mtry perameter for the random forrest model.  

```{r}
x<-wwtrd[,1:11]
y<-wwtrd[,13]
```
```{r}
# Create model with default parameters
control <- trainControl(method="repeatedcv", number=10, repeats=3)
seed <- 99
metric <- "Accuracy"
set.seed(seed)
mtry <- sqrt(ncol(x))
tunegrid <- expand.grid(.mtry=mtry)
rf_default <- train(qlevels~. - quality, data=wwtrd, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)
print(rf_default)
```
TUNE THE MODEL USING GRID SEARCH
```{r}
control <- trainControl(method="repeatedcv", number=3, repeats=1, search="grid")
set.seed(seed)
tunegrid <- expand.grid(.mtry=c(1:11))
rf_gridsearch <- train(qlevels~. -quality, data=wwtrd, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)
print(rf_gridsearch)
plot(rf_gridsearch)
```

```{r}
# Algorithm Tune (tuneRF)
set.seed(seed)
bestmtry <- tuneRF(x, y, stepFactor=1.5, improve=1e-5, ntree=500)
print(bestmtry)
```

# Now we will use mtry = 2 on the validaton data
```{r}
set.seed(99)
rf2.wwtrd<-randomForest(qlevels ~. - quality, data = wwtrd, mtry = 2, importance = TRUE, type = class, ntree = 5000)
rf2.wwtrd
pred.rf2 <- predict(rf2.wwtrd , wwvald, type="class")
```

```{r}
print(addmargins(table(pred.rf2 , wwvald$qlevels)))
```
```{r}
print((215+593)/983)
```

```{r}
importance(rf2.wwtrd)
```

```{r}
varImpPlot(rf2.wwtrd)
```


##Neural Network Model
```{r}
library('keras')
wwtrd.nn<-wwtrd.class
wwvd.nn<-wwvd[,1:12]

wwtrd.nn$quality<-as.character(wwtrd.nn$quality)
wwtrd.nn[wwtrd.nn$quality=='high', 'quality']<-1
wwtrd.nn[wwtrd.nn$quality!=1, 'quality']<-0

wwvd.nn$quality<-as.character(wwvd.nn$quality)
wwvd.nn[wwvd.nn$quality=='high', 'quality']<-1
wwvd.nn[wwvd.nn$quality!=1, 'quality']<-0

wwtrd.nn.scaled<-scale(subset(wwtrd.nn, select = - quality))
wwvd.nn.scaled<-scale(subset(wwvd.nn, select = - quality))
x_train <- array_reshape(wwtrd.nn.scaled, c(nrow(wwtrd.nn.scaled), ncol(wwtrd.nn.scaled)))
x_valid <- array_reshape(wwvd.nn.scaled, c(nrow(wwvd.nn.scaled), ncol(wwvd.nn.scaled)))

y_train <- to_categorical(wwtrd.nn$quality, 2)
y_valid <- to_categorical(wwvd.nn$quality, 2)
wwtrd.nn
```

```{r message=FALSE, error=FALSE}
modelnn <- keras_model_sequential()
modelnn %>%
   layer_dense(units = 32, activation = "relu",
       input_shape = ncol(wwtrd.nn.scaled)) %>%
   layer_dropout(rate=0.2) %>%
     layer_dense(units = 4, activation = "sigmoid")%>%
   layer_dropout(rate=0.1) %>%
   layer_dense(units = 2, activation = "softmax")
```

```{r}
summary(modelnn)
```

```{r}
modelnn %>% compile(loss = "categorical_crossentropy",
    optimizer = optimizer_rmsprop(), metrics = c("accuracy")
  )

callback<-callback_early_stopping(
    monitor="val_loss",
    patience=50,
    min_delta = 0.001,
    mode="min",
    restore_best_weights = TRUE
)
```

```{r}
   system.time (
     history <- modelnn%>%
      fit(x_train, y_train, epochs = 300, batch_size=64,
          validation_data = list(x_valid, y_valid),
          callback=callback)
   )
```

```{r}
accuracy <- function(pred, truth) {
   mean(drop(as.numeric(pred)) == drop(truth)) }
modelnn %>% predict(x_valid) %>% k_argmax() %>% accuracy(wwvd.nn$quality)
```

Now we will run the best model on the test data.  We will use the Random Forest model since it produced the highet accuracy on the validation data.

```{r}
wwtest<-read.csv('winequality-white_test_wquality.csv')
#wwtest<-subset(wwtest, select=-index)
wwtest$qlevels <- factor(ifelse(wwtest$quality <= 5, "No", "Yes"))
head(wwtest)
```


```{r}
pred.rf2_test <- predict(rf2.wwtrd, wwtest, type="class")
print(addmargins(table(pred.rf2_test , wwtest$qlevels)))
```

```{r}
print((223+592)/978)
```
```{r}
results <-as.integer(pred.rf2_test)
results1 <-as.data.frame(results)
test_with_results <-cbind(wwtest, results1)
test_with_results$results <- factor(ifelse(test_with_results$results == 1, "No", "Yes"))
head(test_with_results, 50)
write.csv(test_with_results,"test_file_w_final_results.csv", row.names = FALSE)
```
