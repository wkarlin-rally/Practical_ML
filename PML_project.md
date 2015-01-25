# Practical Machine Learning - Project Writeup
##Summary
The following analysis uses data from individuals performing weight lifting exercises while wearing monitoring devices (accelerometers). These devices gathered information on how exactly these individuals completed the exercises, whether correctly or incorrectly. This analysis will attempt to classify each data record into the appropriate set {A, B, C, D, E} which corresponds to a correct or incorrect style of completing the weight lifting exercise.

##Data Cleansing and Preparation
The training data consists of 19,622 observations of 160 variables, and the testing set picks only 20 observations of the same variables. 


```r
library(caret)
library(randomForest)
library(gbm)

pmltrain <- read.csv("pml-training.csv", na.strings = c("NA", ""))
pmltest <- read.csv("pml-testing.csv", na.strings = c("NA", ""))
```

As the number of variables is very large, the results of my exploratory data analysis will be summarized rather than presented in detail. The most obvious problem is that there a large number of "NA" and "DIV/0" results in the data. These must addressed. It is important to note that the raw data was recorded over a period of time, and has been cut into a number of "windows" (information about these windows is recorded in the "New Window" and "Num Window" variables). After visually inspecting the raw data it became apparent that the NA and DIV/0 errors were confined to only some of the variables. Namely the variables that were summarizing the results of a given "window" -- the errors were confined to variables with names including: "var", "avg", "skewness", "kurtosis", "amplitude", "max", "min", and "stddev". These are all summary statistics (variance, average, etc) of groups of individual observations, and since I am interested in classifying individual records, summary data about groups of records is not helpful and can be removed from the data set.  
  
The first seven columns in the data set can also be removed as they are not useful to the classification process. For example, I assume the timestamp information does not impact what exercise group an individual was assigned. 


```r
cleantrain <- pmltrain[, grep("var_|kurtosis_|avg_|skewness_|max_|min_|amplitude_|stddev_", 
                              names(pmltrain), value = TRUE, invert = TRUE)]
cleantest <- pmltest[, grep("var_|kurtosis_|avg_|skewness_|max_|min_|amplitude_|stddev_", 
                              names(pmltest), value = TRUE, invert = TRUE)]

##also need to remove the first seven columns of data - not relevant to the analysis at hand
cleantrain <- cleantrain[,-c(1:7)]
cleantest <- cleantest[,-c(1:7)]
```

After cleaning the data sets, there are 19,622 observations of 53 variables in the training set and 20 observations of the same variables in the test set.  

Using Principal Components Analysis we can get a visual sense of what the data may be able to tell us:

```r
preProc <- preProcess(cleantrain[,-53], method = "pca", thres = 0.8)
trainPC <- predict(preProc, cleantrain[-53])
##summary(trainPC)
plot(trainPC[,1],trainPC[,2])
```

![](PML_project_files/figure-html/unnamed-chunk-3-1.png) 

There are clearly five groups of observations in the data. Two of the groups may prove difficult to tell apart.  

However, PCA will be difficult to use for prediction because there are five categories to predict (linear regression models cannot be used), so a tree model approach should be appropriate. I will compare the Random Forest and GBM models.  



```r
rfmodFit <- randomForest(x = cleantrain[,-53], y = cleantrain[,53], xtest = cleantest[,-53], importance = TRUE)
importance_vector <- as.data.frame(importance(rfmodFit, type = 2))
importance_vector
```

```
##                      MeanDecreaseGini
## roll_belt                  1255.09605
## pitch_belt                  707.39975
## yaw_belt                    897.53407
## total_accel_belt            225.80518
## gyros_belt_x                 93.01541
## gyros_belt_y                112.24621
## gyros_belt_z                319.77889
## accel_belt_x                111.56469
## accel_belt_y                139.48117
## accel_belt_z                419.62427
## magnet_belt_x               248.22810
## magnet_belt_y               368.37049
## magnet_belt_z               408.73765
## roll_arm                    317.61174
## pitch_arm                   172.30878
## yaw_arm                     250.88328
## total_accel_arm             105.73173
## gyros_arm_x                 133.65002
## gyros_arm_y                 137.64830
## gyros_arm_z                  55.63409
## accel_arm_x                 227.89477
## accel_arm_y                 152.14726
## accel_arm_z                 125.76857
## magnet_arm_x                260.08539
## magnet_arm_y                212.03172
## magnet_arm_z                195.12389
## roll_dumbbell               412.73868
## pitch_dumbbell              182.07477
## yaw_dumbbell                247.22531
## total_accel_dumbbell        255.98159
## gyros_dumbbell_x            124.72560
## gyros_dumbbell_y            245.81455
## gyros_dumbbell_z             84.19384
## accel_dumbbell_x            246.44343
## accel_dumbbell_y            401.29250
## accel_dumbbell_z            335.15180
## magnet_dumbbell_x           474.47255
## magnet_dumbbell_y           666.23623
## magnet_dumbbell_z           753.11052
## roll_forearm                618.55864
## pitch_forearm               806.47674
## yaw_forearm                 168.91655
## total_accel_forearm         114.16966
## gyros_forearm_x              74.42575
## gyros_forearm_y             125.91359
## gyros_forearm_z              81.91971
## accel_forearm_x             320.36751
## accel_forearm_y             145.01081
## accel_forearm_z             247.64567
## magnet_forearm_x            224.23383
## magnet_forearm_y            230.33901
## magnet_forearm_z            271.95946
```
  
The variables "roll_belt" and "yaw_belt" provide the most explantory power to this model.  

##What is predicted?


```r
rfmodFit$confusion
```

```
##      A    B    C    D    E  class.error
## A 5579    1    0    0    0 0.0001792115
## B   10 3784    3    0    0 0.0034237556
## C    0   11 3409    2    0 0.0037989480
## D    0    0   21 3193    2 0.0071517413
## E    0    0    0    7 3600 0.0019406709
```

```r
##give matrix of predictions
rfmodFit$test$votes
```

```
##        A     B     C     D     E
## 1  0.046 0.830 0.092 0.024 0.008
## 2  0.958 0.020 0.014 0.002 0.006
## 3  0.114 0.744 0.082 0.010 0.050
## 4  0.952 0.004 0.020 0.024 0.000
## 5  0.972 0.006 0.016 0.002 0.004
## 6  0.008 0.072 0.082 0.018 0.820
## 7  0.014 0.006 0.060 0.912 0.008
## 8  0.050 0.760 0.052 0.092 0.046
## 9  0.998 0.002 0.000 0.000 0.000
## 10 0.990 0.010 0.000 0.000 0.000
## 11 0.048 0.820 0.072 0.038 0.022
## 12 0.014 0.040 0.874 0.018 0.054
## 13 0.000 0.994 0.002 0.004 0.000
## 14 1.000 0.000 0.000 0.000 0.000
## 15 0.006 0.014 0.020 0.016 0.944
## 16 0.010 0.024 0.002 0.008 0.956
## 17 0.984 0.000 0.000 0.000 0.016
## 18 0.048 0.900 0.004 0.042 0.006
## 19 0.066 0.908 0.012 0.010 0.004
## 20 0.000 1.000 0.000 0.000 0.000
## attr(,"class")
## [1] "matrix" "votes"
```

```r
##manually pick predictions to submit to course website
##largest value in the votes matrix is the prediction
answers <- c("B","A","B","A","A","E","D","B","A","A","B","C","B","A","E","E","A","B","B","B")

## gbm - boosting - method ## ran once, and then commented out code to speed up report generation
##gbmFit <- train(classe ~ ., method = "gbm", data = cleantrain, verbose = FALSE)
##predict(gbmFit, cleantest[,-53])
## this method requires a lot of memory 

##pml_write_files = function(x){
##  n = length(x)
##  for(i in 1:n){
##    filename = paste0("problem_id_",i,".txt")
##    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
##  }
##}
##pml_write_files(answers)
```
##Results and Cross-Validation
The random forest model predicts these 20 results: "B","A","B","A","A","E","D","B","A","A","B","C","B","A","E","E","A","B","B","B".  

The GBM model also predicts the exact same results. Given the two models give the same results I expect the out of sample error to be quite low. The in-sample error from the random forest confusion matrix supports that expectation. The largest error shown is for classifying "D" and it is approximately 0.007 - this roughly equates to one classification error in 140 samples.  However, it is appropriate to also measure the out of sample error to obtain a less optimistic (or, more realistic) view of the limitations of the model. To do this, the training data set will be split into two parts - a new testing set will be created via random sampling, and the (reduced) training set will have those items removed. As the random forest model I use is very computationally intensive I am choosing to run only two cross-validation tests to conserve time.



```r
##out of sample cross validation
set.seed(444) ## to ensure consistent selection of the test set
newtest1 <- cleantrain[sample(dim(cleantrain)[1], size = 100),]
newtrain1 <- cleantrain[-sample(dim(cleantrain)[1], size = 100),]

rfmodFit1 <- randomForest(x = newtrain1[,-53], y = newtrain1[,53], xtest = newtest1[,-53], 
                         ytest = newtest1[,53], importance = TRUE)

set.seed(1001) ## to ensure consistent selection of the test set
newtest2 <- cleantrain[sample(dim(cleantrain)[1], size = 100),]
newtrain2 <- cleantrain[-sample(dim(cleantrain)[1], size = 100),]

rfmodFit2 <- randomForest(x = newtrain2[,-53], y = newtrain2[,53], xtest = newtest2[,-53],
                          ytest = newtest2[,53], importance = TRUE)

rfmodFit1$confusion
```

```
##      A    B    C    D    E  class.error
## A 5554    1    0    0    1 0.0003599712
## B    8 3759    3    0    0 0.0029177719
## C    0    9 3396    3    0 0.0035211268
## D    0    0   19 3182    2 0.0065563534
## E    0    0    2    3 3580 0.0013947001
```

```r
rfmodFit2$confusion
```

```
##      A    B    C    D    E class.error
## A 5542    2    0    0    1 0.000541028
## B   13 3766    4    0    0 0.004493788
## C    0   13 3391    1    0 0.004111601
## D    0    0   21 3179    2 0.007183011
## E    0    0    1    3 3583 0.001115138
```

Both out of sample errors are similar to those found for the in-sample errors. As the data set is quite large, and the groupings of the category variables are quite distinct, it is likely that to see material out of sample error we would need to reduce the size of the training set rather dramatically.  

##Conclusion
Both the Random Forest and GBM models predict exactly the same classifications for the testing set data. Those results were submitted to the Coursera website for validation. Given the low expected (in-sample and out-of-sample) classification error, it is expected that at most only one of the results will be incorrect (though even one error should be unlikely). The final tally indeed was 100% accuracy. We would expect to see some errors if a larger testing set were used.
