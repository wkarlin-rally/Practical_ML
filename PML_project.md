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
## roll_belt                  1267.74111
## pitch_belt                  693.12484
## yaw_belt                    939.27211
## total_accel_belt            232.20237
## gyros_belt_x                 96.24973
## gyros_belt_y                104.64880
## gyros_belt_z                305.17805
## accel_belt_x                115.60598
## accel_belt_y                119.36808
## accel_belt_z                422.87383
## magnet_belt_x               248.91804
## magnet_belt_y               366.47437
## magnet_belt_z               389.07780
## roll_arm                    330.68151
## pitch_arm                   182.51521
## yaw_arm                     230.61046
## total_accel_arm             100.21647
## gyros_arm_x                 129.56205
## gyros_arm_y                 135.38979
## gyros_arm_z                  56.82930
## accel_arm_x                 236.74633
## accel_arm_y                 151.29424
## accel_arm_z                 127.56870
## magnet_arm_x                257.97861
## magnet_arm_y                241.83432
## magnet_arm_z                189.37076
## roll_dumbbell               409.24906
## pitch_dumbbell              173.25811
## yaw_dumbbell                251.02629
## total_accel_dumbbell        267.56374
## gyros_dumbbell_x            120.67173
## gyros_dumbbell_y            240.09872
## gyros_dumbbell_z             84.92547
## accel_dumbbell_x            252.12556
## accel_dumbbell_y            413.64134
## accel_dumbbell_z            333.36281
## magnet_dumbbell_x           477.63964
## magnet_dumbbell_y           661.57974
## magnet_dumbbell_z           753.27508
## roll_forearm                623.48620
## pitch_forearm               761.50067
## yaw_forearm                 174.41184
## total_accel_forearm         105.13032
## gyros_forearm_x              70.38335
## gyros_forearm_y             123.32932
## gyros_forearm_z              80.95114
## accel_forearm_x             321.05672
## accel_forearm_y             146.75451
## accel_forearm_z             245.48882
## magnet_forearm_x            233.77581
## magnet_forearm_y            229.15041
## magnet_forearm_z            287.16176
```
  
The variables "roll_belt" and "yaw_belt" provide the most explantory power to this model.  

##What is predicted?


```r
rfmodFit$confusion
```

```
##      A    B    C    D    E  class.error
## A 5577    3    0    0    0 0.0005376344
## B    9 3785    3    0    0 0.0031603898
## C    0   11 3410    1    0 0.0035067212
## D    0    0   24 3190    2 0.0080845771
## E    0    0    2    4 3601 0.0016634322
```

```r
##give matrix of predictions
rfmodFit$test$votes
```

```
##        A     B     C     D     E
## 1  0.040 0.878 0.066 0.006 0.010
## 2  0.942 0.040 0.012 0.004 0.002
## 3  0.108 0.760 0.080 0.006 0.046
## 4  0.966 0.004 0.016 0.012 0.002
## 5  0.964 0.008 0.028 0.000 0.000
## 6  0.008 0.072 0.082 0.024 0.814
## 7  0.010 0.004 0.052 0.900 0.034
## 8  0.064 0.760 0.068 0.082 0.026
## 9  1.000 0.000 0.000 0.000 0.000
## 10 0.988 0.008 0.002 0.002 0.000
## 11 0.050 0.818 0.068 0.042 0.022
## 12 0.014 0.048 0.892 0.012 0.034
## 13 0.002 0.984 0.002 0.002 0.010
## 14 1.000 0.000 0.000 0.000 0.000
## 15 0.016 0.012 0.006 0.012 0.954
## 16 0.016 0.038 0.004 0.010 0.932
## 17 0.986 0.000 0.000 0.000 0.014
## 18 0.034 0.880 0.016 0.060 0.010
## 19 0.072 0.910 0.010 0.006 0.002
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

The GBM model also predicts the exact same results. Given the two models give the same results I expect the out of sample error to be quite low. The **cross-validation** result from the random forest confusion matrix supports that expectation. The largest error shown is for classifying "D" and it is only 0.00746 - this roughly equates to one classification error in 134 samples.

##Conclusion
Both the Random Forest and GBM models predict exactly the same classifications for the testing set data. Those results were submitted to the Coursera website for validation. Given the low expected classification error, it is expected that at most only one of the results will be incorrect (though even one error should be unlikely). The final tally indeed was 100% accuracy. We would expect to see errors if a larger testing set were used.
