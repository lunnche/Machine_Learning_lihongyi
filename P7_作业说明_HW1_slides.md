# 作业说明 HW1 slides  
## Machine Learning HW1
## COCID-19 Cases Prediction  

## Objectives  
* Solve a regression problem with deep neural networks(DNN).
* Understand basic DNN training tips
e.g. hyper-parameter tuning, feature selection,regularization,...
* Get familiar with PyTorch.  

## Task Description  
* COVID-19 Cases Prediction
* Source:Delphi group @ CMU
    * A daily survey since April 2020 via facebook.

<font color="red">Do not attempt to find any related data!Using additional data is prohibited and your final grade $\times$ 0.9</font>

![image-20220201192335608](https://raw.githubusercontent.com/lunnche/picgo-image/main/image-20220201192335608.png)

* Given survey results in the past 3 days in a specific state in U.S.,then predict the percentage of new tested positive cases in the 3rd day.  

![image-20220201192615867](https://raw.githubusercontent.com/lunnche/picgo-image/main/image-20220201192615867.png)

## Data -- Delphi's COVID-19 Surveys  

![image-20220201193053868](https://raw.githubusercontent.com/lunnche/picgo-image/main/image-20220201193053868.png)

![image-20220201193159288](https://raw.githubusercontent.com/lunnche/picgo-image/main/image-20220201193159288.png)

* States (40,encoded to one-hot vectors)
    * e.g. AL,AK,AZ,...
* COVID-like illness(4)
    * e.g. cli,ili(influenza-like illness),...
* Behavior Indicators(8)
    * e.g. wearing_mask,travel_outside_state,...
* Mental Health Indicators(5)
    * e.g. anxious,depressed,...
* Tested Positive Cases(1)
    * <font color="red">tested_positive(this is what we want to predict)</font>  
    上面涉及的数字都是百分比  

## Data -- One-hot Vector  
* One-hot vectors:
Vectors with only one element equals to one while others are zero.
Usually used to encode discrete values.  

![image-20220201195010005](https://raw.githubusercontent.com/lunnche/picgo-image/main/image-20220201195010005.png)

## Data -- Training  

covid.train.csv (2700 samples)  

![image-20220201195231969](https://raw.githubusercontent.com/lunnche/picgo-image/main/image-20220201195231969.png)

![image-20220201195330004](https://raw.githubusercontent.com/lunnche/picgo-image/main/image-20220201195330004.png)

## Evaluation Metric  
* Root Mean Squared Error (RMSE)
$$
RMSE=\sqrt{\frac{1}{N}\sum_{n=1}^{N}(f(X^n)-{\hat{y}}^n)^2}
$$

![image-20220201200112769](https://raw.githubusercontent.com/lunnche/picgo-image/main/image-20220201200112769.png)

## Kaggle  
* Link:https://www.kaggle.com/c/ml2021spring-hw1  

* Displayed name:<student ID>_<anything>
    * e.g. b06901020_puipui
    * For auditing, don't put student ID in your displayed name.  

* Submission format: .csv file  
    * See sample code  

![image-20220201200517940](https://raw.githubusercontent.com/lunnche/picgo-image/main/image-20220201200517940.png)

## Kaggle -- Submission  
* You may submit up to 5 results each day (UTC).
* Up to 2 submissions will be considered for the private leaderboard.  

![image-20220201200758945](https://raw.githubusercontent.com/lunnche/picgo-image/main/image-20220201200758945.png)

## Grading 
到10：28
