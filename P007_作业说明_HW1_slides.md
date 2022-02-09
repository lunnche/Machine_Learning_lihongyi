# 作业说明 HW1 slides  
## Machine Learning HW1
## COVID-19 Cases Prediction  

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

![image-20220204193413326](https://raw.githubusercontent.com/lunnche/picgo-image/main/image-20220204193413326.png)

## Grading -- Kaggle  
* We might change the strong baseline if it's too hard.  

![image-20220204193541400](https://raw.githubusercontent.com/lunnche/picgo-image/main/image-20220204193541400.png)

## Grading -- Bonus  
* If you got 10 points, we make your code public to the whole class.  

* In this case, if you also submit **a PDF report briefly describing your methods**  less than 100 words in English,you get a bonus of 0.5 pt.  
(your report will also be available to all students)  

report template 

![image-20220204194142067](https://raw.githubusercontent.com/lunnche/picgo-image/main/image-20220204194142067.png)

## Code Submission  
* NTU COOL (4pts)  
    * Compress your code and report into `<student ID>_hw1.zip`

    * We can only see your last submission.
    * Do not submit your model or dataset.
    * If your code is not reasonable,your semester grade $\times$ 0.9.
    * You must specify the source of your code.  
    * E.g., add a **Reference** block at the bottom of your code.  

Reference格式
```
Reference

Source:Heng-Jui Chang @ NTUEE(https://github.com/ga642381/ML2021-Spring/blob/main/HW01/HW01.ipynb)
```

* Your .zip file should include only
    * Code: either .py or .ipynb
    * Report: .pdf (only for those who get 10 points)  

* Example:

![image-20220204194953695](https://raw.githubusercontent.com/lunnche/picgo-image/main/image-20220204194953695.png)

* How to download your code from Google Colab?  

![image-20220204195121001](https://raw.githubusercontent.com/lunnche/picgo-image/main/image-20220204195121001.png)

* How to compress your folder?
* Method 3 (command line)
```
zip -r <name>.zip <directory name>
e.g.
zip -r b06901020_hw1.zip b06901020_hw1
```

## Deadlines
* Kaggle 
2021/03/26 23:59(UTC+8)
* Code Submission (NTU COOL)
2021/03/28 23:59(UTC+8)

## Hints
* Simple Baseline
    * Sample code
* Medium Baseline
    * Feature selection: 40 states + 2 tested_positive
    (will be demonstrated in class)
* Strong Baseline
    * Feature selection (what other features are useful?)
    * DNN architecture (layers?dimension?activation function?)
    * Training (mini-batch?optimizer?learning rate?)
    * L2 regularization
    * There are some mistakes in the sample code, can you find them?

## Regulations Again  
* You should finish your homework on your own.  
* You should not modify your prediction files manually.
* Do not share codes or prediction files with any living creatures.
* Do not use any approaches to submit your results more than 5 times a day.
* Do not search or use additional data or pre-trained models.
* Your final grade x 0.9 if you violate any of the above rules.
* Prof. Lee & TAs preserve the rights to change the rules & grades.  

## If any questions, you can ask us via...
* NTU COOL (recommended)
    * https://cool.ntu.edu.tw/courses/4793
* Email
    * ntu-ml-2021spring-ta@googlegroups.com
    * The title should begin with "[hw1]"
* TA hour
    * Each Friday during class  

## Useful Links  
* Hung-yi Lee, Regression & Gradient Descent (Mandarin)
* Hung-yi Lee, Tips for Training Deep Networks(Mandarin)
* Google Machine Learning Crash Course (English)
    * (Regularization,NN Trainning)
* https://pytorch.org/docs/stable/index.html
* https://www.google.com/  

(If Google or Stackoverflow can answer your questions, you may take advantage of them before asking the TAs.)  

![image-20220204203430531](https://raw.githubusercontent.com/lunnche/picgo-image/main/image-20220204203430531.png)

完结撒花🌺
