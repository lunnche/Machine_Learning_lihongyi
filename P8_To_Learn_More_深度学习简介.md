# vTo Learn More - 深度学习简介  

该P8了  

## Deep learning  

![image-20220207201345839](https://raw.githubusercontent.com/lunnche/picgo-image/main/image-20220207201345839.png)

横轴时间，纵轴Google内部用到深度学习的项目数量  

## Ups and downs of Deep Learning  
* 1958:Perceptron(linear model)  
* 1969:Perceptron has limitation
* 1980s:Multi-layer perceptron (neural network)
    * Do not have significant difference from DNN today
* 1986: Backpropagation
    * Usually more than 3 hidden layers is not helpful
* 1989: 1 hidden layer is "good enough",why deep?  
* 2006: RBM initialization (breakthrough)   (Restricted Boltzmann Machine)

    做gradient descent，如果是用RBM找的初始值，叫做deep learning，没用RBM找，那就是1980年代的Multi-layer perceptron  

RBM（受限玻尔兹曼机）非常复杂，大概要讲三周的课才能听懂。  它不是neural network base的方法,它是graphical model，

RBM又复杂，又没用？现在已经不太有人用RBM做initialization了  

**RBM**就是石头汤里的石头，没什么鸟用，但因为很复杂，吸引了很多人来搞deep learning  

* 2009 GPU

过去训练一次，一周就过去了，结果不好，就没人想做下去了，有了GPU本来要一周的东西，只需几个小时。

* 2011： Start to be popular in speech recognition  
* 2012: win ILSVRC image competition  

![image-20220208101746826](https://raw.githubusercontent.com/lunnche/picgo-image/main/image-20220208101746826.png)



![image-20220208102110572](https://raw.githubusercontent.com/lunnche/picgo-image/main/image-20220208102110572.png)

怎么把这些神经元连接起来其实是手动设计的，最常见的方法：Fully Connct Feedforward Network  

![image-20220208102922675](https://raw.githubusercontent.com/lunnche/picgo-image/main/image-20220208102922675.png)

![image-20220208103201223](https://raw.githubusercontent.com/lunnche/picgo-image/main/image-20220208103201223.png)
如果一个神经网络，w和b等参数都确定了，那就可以把它看成一个函数，输入一个向量，得到一个向量  

如果还不知道w和b等参数，我只是把神经网络怎么连接，它的结构是什么定好了，这样实际上是定了一个function set.  

Given network structure,define a function set .  

概括下fully connect feedforward network 的结构：

![image-20220208103948713](https://raw.githubusercontent.com/lunnche/picgo-image/main/image-20220208103948713.png)

Residual Net 不是一般的Fully connected feedforward network,152层你拿一般的fully connected feedforward network 来跑，不是overfitting的问题，是连train 都 train 不起来，要有特殊的structure，才能搞定这么深的network

![image-20220208104952640](https://raw.githubusercontent.com/lunnche/picgo-image/main/image-20220208104952640.png)

network的运作，我们常常会用matrix operation来表示，

![image-20220208105351756](https://raw.githubusercontent.com/lunnche/picgo-image/main/image-20220208105351756.png)



![image-20220208105617380](https://raw.githubusercontent.com/lunnche/picgo-image/main/image-20220208105617380.png)

![image-20220208105841393](https://raw.githubusercontent.com/lunnche/picgo-image/main/image-20220208105841393.png)

所以，为啥gpu能加速训练，就是因为gpu做矩阵运算快  

![image-20220208110152288](https://raw.githubusercontent.com/lunnche/picgo-image/main/image-20220208110152288.png)

所以呢，可以把隐层看成是帮你在做特征工程，feature extractor replacing feature engineering.
输出层呢可以看成是一个Multi-class classifier  

一个栗子🌰  

手写一个数字，让机器识别  

输入呢，把图像分解成像素，用向量来表示，
输出呢，如果你用softmax，那就是可能是数字1的几率，可能是数字2的几率。。。。

![image-20220208110724101](https://raw.githubusercontent.com/lunnche/picgo-image/main/image-20220208110724101.png)

![image-20220208111057611](https://raw.githubusercontent.com/lunnche/picgo-image/main/image-20220208111057611.png)

![image-20220208111428334](https://raw.githubusercontent.com/lunnche/picgo-image/main/image-20220208111428334.png)

找一个好的network structure 还是你比较难的， 有时候甚至需要一些domain knowledge
从非deep learning 的方法到 deep learning，machine learning并没有变得简单，只是我们把一个问题转化为另一个问题：  
本来呢，不是deep的model，我们要得到好的结果，我们往往要做特征工程，找一组好的feature，

而做deep learning的时候，你并不需要找一个好的feature，
比如以前你做影像辨识的时候，你要挑一些feature，有了deep learning，你可以直接把pixel丢进去硬做，嗯，就是要硬做，  

但是呢，deep learning 又制造了一个新的问题：你需要去design这个network structure，
所以问题从如何抽featrure转化成如何design network的结构  

所以deep learning 是否对你的问题来说是一个好的方案，取决于上述两项工作哪个来的容易  

对于语音辨识、影响辨识的话，design network structure 比 feature engineering 容易  

因为识别图像、语音这件事虽然人会干，但太过潜意识了，我们无法明确算法化我们是如何做到语音图像辨识这件事情的，想要让人来抽一组好的feature，很难，人根本不知道好的feature是什么。倒不如尝试各种network structure，让machine自己去找出好的feature。  

语音辨识和影像辨识这两个community是最早使用deep learning的  一用下去，进步就非常惊人。比如说辨识的错误率下降了20%，

比如，一种说法，deep learning 在nlp上效果没有那么好，进步不明显，不那么work，原因猜测是人对于文字处理这件事是比较强的，比如让你设计一个算法判断一个document是正面情绪还是负面情绪，我就可以列表，正面情绪词汇多少，负面多少，就可以得出比较好的结果，  

那么network structure 能不能机器自己学出来？可以，只是目前没有普及，你看到的那些非常惊人的应用，比如alphago都不是用这种方法学出来的，  

那么我们能不能自己涉及network structure，不要fully connected ?自己乱接？可以  
比如一种特殊的接法就是Convolutional Neural Network(CNN),

![image-20220208141636962](https://raw.githubusercontent.com/lunnche/picgo-image/main/image-20220208141636962.png)

## 定义一个function的好坏 Loss for an Example   

一般可以用MAE、MSE来表示label和taget的差距，这个例子里涉及概率，所以用Cross Entropy  
$$
C(y,\hat{y})
=
- \sum_{i=1}^{10} \hat{y_i} \ln{y_i}
$$

![image-20220208142252907](https://raw.githubusercontent.com/lunnche/picgo-image/main/image-20220208142252907.png)

## Total Loss  

![image-20220208143956541](https://raw.githubusercontent.com/lunnche/picgo-image/main/image-20220208143956541.png)

怎么解Total Loss?用的方法就是梯度下降  
## Gradient Descent  

深度学习里做梯度下降和线性回归里的没什么大差别，只是function复杂了一点：  
$\theta$里面是一大堆的weight和bias， 先每个参数random找一个初始值，接下来计算每个参数的gradient，即计算每个参数对Total Loss的偏微分，  
gradient:
$$
\nabla L =
\left[
\begin{matrix}
\frac{\partial L}{\partial{w_1}}\\
\frac{\partial L}{\partial{w_2}}\\
\vdots\\
\frac{\partial L}{\partial{b_1}}\\
\vdots
\end{matrix}
\right]

$$
