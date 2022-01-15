## Machine Learning

machine learning ≈ looking for function

* Speech Recognition

![Screen Shot 2021-12-20 at 2.11.16 PM](https://raw.githubusercontent.com/lunnche/picgo-image/main/Screen%20Shot%202021-12-20%20at%202.11.16%20PM.png)

* Image Recognition

![Screen Shot 2021-12-20 at 2.11.42 PM](https://raw.githubusercontent.com/lunnche/picgo-image/main/Screen%20Shot%202021-12-20%20at%202.11.42%20PM.png)

* Playing Go

![Screen Shot 2021-12-20 at 2.12.08 PM](https://raw.githubusercontent.com/lunnche/picgo-image/main/Screen%20Shot%202021-12-20%20at%202.12.08%20PM.png)

### Different types of Functions
Regression:The function outputs a scalar 用于连续值   

Classification:Given options(classes),the function outputs the correct one 用于离散值  

alphaGo其实也是分类问题 输入棋局，输出从19×19个分类（位置）中选一个

除了regression和classification，还有strctured learning,即机器学习除了分类，输出一个数值之外，还可以创造一个有结构的物件。
例如 画图，写文章 这种让机器产生有结构的东西的问题就叫structured learning
,即让机器学会创造。

一个例子 youtube过去的数据，预测明天的点阅总人数  

分三个步骤：

## 1
猜测一个含有未知参数的函数 （待定系数法） 这个猜测是来源于对问题本质上的了解（domain knowledge）  
$$
    y=b+wx_1
$$

y是要预测的未来的点阅量    x是已知的点阅量
    b和w是未知的参数，要从数据中学习得来
带有未知参数的function就叫做model

$x_1$是已知的youtube后台的资讯，叫做feature,w叫做weight，b叫做bias

## 2 
Define Loss from Training Data

Loss is a function of parameters  L(b,w)
Loss : how good a set of values is .

真实值，正确的数值叫label   怎么计算Loss？给定一组（b,w），根据训练数据，计算所有预测值和真实值的偏差的总和求平均

$$
Loss:L=\frac{1}{N}\sum_{n}{e_n}
$$
L越大，代表这一组参数越不好，e就是计算估算值和实测值之间的差距:  

***

$$
e=\left|y-\widehat{y}\right|        
$$
<center>L is mean absolute error (MAE)</center>  

***

$$
e=(y-\widehat{y})^2
$$
<center>L is mean square error (MSE)</center>
***

MSE和MAE有微妙的差别，具体选哪个应根据具体问题分析，这里选MAE
MSE也可以，作业里用MSE


有一些任务：
if y and $\widehat{y}$ are both probability distributions 概率分布
此时可能会选择Cross-entropy

![Screen Shot 2021-12-24 at 9.10.01 AM](https://raw.githubusercontent.com/lunnche/picgo-image/main/Screen%20Shot%202021-12-24%20at%209.10.01%20AM.png)
<center>上图为Loss函数图像，横坐标w，纵坐标b</center>
这种等高线图叫error surface

## 3
Optimization  

$$
w^*,b^* = \mathop{\arg\min}\limits_{w,b} L
$$

这个最优化问题可表述为：找一个最优的w,b值组，可以让L值最小，这个具体的w,b值，记为$w^*$,$b^*$
要怎么找出这个$w^*$,$b^*$值呢，用Gradient Descent方法 梯度下降

这个Gradient Descent方法具体怎么做呢，为表述方便，先假设只有一个未知参数w：
$$
w^*=\mathop{\arg\min}\limits_{w}L
$$

![Screen Shot 2021-12-26 at 6.03.42 PM](https://raw.githubusercontent.com/lunnche/picgo-image/main/Screen%20Shot%202021-12-26%20at%206.03.42%20PM.png)
这条曲线就是error surface的二维版


怎么找到这个让L值最小的w呢，即Gradient Descent的步骤：

**1 (Randomly)Pick an initial value**  
   $w^0$（有一些方法可以给我们一个比较好的初始值，但这里先整个随机的）

**2 compute**
   $\frac{\partial{L}}{\partial{w}}|_{w=w^0}$   

   计算w=$w^0$时，w这个参数对Loss的微分,即计算在$w_0$这一点，error surface切线的斜率，若切线的斜率是负的，即左边高右边低，那就把w值变大，就可以让Loss变小。
    反之减小w。
    增加或减小w，步幅多大呢，取决于两件事，1 斜率 2 $\eta$
    $$
    w^1\leftarrow w^0-\eta\frac{\partial{L}}{\partial{w}}|_{w=w^0}
    $$

$\eta$: learning rate
learning rate是自己设定的，像learning rate这种在机器学习中需要自己设定的东西叫做hyperparameters

之前说做机器学习第一步中包含未知参数，这些参数是机器自己找出来的  

为什么Loss可以是负的，Loss是估测的值和正确的值之差的绝对值，按理不可能是负，但是Loss这个function是你自己决定的，比如我定义一个Loss=绝对值-100，上图的Loss只是随便瞎画的，没有任何实际意义，若按一般问题中Loss为绝对值的定义，则不可能出现负值  

**3 Update w iteratively** 什么时候停下来呢 两种情况 1 你失去耐心 设定参数更新多少次后停下，这个多少次也是个hyperparameter，2 微分那一项算出来是0

因此，gradient descent方法有一个巨大的问题，那就是我们没有找到那个最好的解，梯度下降找到的是局部最优(local minima)，而非全局最优(global minima)

![Screen Shot 2021-12-26 at 6.24.01 PM](https://raw.githubusercontent.com/lunnche/picgo-image/main/Screen%20Shot%202021-12-26%20at%206.24.01%20PM.png)

所以有人说gradient descent不是个好方法，这个方法有local minima的问题，没有办法真的找到global minima。但这其实只是幻觉而已，假设你有做过深度学习相关的事情，假设你有自己训练network，自己做过Gradient Descent经验的话，其实local minima是一个假问题，我们在做Gradient Descent的时候，真正面对的难题不是local minima，到底是什么，之后会讲。

## 有两个参数的情况

$$
w^*,b^* = \mathop{\arg\min}_{w,b} L
$$

* (Randomly)Pick initial values $w^0,b^0$  
* Compute  
$$
\frac{\partial{L}}{\partial{w}}|_{w=w^0,b=b^0}
$$
$$
\frac{\partial{L}}{\partial{b}}|_{w=w^0,b=b^0}
$$

分别计算w对Loss的微分，b对Loss的微分  

$$
w^1 \leftarrow w^0 - \eta \frac{\partial{L}}{\partial{w}}|_{w=w^0,b=b^0}
$$
$$
b^1 \leftarrow b^0 - \eta \frac{\partial{L}}{\partial{b}}|_{w=w^0,b=b^0}
$$

如上，$w^0$ 减去 learning rate 乘上 微分的结果 得到 $w^1$  

那么这个微分要怎么算呢，其实 Can be done in one line in most deep learning frameworks

* Update w and b interatively  

为什么在这个方向前面要加上负号？  
因为如果算出来的微分是负的，说明右侧的L小，因此w要增大（往右移动），加上负号位移就为正了。

![Screen Shot 2021-12-26 at 8.10.37 PM](https://raw.githubusercontent.com/lunnche/picgo-image/main/Screen%20Shot%202021-12-26%20at%208.10.37%20PM.png)

若
$$
w^*=0.97,b^*=0.1k\\
L(w^*,b^*)=0.48k
$$
说明啥：
取w为0.97，b为0.1k 效果最好，即其实x1和y是接近的，所以当weight取接近1，bias取小值时，效果较好，L表明预测和实际点阅量的平均偏差在500人左右。

总结：
Machine Learning is so simple:
* Step1: function with unkown
* Step2: define loss from training data 
    注意什么: 
    $L$ :训练数据的Loss 
    $L^{'}$ :未知数据的Loss
* Step3: optimization

然后，上例的结果是最令人满意的结果吗：
$y=0.1k+0.97x_1$ achieves the smallest loss $L=0.48k$ on data of 2017-2020(training data)

也许不是，因为上述三部统称为训练(training)，即在已知的资料上去计算loss，其实只是在自high而已,即假装自己不知道隔天的点阅量，拿求出的函数y来预测，能得到0.48k的Loss，但我们真正在意的不是过去已经知道的东西，而是未来未知的观看次数

How about data of 2021 (unseen during training)?
$$
L'=0.58k
$$

 ![Screen Shot 2021-12-29 at 3.37.40 PM](https://raw.githubusercontent.com/lunnche/picgo-image/main/Screen%20Shot%202021-12-29%20at%203.37.40%20PM.png)

可是化后如上图所示：
发现每到周五、周六就没人学机器学习，可以理解，出去浪，
为什么周日大家都学机器学习，也许和youtube的推荐机制有关，一到周五、六就不推荐给大家机器学习的视频，搞不懂
总之，发现了这种周期性的规律后，就可以改进模型。

对模型的改进往往来自于你对这个问题的理解。即domain knowledge

一开始我们对问题完全不理解的时候，就胡乱写个$y=b+wx_1$，接下来我们观察真实数据后，得到一个结论：每隔七天有个循环，所以应该把前七天的观看人次都列入考虑。写出如下模型：
$$
y=b+\sum_{j=1}^{7}{w_jx_j}
$$

上面考虑了前7天，如果考虑更多会怎样呢？
$$
y=b+\sum_{j=1}^{28}{w_jx_j}
$$

![Screen Shot 2021-12-29 at 3.49.12 PM](https://raw.githubusercontent.com/lunnche/picgo-image/main/Screen%20Shot%202021-12-29%20at%203.49.12%20PM.png)

上图为考虑不断增加天数，对Loss的影响，可以看到考虑越多的天，Loss越小，但从28天增加到56天，对于预测未来已无提高，$L'$保持为0.46k  

上述的模型都是feature乘上weight加上bias，这类模型都叫做linear models  

linear models are too simple... we need more sophisticated models.  
也许x1和y之间有复杂的关系，但linear models永远只能表征一条直线，随着x1越来越高，y就理应越来越大，改变w和b只能调整斜率和y周截距，不管你怎么摆弄w和b，永远也造不出类似红色线这种体现物极必反规律的曲线  
Linear models have severe limitation   <b> Model Bias </b>

![Screen Shot 2021-12-29 at 4.00.13 PM](https://raw.githubusercontent.com/lunnche/picgo-image/main/Screen%20Shot%202021-12-29%20at%204.00.13%20PM.png)

We need a more flexible model!  

怎么做出红色线这样的model呢

观察发现：
<font color='red'>red curve</font>= constant + sum of a set of ![Screen Shot 2021-12-29 at 4.05.06 PM](https://raw.githubusercontent.com/lunnche/picgo-image/main/Screen%20Shot%202021-12-29%20at%204.05.06%20PM.png)


红色曲线可视做常量+一群蓝色图像的结果，蓝色图像的特点是：
当x轴的值小于某一个threshhold的时候，它是一个定值，大于另一个threshhold的时候，又是另一个定值，中间有一个斜坡，

常数项constant取决于图像和y轴的交点

怎么往上加蓝色图像呢，和红线同起点，斜坡的终点设在第一个转角处，保持蓝色图像和红线的斜率一样，0线+1线可得红线在第一个转折点之前的部分  

接下来再加蓝色线2，0线+1线+2线可得红线第二个转折点之前的部分。  

同理加线3 ![Screen Shot 2021-12-29 at 4.20.31 PM](https://raw.githubusercontent.com/lunnche/picgo-image/main/Screen%20Shot%202021-12-29%20at%204.20.31%20PM.png)



![Screen Shot 2021-12-29 at 4.23.16 PM](https://raw.githubusercontent.com/lunnche/picgo-image/main/Screen%20Shot%202021-12-29%20at%204.23.16%20PM.png)

即使x和y的关系，不是Piecewise Linear的Curves，可以先在曲线上取一些点，再把这些点连起来，变成一个piecewise linear 的curves，点取得越多，取的位置越合适，就越能逼近原来的曲线。

![Screen Shot 2021-12-29 at 4.29.40 PM](https://raw.githubusercontent.com/lunnche/picgo-image/main/Screen%20Shot%202021-12-29%20at%204.29.40%20PM.png)

b站弹幕说：
多次样条插值
每个蓝色function就是一个神经元？  

这个蓝色function，它的式子，要怎么把它写出来呢？
直接写也许没那么容易，但是你可以用一条曲线来理解它，用sigmoid的function
$$
\begin{aligned}
y&=c\frac{1}{1+e^{-(b+wx_1)}} \\
 &=c\,sigmoid(b+wx_1)
\end{aligned}
$$

x趋于无穷大时，曲线会收敛在高度是c的地方  
x趋于无穷小时，曲线会趋于0  

sigmoid：硬翻成中文，就是“s型的”

可以用sigmoid去逼近蓝色function，所以蓝色function也叫做Hard Sigmoid。

![Screen Shot 2021-12-30 at 11.12.13 AM](https://raw.githubusercontent.com/lunnche/picgo-image/main/Screen%20Shot%202021-12-30%20at%2011.12.13%20AM.png)

各种各样的sigmoid，是通过调整b、w、c来得到的

![Screen Shot 2021-12-30 at 11.15.30 AM](https://raw.githubusercontent.com/lunnche/picgo-image/main/Screen%20Shot%202021-12-30%20at%2011.15.30%20AM.png)

所以，知道不同的蓝色function（用sigmoid近似），把它们叠起来，就构成了红色的Piecewise Linear Curves,就可以去逼近各式各样的不同的Continuous Function。  
所以，到此处，我们已经有能力写出一个有弹性的有未知参数的function

![Screen Shot 2021-12-30 at 11.22.07 AM](https://raw.githubusercontent.com/lunnche/picgo-image/main/Screen%20Shot%202021-12-30%20at%2011.22.07%20AM.png)

# New Model: More Features

$$
y=b+wx_1\\
\downarrow \\
y=b+\sum_{i}c_i\,sigmoid(b_i+w_ix_1)\\
\\
y=b+\sum_jw_jx_j \ \ \ \ 考虑多个feature\\
\downarrow \\
y=b+\sum_ic_i\,sigmoid(b_i+\sum_jw_{ij}x_j)
$$

<hr>

上式好复杂，来理解下：
首先假设只有三个feature,即j:1,2,3，只考虑前面三天的case
j是number of features
x1:一天前的观看人数
x2:两天前的观看人数
x3:三天前的观看人数

i是什么，每一个i代表一个蓝色function,每一个蓝色function都用一个sigmoid function来近似它。所以每一个i都代表了一个sigmoid function



$$
y = b+\sum_jc_i\,sigmoid(b_i+\sum_jw_{ij}x_j)\ \ \ j:1,2,3
$$

![Screen Shot 2021-12-30 at 2.17.10 PM](https://raw.githubusercontent.com/lunnche/picgo-image/main/Screen%20Shot%202021-12-30%20at%202.17.10%20PM.png)

那这个x1 x2 x3 和 r1 r2 r3 什么关系呢  
可以用矩阵和向量相乘的方法写一个比较简单的简洁的写法  

$$
y=b+
\sum_ic_i\,sigmoid(b_i+\sum_jw_{ij}x_j) \ \ \ \ i:1,2,3\ \ \ \ j:1,2,3
$$
括号中的
$$
b_i+\sum_jw_{ij}x_j
$$
记为r，则：
$$
r_1=b_1+w_{11}x_1+w_{12}x_2+w_{13}x_3\\
r_2=b_2+w_{21}x_1+w_{22}x_2+w_{23}x_3\\
r_3=b_3+w_{31}x_1+w_{32}x_2+w_{33}x_3\\
\downarrow\\

\left[
\begin{matrix}
r_1\\
r_2\\
r_3
\end{matrix}
\right]
=
\left[
\begin{matrix}
b_1\\
b_2\\
b_3
\end{matrix}
\right]
+\left[
\begin{matrix}
w_{11}&w_{12}&w_{13}\\
w_{21}&w_{22}&w_{23}\\
w_{31}&w_{32}&w_{33}
\end{matrix}
\right]
\left[
\begin{matrix}
x_1\\
x_2\\
x_3
\end{matrix}
\right]
\\
\downarrow\\
r=b+Wx
$$

然后这些r再通过sigmoid function 得到a
$$
a_i=\,sigmoid(r_i)=
\frac1{1+e^{-r_i}}
$$
即
$$
a=\sigma(r)\\
\downarrow\\
y=b+c^Ta
$$

![Screen Shot 2021-12-30 at 2.58.44 PM](https://raw.githubusercontent.com/lunnche/picgo-image/main/Screen%20Shot%202021-12-30%20at%202.58.44%20PM.png)

最终：
$$
y=b+c^T
\sigma(b+Wx)
$$
<font color="red">注意上面两个b不一样，左边是数值，右边是向量，不该用同一个字母</font>


来明确下各参数到底是什么：

![Screen Shot 2021-12-30 at 3.25.35 PM](https://raw.githubusercontent.com/lunnche/picgo-image/main/Screen%20Shot%202021-12-30%20at%203.25.35%20PM.png)

一些问题：

暴力搜索:系统地枚举解决方案的所有可能的候选项，以及检查每个候选项是否符合问题的描述。
未知参数只有w,b的情况，可以用暴力搜索，但随着问题规模的增大，穷举所有可能是不现实的，要用梯度下降。  

sigmoid越多，你可以产生的piecewise linear function就越复杂，
这个例子里input feature是3个，sigmoid也是3个，这只是巧合，不必相同  

hard sigmoid的function比较难写，所以用sigmoid，如果你能写出来，用hard sigmoid也可以。  

<hr>
ok,现在我们有了新的function with unkonwns
接下来处理Loss
现在Loss是theta的函数：

$$
L(\theta)
$$

给定某一组
$$
w,b,c^T,b
$$
的值

$$
y=b+c^T\sigma(b+Wx)\\
\\
\widehat{y}\ \ \ label
\\
\\
e=\left|y-\widehat{y}\right| \\
\\
\\

Loss:\ \ \ L=\frac1N\sum_ne_n
$$

<hr>

Optimization of New Model
$$
\theta^*=\mathop{\arg\min}_{\theta}L\ \ \ \ \ \ \ \ \theta=
\left[
\begin{matrix}
\theta_1\\
\theta_2\\
\theta_3\\
\vdots
\end{matrix}
\right]
$$

* (Randomly)Pick initial values  $\theta^0$


$$
g=
\left[
\begin{matrix}
\frac{\partial{L}}{\partial{\theta_1}}|_{\theta=\theta^0}\\
\frac{\partial{L}}{\partial{\theta_2}}|_{\theta=\theta^0}\\
\vdots
\end{matrix}
\right]
$$

g即gradient

$$
g=\nabla L(\theta^0)
$$

这个倒三角的意思就是，把所有参数  $\theta$ 通通拿去对L作微分
后边放  $\theta^0$ 的意思是，算微分的位置是在  $\theta$ 等于  $\theta^0$ 的地方

算出gradient 后，接下来update我们的参数

$$
\left[
\begin{matrix}
\theta_1^1\\
\theta_2^1\\
\vdots
\end{matrix}
\right]
\leftarrow
\left[
\begin{matrix}
\theta_1^0\\
\theta_2^0\\
\vdots
\end{matrix}
\right]-
\left[
\begin{matrix}
\eta\frac{\partial{L}}{\partial{\theta_1}}|_{\theta=\theta^0}\\
\eta\frac{\partial{L}}{\partial{\theta_2}}|_{\theta=\theta^0}\\
\vdots
\end{matrix}
\right]
$$

上式简写为：
$$
\theta^1 \leftarrow \theta^0 - \eta g
$$

* Compute gradient  $g=\nabla L(\theta^0)$  
$$
\theta^1 \leftarrow \theta^0 - \eta g
$$
* Compute gradient  $g=\nabla L(\theta^1)$
$$
\theta^2 \leftarrow \theta^1 - \eta g
$$
* Compute gradient  $g=\nabla L(\theta^2)$
$$
\theta^3 \leftarrow \theta^2 - \eta g
$$


照此做下去，直到不想做了，或者直到g算出来是0向量 (Zero Vector)，不过
在实际操作中几乎不太可能作出Gradient是0向量的结果，通常你停下来只是因为你不想做了。  

实际操作中，我们这样做gradient descent：
1  有N笔资料，把它分成一个个batch，怎么分？随机分就好
2  分完后每个batch里有B笔资料
3  本来是把所有的Data拿出来算一个Loss（L），但现在我们只拿出第一个batch里的data出来算一个Loss（$L^1$）
4  试想，假设这个B够大，也许L跟$L^1$很接近也说不定
5  实作中，会先选一个batch，用这个batch来算$L^1$，用这个$L^1$来算gradient:$g=\nabla L^1(\theta^0)$，用这个gradient来更新参数：
$$
\theta^1 \leftarrow \theta^0 - \eta g
$$
6  接下来，再选下一个batch算出$L^2$,根据$L^2$算出gradient $g=\nabla L^2(\theta^1)$，再更新参数：
$$
\theta^2 \leftarrow \theta^1 - \eta g
$$
7  再取下一个batch算出$L^3$，根据$L^3$算出gradient，$g=\nabla L^3(\theta^2)$, 再更新参数

$$
\theta^3 \leftarrow \theta^2 - \eta g
$$

8 所以我们不是拿L来算gradient，实际上是拿一个batch算出来的$L^1,L^2,L^3$来计算gradient

9 把所有batch都看过一次，叫做一个Epoch，每一次更新参数叫做一次Update，
    1 epoch = see all the batches once  

10 至于为什么要分一个个batch，下周再讲nice

![Screen Shot 2021-12-31 at 9.26.42 AM](https://raw.githubusercontent.com/lunnche/picgo-image/main/Screen%20Shot%202021-12-31%20at%209.26.42%20AM.png)

为了更好地区分epoch和update，这里举一个栗子🌰
Example 1:
* 10000 examples (N=10000)
* Batch size is 10 (B=10)
How many update in 1 epoch?  (epoch就如同游戏里的周目）
                1000 updates 
所以一个epoch并不是更新参数一次，上栗一个epoch更新参数1000次  

Example 2
* 1000 examples (N=1000)
* Batch size is 100 (B=100) （Batch size的大小也是人为决定的，也是hyper parameter）
<hr>

插入：所以到目前为止都有啥是hyper parameter:  <br>
learning rate $\eta$ <br>
几个sigmoid <br>
batch size <br>

<hr>

How many update in 1 epoch
                 10 updates

所以如果有人跟你说做了一个epoch的训练，你其实并不知道他更新了几次参数。取决于总样本数和batch size。

我们还可以对模型做更多的变形：

Sigmoid $\rightarrow$ ReLU

hard sigmoid 不好吗，为什么要把它换成soft sigmoid,不一定要换成soft sigmoid 还有其他选择。
比如，hard sigmoid函数有点难写出来，其实也没有那么难，它可以看作是两个Rectified Linear Unit的加合。
Rectified Linear Unit:
$$
c\,max(0,b+wx_1)
$$
两个ReLu叠加起来就可以变成hard sigmoid  

![Screen Shot 2021-12-31 at 9.56.15 AM](https://raw.githubusercontent.com/lunnche/picgo-image/main/Screen%20Shot%202021-12-31%20at%209.56.15%20AM.png)

所以，可以把sigmoid换成ReLU
$$
Sigmoid \rightarrow ReLU \\
\\
y = b + \sum_ic_i\,sigmoid(b_i + \sum_jw_{ij}x_j)\\
\\
\downarrow
\\
y = b + \sum_{2i}c_i\,max(0,b_i + \sum_jw_{ij}x_j)
$$


![Screen Shot 2021-12-31 at 10.04.07 AM](https://raw.githubusercontent.com/lunnche/picgo-image/main/Screen%20Shot%202021-12-31%20at%2010.04.07%20AM.png)

类似sigmoid ReLU 叫做 activation function 激活函数

sigmoid 和ReLU 哪个更好？
ReLU更好，后面讲  

来看看实际结果：100个ReLU就可以制造比较复杂的曲线，效果就显现出来了，但1000个ReLU改进就没有那么大了：

![Screen Shot 2021-12-31 at 10.08.54 AM](https://raw.githubusercontent.com/lunnche/picgo-image/main/Screen%20Shot%202021-12-31%20at%2010.08.54%20AM.png)

还可以怎么改进我们的模型呢，举例来说，如下图，之前说过从x到a做的事情是什么？是把x乘上w加上b，再通过sigmoid function（通过ReLU也可以）得到a  
我们可以把上面这件事反复多做几次，x通过一连串运算产生a，a通过一连串运算产生a',可以这样反复多做几次  

那么要做几次？这个做几次又是一个hyper parameter  

![Screen Shot 2021-12-31 at 10.18.50 AM](https://raw.githubusercontent.com/lunnche/picgo-image/main/Screen%20Shot%202021-12-31%20at%2010.18.50%20AM.png)

Experimental Results  
* Loss for multiple hidden layers
    * 100 ReLU for each layer
    * input features are the no. of views in the past 56 days


![Screen Shot 2021-12-31 at 10.23.10 AM](https://raw.githubusercontent.com/lunnche/picgo-image/main/Screen%20Shot%202021-12-31%20at%2010.23.10%20AM.png)

![Screen Shot 2021-12-31 at 10.28.45 AM](https://raw.githubusercontent.com/lunnche/picgo-image/main/Screen%20Shot%202021-12-31%20at%2010.28.45%20AM.png)

红圈为啥没有报出低值，那天是除夕，众所周知，过年不学机器学习，而模型不了解除夕是个什么玩意，它只懂每56天的变化规律。

好了，现在我们还缺了一个东西：a fancy name   
织席贩履之徒，说他是汉左将军宜城亭侯中山靖王之后，也就潮了起来，  
所以我们的模型也需要一个好名字  

这些sigmoid ReLU啊 叫做 Neuron 神经元，很多的Neuron 组成 Neural Networ 神经网络  

Neural Network 80 90 年代就有 被玩到 臭大街  

为了重振 Neural Network的雄风，需要起个新的名字，有很多hidden layer 就叫做Deep  $\rightarrow$ Deep Learning  

然后人们就开始把类神经网络越叠越多，越叠越深，

![Screen Shot 2021-12-31 at 10.49.24 AM](https://raw.githubusercontent.com/lunnche/picgo-image/main/Screen%20Shot%202021-12-31%20at%2010.49.24%20AM.png)

这里有个问题，用足够多的sigmoid 或者 ReLU 排一排（1 hidden layer）就可以逼近任何复杂的连续函数，搞多层，搞deep的意义何在？所以有人说deep learning 只是个噱头  
把sigmoid ReLU排一排，构造一个fat neural network不行吗，为啥要搞deep neural network?  以后再讲  

另一个问题，为何不做得更深？上例为何只搞3层，不稿100层？因为发现到4 layer的时候，已知数据Loss虽然从0.14k减少为0.10k，但未知数据的Loss从0.38k增加到0.44k  
Better on training data,worse on unseen data  
这就是 overfitting  

所以，以上就是深度学习的介绍，介绍的形式比较特殊  
想看一般形式的介绍： https://youtu.be/Dr-WRIEFefw    啥破玩意，看不了

深度学习的训练会用到一个东西叫做 backpropagation  
https://youtu.be/ibJpTrp5mcE  

p3 结束

# Backpropagation  反向传播  

如果你用Gradient descent 来训练一个神经网络，应该怎么做  

在训练神经网络时，反向传播这个算法是如何运作？从而使得neural network的training 比较有效率  

当你用Gradient Descent方法的时候，跟我们在做Logistic Regression，还有Linear Regression等等，没有太大的差别，但最大的问题是，在neural network里面，我们有非常非常多的参数，  
如果你要做语音辨识系统的话呢，你的neural network通常会有7，8层，每层1000个neuron，它有上百万个参数，所以 $\nabla L(\theta)$ 这个向量它非常非常长，上百万维的vector，  
要如何有效地去把这一个百万维的vector计算出来，这就是Backpropagation做的事情。  

所以，Backpropagation并不是一个和Gradient Descent不同的训练方法，它就是Gradient Descent,它只是一个比较有效率的演算法，让你在计算gradient那个vector时比较有效率，

![Screen Shot 2022-01-10 at 10.06.04 PM](https://raw.githubusercontent.com/lunnche/picgo-image/main/Screen%20Shot%202022-01-10%20at%2010.06.04%20PM.png)

Backpropagation里面没有高深的数学，唯一需要记住的就只有 Chain Rule  

![Screen Shot 2022-01-10 at 10.12.22 PM](https://raw.githubusercontent.com/lunnche/picgo-image/main/Screen%20Shot%202022-01-10%20at%2010.12.22%20PM.png)

![Screen Shot 2022-01-10 at 10.16.35 PM](https://raw.githubusercontent.com/lunnche/picgo-image/main/Screen%20Shot%202022-01-10%20at%2010.16.35%20PM.png)

对于这个式子，我们不用想说计算 $\frac{\partial{L}}{\partial{w}}$ 我们只考虑如何计算
对某一笔data的 $\frac{\partial{C^n(\theta)}}{\partial{w}}$ ,再summation over 所有的training data。 你就可以把total loss对某一个参数的偏微分算出来了。我们只需focus对某一笔data，它的cost $C^n$ 对 w的偏微分怎么计算这一项上面，  

怎么做呢，我们先考虑某一个neuron，这个neuron在第一个layer，所以它前面的input就是外界给它的input，x1,x2,他们分别乘上权重w1,w2,再加上b，会得到z，得到z以后通过activation function ，再经过非常非常多的事情以后，你会得到最终的output，y1,y2。  

现在问题是：假设我们从这边拿出一个w出来，拿weight当做z，来看怎么计算，某一个weight对cost,对example的某一个cost的偏微分，按照Chain Rule，$ \frac{\partial{C}}{\partial{w}} $可以把它拆成两项:  
$$
\frac{\partial{C}}{\partial{w}}=? \ \ 
\frac{\partial{z}}{\partial{w}}
\frac{\partial{C}}{\partial{z}}
$$

右边的两项，我们分别计算，第一项很简单，后面项比较复杂。  

计算前面这项，即 Compute $\frac{\partial{z}}{\partial{w}}$ for all parameters 称为Forward pass,  
计算后面这项，叫Backward pass  

计算 

$$
\frac{\partial{z}}{\partial{w_1}}=?  
$$
就是秒算，因为 $ z=x_1w_1+x_2w_2+b$
所以，一眼就知道等于x1  

那 $\frac{\partial{z}}{\partial{w_2}}$呢 x2   都是秒算  
规律就是：

![Screen Shot 2022-01-12 at 5.49.55 PM](https://raw.githubusercontent.com/lunnche/picgo-image/main/Screen%20Shot%202022-01-12%20at%205.49.55%20PM.png)

所以给你一个神经网络，计算 $\frac{\partial{z}}{\partial{w}}$ 非常简单
非常直觉，同样的process你就反复在做  
你要算出这个neural network里面的每一个weight对它的activation function的inputz 的偏微分，你就把input丢进去，然后，计算每一个neuron的output，就结束了。这个步骤叫做Forward pass.


![Screen Shot 2022-01-15 at 3.45.23 PM](https://raw.githubusercontent.com/lunnche/picgo-image/main/Screen%20Shot%202022-01-15%20at%203.45.23%20PM.png)

## Backward pass  

compute $\frac{\partial{C}}{\partial{z}}$ for all activation function inputs z  

z经过很复杂的process得到C，为求解，做下拆解  

假设activation function是sigmoid function   

$$
a=\sigma(z)
$$

z通过sigmoid funtion 得到a，这个neuron的output是a，接下来这个a乘上一个weight，再加其他一大堆value，得到z‘  ，这个z'是下一个neuron activation function的input  

a会再乘上另一个weight，这边写成w4,再加其他一大堆东西，得到z'',  

$$
\frac{\partial{C}}{\partial{z}}
=
\frac{\partial{a}}{\partial{z}}
\frac{\partial{C}}{\partial{a}}
$$

$ a=\sigma(z)$ 所以 $\frac{\partial{a}}{\partial{z}}$ = $\sigma'(z)$  
其实就是这个sigmoid function的微分 它的样子见下图  

![Screen Shot 2022-01-15 at 4.28.10 PM](https://raw.githubusercontent.com/lunnche/picgo-image/main/Screen%20Shot%202022-01-15%20at%204.28.10%20PM.png)

然后，$\frac{\partial{C}}{\partial{z}}$ 长什么样子呢：  

$$
\frac{\partial{C}}{\partial{a}}=
\frac{\partial{z'}}{\partial{a}}
\frac{\partial{C}}{\partial{z'}}+
\frac{\partial{z''}}{\partial{a}}
\frac{\partial{C}}{\partial{z''}}
\ \ \ \ (Chain\ \ rule)
$$

上式：a和C的关系是怎样的，a会影响z'和z''，z'和z''会影响C  
a透过z'和z''去影响C  

我们假设a后面，就是这个蓝色neuron的下一个layer，红色neuron只有两个，所以这边就只有两项，如果有1000个红色neuron的话，那这边chain rule里的summation就是summation over 1000项   

接下来$\frac{\partial{z''}}{\partial{a}}$ 
$$
z'=aw_3+\dots
$$

z'等于a乘上w3，再加上一些有的没的东西
这个也是秒算，就是w3  

最后的问题 z'和z''对C的偏微分怎么算  
$$
\frac{\partial{C}}{\partial{z''}}\\
\frac{\partial{C}}{\partial{z'}}
$$

我们不知道z’和z''跟C的关系，因为后边还有一长串，我们搞不清后面会发生什么事情，不过没关系，我们就假设我们知道，假设这两项的值，我们已经透过某种方法把它算出来，我们透过一个等一下会讲，但你还不知道怎么做的方法  

![Screen Shot 2022-01-15 at 4.52.02 PM](https://raw.githubusercontent.com/lunnche/picgo-image/main/Screen%20Shot%202022-01-15%20at%204.52.02%20PM.png)

假设我们算出了这两项的值，我们就可以把$\frac{\partial{C}}{\partial{z}}$轻易地算出来  

$$
\begin{aligned}
\frac{\partial{C}}{\partial{z}}&=
\frac{\partial{a}}{\partial{z}}
\frac{\partial{C}}{\partial{a}} \\
&=\sigma'(z)\left[
w_3 \frac{\partial{C}}{\partial{z'}}+
w_4 \frac{\partial{C}}{\partial{z''}}
\right]
\end{aligned}
$$



![Screen Shot 2022-01-15 at 5.03.56 PM](https://raw.githubusercontent.com/lunnche/picgo-image/main/Screen%20Shot%202022-01-15%20at%205.03.56%20PM.png)



上式可以从另一个观点来看待：
你可以想象，现在有另一个neuron，这个neuron并不在我们原来的network里面，我们把它画成三角形，这个neuron的input就是$\frac{\partial{C}}{\partial{z'}}$和$\frac{\partial{C}}{\partial{z''}}$  
第一个input乘上w3，第二个input乘上w4,两个乘积加起来的和再乘上activation function，也就是$\sigma'(z)$,得到output，就是$\frac{\partial{C}}{\partial{z}}$  

即下图中上面这个neuron所做的运算跟下面这个式子，是一模一样的，我们只是把下面这个式子画出来，让它看起来像一个neuron一样，那这个$\sigma'(z)$ 其实是一个常数，是个constant而非function，因为z其实在计算Forward pass 的时候就被决定了，z是一个已经固定的我们知道它是多少的值，所以在给定z的情况下，$\sigma'(z)$就是一个常数，  

所以这个neuron和我们之前看到的sigmoid function是不一样的，它并不是把input通过一个non-linear的转换，而是直接把input乘上一个constant，$\sigma'(z)$ ,就得到一个output，所以把这个neuron画成三角以示和圆形neuron的运作方式是不一样的，  
然后你可能会问，为什么是三角形，因为我是电机系的，我觉得这是一种op-amp这样子，op-amp就会乘上一个constant，它是一个放大器。



![Screen Shot 2022-01-15 at 5.24.40 PM](https://raw.githubusercontent.com/lunnche/picgo-image/main/Screen%20Shot%202022-01-15%20at%205.24.40%20PM.png)

那最后的问题还是回到，怎么算那两项：z'和z''对C的偏微分。  

我们假设两个不同的case  
第一个case：假设现在红色两个neuron已经是output layer 即y1和y2了，它的output就已经是这个network的output了。所以：  
$$
\frac{\partial{C}}{\partial{z'}}=
\frac{\partial{y_1}}{\partial{z'}}
\frac{\partial{C}}{\partial{y_1}}
$$

$\frac{\partial{y_1}}{\partial{z'}}$ 没问题，你只要知道activation funcation长什么样子，这项就轻而易举地算出来了，  
$\frac{\partial{C}}{\partial{y_1}}$ ，y1对C有什么影响，depend on你的cost function是怎么定义的，你的output跟target之间是怎么evaluate的，你可以用cross entropy,可以用mean square error,总之它是个比较简单的东西，你可以把它算出来。
同样的$\frac{\partial{C}}{\partial{z''}}$也没问题

所以如果蓝色的neuron在最后一个hidden layer里面，它后面就是output layer了，那根据刚才所学，就可以把w1跟w2对C的偏微分算出来

![Screen Shot 2022-01-15 at 9.54.21 PM](https://raw.githubusercontent.com/lunnche/picgo-image/main/Screen%20Shot%202022-01-15%20at%209.54.21%20PM.png)

令人烦恼的是第二个case： Not Output Layer  

假设红色neuron不是整个network的output  

后面的其他东西可能长这样：就是z'再通过activation function得到a',再乘上另一个weight w5，再加上其他一些东西，得到za，再把a'乘上w6再加上其他一大堆东西，得到zb，然后再丢进另外两个activation function里面，现在问题是我们想要求
$\frac{\partial{C}}{\partial{z'}}$ ,如果我们知道$\frac{\partial{C}}{\partial{z_a}}$跟$\frac{\partial{C}}{\partial{z_b}}$ ,我们就可以计算$\frac{\partial{C}}{\partial{z'}}$  
如图：  

![Screen Shot 2022-01-15 at 10.18.59 PM](https://raw.githubusercontent.com/lunnche/picgo-image/main/Screen%20Shot%202022-01-15%20at%2010.18.59%20PM.png)

不断递归算下去，直到output layer，这听上去挺崩溃，但实际上不是这么做的，  

实际上你只要换一个方向，从output layer往回算，你就会发现它的运算量跟原来的network的feedforward path 其实是一样的  

如图假设，有六个neuron，六个activation function分别是z1-z6,我们现在要计算这些z对C的偏微分，本来呢，我们应该是想要知道z1的偏微分，你就要算z3，z4的偏微分，加入想知道z3的偏微分，你就要算z5跟z6的偏微分，如果我们是从z1，z2的偏微分开始算，那就没有效率，  

但是如果你反过来先算z5，z6的偏微分的话，这个process，就突然之间变得有效率起来了，

![Screen Shot 2022-01-15 at 10.28.45 PM](https://raw.githubusercontent.com/lunnche/picgo-image/main/Screen%20Shot%202022-01-15%20at%2010.28.45%20PM.png)

那么怎么用z5,z6来算z3呢？如前所述，用一个op-amp来算，这每一个op-amp它放大的倍率呢就是$\sigma'(z_1)$-$\sigma'(z_4)$

你先算出z5对C的偏微分，z6对C的偏微分，再把这些偏微分值乘上weight加起来，再通过op-amp,你就得到z3对C的偏微分的值和z4对C的偏微分的值，以此类推，这个步骤就叫做Backward Pass,你在做Backward Pass的时候，你实际上的做法就是，建另外一个neural network:我们本来有一个正向的neural network，里面的activation function都是sigmoid function,那现在，你在算Backward pass 的时候，你就是建一个反向的neural network,反向网络的activation function你要先算完Forward pass以后，你才算得出来  

![Screen Shot 2022-01-15 at 10.39.20 PM](https://raw.githubusercontent.com/lunnche/picgo-image/main/Screen%20Shot%202022-01-15%20at%2010.39.20%20PM.png)

## Backpropagation - Summary  
Backpropagation是怎么做的：
首先你做一个Forward pass，在做Forward pass的时候，你可以算出：只要你知道每一个activation function的output，即它所连接的weight的$\frac{\partial{z}}{\partial{w}}$ ,在Backward pass 里面，你要把原来的neural network的方向呢倒过来，倒过来的neural network里，它的每一个三角形的output呢就是$\frac{\partial{C}}{\partial{z}}$ ,然后把上述两个东西乘起来，就得到$\frac{\partial{C}}{\partial{w}}$ 

$$
\frac{\partial{z}}{\partial{w}}=a\\
\\
a \times 
\frac{\partial{C}}{\partial{z}}=
\frac{\partial{C}}{\partial{w}}
$$

![Screen Shot 2022-01-15 at 10.53.35 PM](https://raw.githubusercontent.com/lunnche/picgo-image/main/Screen%20Shot%202022-01-15%20at%2010.53.35%20PM.png)


