# PyTorch Tutorial  

## Outline  
* Prerequisites  
* What is PyTorch?  
* PyTorch v.s. TensorFlow  
* Overview of the DNN Training Procedure  
* Tensor  
* How to Calculate Gradient?  
* Dataset & Dataloader  
* torch.nn  
* torch.optim  
* Neural Network Training/Evaluation  
* Saving/Loading a Neural Network  
* More About PyTorch  

## Prerequisites  
* We assume you are already familiar with... 
    * Python3  
        * if-else,loop,function,file IO,class, ...  

    * Numpy  
        * array & array operations  

## What is PyTorch?  
* An open source machine learning framework.  
* A Python package that provides two high-level features:  
    * Tensor computation(like NumPy)with strong GPU acceleration  
        Tensor:就比如说NumPy的array
    * Deep neural networks built on a tape-based autograd system
        算梯度下降之类  

![Screen Shot 2022-01-21 at 10.10.04 AM](https://raw.githubusercontent.com/lunnche/picgo-image/main/Screen%20Shot%202022-01-21%20at%2010.10.04%20AM.png)

一句话说PyTorch和TensorFlow的区别：PyTorch科研，TensorFlow做产品  

## Overview of the DNN Training Procedure  

![image-20220121105919101](https://raw.githubusercontent.com/lunnche/picgo-image/main/image-20220121105919101.png)

## Tensor  

* High-dimensional matrix(array)  

![image-20220121110121644](https://raw.githubusercontent.com/lunnche/picgo-image/main/image-20220121110121644.png)


## Tensor -- Data Type  

那么一个Tensor里存什么东西呢？ 最常见的就是 浮点数和整数

![image-20220121110544580](https://raw.githubusercontent.com/lunnche/picgo-image/main/image-20220121110544580.png)

![image-20220121110828178](https://raw.githubusercontent.com/lunnche/picgo-image/main/image-20220121110828178.png)

注意一个巧妙的点：  
一维是(5,），不是(1,5),(1,5)是一行五列，表示的是二维空间里面的概念  
一维是(5,)而不是(5)，是因为python会把后者认为是一个int，而非tuple。【弹幕】

PyTorch里面的dim 和 NumPy里面的axis 一样  

## Tensor -- Constructor  
* From list/NumPy array  
    x = torch.tensor([[1,-1],[-1,1]])  
    x = torch.from_numpy(np.array([[1,-1],[-1,1]]))  
* Zero tensor  
    x = torch.zeros([2,2])  
* Unit tensor  
    x = torch.ones([1,2,5])  

![image-20220121112032504](https://raw.githubusercontent.com/lunnche/picgo-image/main/image-20220121112032504.png)

## Tensor -- Operators  

到05:41

* Squeeze: remove the specified dimension with length = 1  
```
>>> x = torch.zeros([1,2,3])
>>> x.shape
torch.Size([1,2,3])
>>> x = x.squeeze(0)
>>> x.shape
torch.Size([2,3])
```

![image-20220125084945699](https://raw.githubusercontent.com/lunnche/picgo-image/main/image-20220125084945699.png)

* Unsqueeze: expand a new dimension  
```
>>> x = torch.zeros([2,3])
>>> x.shape
torch.Size([2,3])
>>> x = x.unsqueeze(1)
>>> x.shape
torch.Size([2,1,3])
```

![image-20220125085433022](https://raw.githubusercontent.com/lunnche/picgo-image/main/image-20220125085433022.png)

* Transpose: transpose two specified dimensions  
```
>>> x = torch.zeros([2,3])
>>> x.shape
ttorch.Size([2,3])
>>> x = x.transpose(0,1)
>>> x.shape
torch.Size([3,2])
```

![image-20220125215355409](https://raw.githubusercontent.com/lunnche/picgo-image/main/image-20220125215355409.png)


* Cat: concatenate multiple tensors
```
>>> x = torch.zeros([2,1,3])
>>> y = torch.zeros([2,3,3])
>>> z = torch.zeros([2,2,3])
>>> w = torch.cat([x,y,x,z],dim=1)
>>> w.shape
torch.Size([2,6,3])
```

![image-20220125215611862](https://raw.githubusercontent.com/lunnche/picgo-image/main/image-20220125215611862.png)

* Addition 
z = x + y

* Subtraction  
z = x - y  

* Power  
y = x.pow(2)  

* Summation   
y = x.sum()  

* Mean  
y = x.mean()  

more operators:  https://pytorch.org/docs/stable/tensors.html  

## Tensor -- PyTorch v.s. NumPy  
* Attributes  
|PyTorch|NumPy|
|:-:|:-:|
|x.shape|x.shape|
|x.dtype|x.dtype|
|x.reshape / x.view|x.reshape|
|x.squeeze()|x.squeeze()|
|x.unsqueeze(1)|np.expand_dims(x,1)|

## Tensor -- Device  
* Default: tensors & modules will be computed with CPU  
* CPU
x = x.to('cpu')  
* GPU
x = x.to('cuda')  

## Tensor -- Device(GPU)
* Check if your computer has NVIDIA GPU
torch.cuda.is_available()
* Multiple GPUs:specify 'cuda:0','cuda:1','cuda:2',...
* Why GPU?  
    * Parallel computing  
    * https://towardsdatascience.com/what-is-a-gpu-and-do-you-need-one-in-deep-learning-718b9597aa0d  

## What is a GPU and do you need one in Deep learning?  

![image-20220125221539449](https://raw.githubusercontent.com/lunnche/picgo-image/main/image-20220125221539449.png)

任何一名数据科学家或机器学习爱好者，如果一直在尝试大规模地激发训练模型的性能，将在某一时刻达到上限，并开始经历不同程度的处理延迟. 当数据集变得更大时，原本只需几分钟就能完成的任务现在可能要花费更多的时间——有时甚至是几周。

有人说Deep Learning requires big systems to run execute.  

对于神经网络来说，训练阶段是最需要计算资源的。  

在训练过程中，神经网络会接收输入信息，然后利用训练过程中调整的权值在隐藏层中进行处理，然后模型会给出一个预测。调整权重以找到模式，以便更好地进行预测。  

这两个操作本质上都是矩阵乘法。一个简单的矩阵乘法可以用下图来表示

![image-20220125223117449](https://raw.githubusercontent.com/lunnche/picgo-image/main/image-20220125223117449.png)

在神经网络中，我们可以将第一个数组作为神经网络的输入，而第二个数组构成神经网络的权值。如果你的神经网络有10个、100个甚至10万个参数。一台计算机仍然能够在几分钟内处理这些问题，甚至最多几个小时  

但是如果你的神经网络有超过100亿个参数呢?采用传统方法来训练这种系统将需要数年时间。你的电脑很可能在你还没走到十分之一的路程时就放弃了。  

一个神经网络接受搜索输入，并从1亿个输出或产品中进行预测，通常会在每个产品中获得大约2000个参数。所以你把它们相乘，神经网络的最后一层现在是2000亿个参数。我没有做任何复杂的事情。我说的是一个非常非常简单的神经网络模型  

通过简单地同时运行所有操作，而不是一个接一个地运行，深度学习模型可以训练得更快  

图形处理器(Graphics Processing Unit)是一种具有专用内存的专用处理器，它通常执行渲染图形所需的浮点运算.它是一种用于广泛的图形和数学计算的单芯片处理器，可以为其他工作释放CPU周期  

gpu和cpu之间的主要区别是，与cpu相比，gpu用于算术逻辑单元的晶体管比例更高，用于缓存和流控制的晶体管比例更低,虽然cpu主要适用于需要解析或解释代码中的复杂逻辑的问题，但gpu是专门为计算机游戏的图形渲染而设计的，后来被增强以加速其他几何计算(例如，转换多边形或旋转垂线到不同的坐标系统，如3D)。  

GPU比CPU小，但往往有更多的逻辑核心(算术逻辑单元或alu，控制单元和内存缓存)。

![image-20220125224152546](https://raw.githubusercontent.com/lunnche/picgo-image/main/image-20220125224152546.png)

在上图中，你可以看到gpu(红/绿)理论上可以完成cpu(蓝)的10 - 15倍的操作。这种加速在实践中也非常适用。  

如果把CPU比作玛莎拉蒂(Maserati)，那么GPU就是一辆大卡车  

CPU(玛莎拉蒂)可以在RAM中快速获取少量的包(3 -4名乘客)，而GPU(卡车)速度较慢，但可以一次获取大量的内存(约20名乘客)

看看这个视频 https://www.youtube.com/watch?v=-P28LKWTzrI&t=1s  

## Why choose GPUs for Deep Learning  

决定是使用CPU还是GPU来训练深度学习模型，有几个决定参数:

1. 内存带宽  
带宽是gpu在计算上比cpu快的主要原因之一。对于大型数据集，CPU在训练模型时需要占用大量内存。  
计算巨大而复杂的任务会占用CPU大量的时钟周期——CPU顺序地占用任务，并且比GPU的核数更少。


