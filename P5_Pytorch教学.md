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
>>> x = torch.zeros([2,
