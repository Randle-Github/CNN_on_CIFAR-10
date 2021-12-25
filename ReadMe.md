# CS231n assignment2 心得分享

分享人：本-2020-刘洋岑

分享人的mentor：硕-2020-李洵松

## 1.CNN基本知识

### 1.总论

卷积神经网络早在上个世纪即被提出，然而直到21世纪10年代才成为学术界主流研究方向，这种模型被证明对于图像处理有着极其优秀的效果。ANN全连接对于计算造成了巨大的损耗，比如最基本的RGB图像是3072个特征，第一层使用20个神经元，则需要$3072\times 20$次连接。

CNN的提出是一定程度上减少了参数（权值共享），并且更加有效的利用了图像的一些特征信息。

基本的CNN框架为：将输入进行卷积+激励+池化操作。进行特征提取后，将张量拉长传入普通的ANN进行分类任务。

![CNN](D:\Work\Paper\week report\20级-本-刘洋岑-11月7日交流会准备材料\CNN.png)

### 2.卷积

卷积的模式是使用滑动窗口，将特定的卷积核进行扫描并且按元素位置相乘，将上一层所有通道进行这个操作后，求各个通道这个位置的元素和，将得到的值放入新的图像位。

疑问：此处的卷积和数学上的卷积关系。

#### 关于几种常用卷积核的测试结果：

##### 原图：

<img src="D:\Work\Paper\week report\20级-本-刘洋岑-11月7日交流会准备材料\3origin.png" alt="3origin" style="zoom:50%;" />

##### 均值卷积核：$$\left[ \begin{matrix} \frac{1}{9} & \frac{1}{9} & \frac{1}{9} \\ \frac{1}{9} & \frac{1}{9} & \frac{1}{9} \\ \frac{1}{9} & \frac{1}{9} & \frac{1}{9} \end{matrix} \right]$$

<img src="D:\Work\Paper\week report\20级-本-刘洋岑-11月7日交流会准备材料\3conv1.png" alt="3conv1" style="zoom:50%;" />

##### 锐化卷积核（加强边界特征）：$$\left[ \begin{matrix} -1 & -1 &-1 \\-1 &9 & -1 \\ -1 & -1 & -1 \end{matrix} \right] \times \frac{1}{16}$$

<img src="D:\Work\Paper\week report\20级-本-刘洋岑-11月7日交流会准备材料\3conv2.png" alt="3conv2" style="zoom:50%;" />

##### Laplace算子卷积核（提取边缘特征）：$$\left[ \begin{matrix} 0 & 1 &0 \\1 &-4 &1 \\ 0 & 1 & 0 \end{matrix} \right]$$

<img src="D:\Work\Paper\week report\20级-本-刘洋岑-11月7日交流会准备材料\3conv3.png" alt="3conv3" style="zoom:50%;" />

##### 水平卷积核（水平边缘检测）：$\left[ \begin{matrix} 1 & 1 &1 \\0 &0 &0\\ -1 & -1 & -1 \end{matrix} \right]$

<img src="D:\Work\Paper\week report\20级-本-刘洋岑-11月7日交流会准备材料\3conv4.png" alt="3conv4" style="zoom:50%;" />

以上是一些人们也能够理解含义的特殊卷积核，但是初始化以及机器学习的卷积则可能不太容易理解。可以看出，使用不同的卷积核（滤波器)，得到的图像特征是不同的。对于不同的图像，使用卷积化可以减少特征的同时减少参数的使用。

卷积学习到的特征有什么含义？

> 卷积学习到的特征和人学习到的特征概念是不同的。比如，浅层卷积可能学到一些特定的形状，深层的卷积可能学到一些纹路颜色特点。总体而言，这种特征是自然语言不太好描述的类型。使用猫的图片加入大象的皮肤褶皱，CNN更加倾向于将图片识别为大象。而今年提出的Transformer倾向于将图片识别为猫（证明Transformer更加接近于人类视觉运作方式）。

在实际操作时，卷积层根据需要有一个padding机制，保证图片尺寸不变。

### 3.激励层

此处使用RELU激活函数。
$$
ReLU(x)=max(0,x)
$$
这种激励函数容易遇到一个DEAD的情况，人们使用一种Leakly ReLU进行试验，但是实际上没有什么提升。
$$
LeaklyReLU(x)=\begin{cases} &\beta x(x\textless0)\\&x(x\geq0) \end{cases}
$$
为什么要用ReLU不用其余的激励函数？

> 按照CS231n的说明，首先是实验结果ReLU在CNN表现更好。其次人类的神经元激励作用更加接近于ReLU的形式，并且Sigmoid,Tanh存在一定的梯度消失问题。

### 4.池化层

常见池化大致分为三种类型：均值池化，最大值池化和随机池化。

实验表明最大池化效果较好。Max_Pooling即在得到的每一个通道进行，对于每一个滑动窗口，如$2*2$的滑动窗口，则保留这个窗口内最大的数值。池化后，图像的尺寸会进行相应的减少（损失一定的信息）。

关于反向传播：

a.对于最大池化：只对窗口中最大的这个值传入梯度，其余置0；

b.对于均值池化：窗口中每个值获得相同的梯度；

c.对于随机池化：根据正向传播时保留的那一个位置进行梯度传播。

### 5.归一化

Batch_norm对于指定的维度，将所有数据进行重新分布。比如规定所有图像的某个像素点在这一层服从均值为0，方差为1的分布。则使用如下公式：
$$
value\prime_i=\beta+\gamma(value_i-m)/(\sigma+\epsilon)
$$

偏置项防止方差为0。关于归一化，需要分为训练与测试两种情况进行讨论。训练时方差以及均值需要计算，而测试时方差以及均值已经固定。

### 6.Drop Out 技巧

Drop Out技巧在全连接层训练时按照一定的概率使得一些节点消失，防止过度依赖某一个特征。测试时关闭drop out，对正确率有少量提升。

### 7.全连接层（接ANN）

此时经过前期的预处理，图像的特征已经大致保留并且减少。直接接入ANN，最终放置SoftMax分类器进行处理。关于仿射（从第i到i+1层）：
$$
&X_{i+1}=W_iX_i+b_i\\
&dX_i=\frac{\partial loss}{\partial X_{i+1}}\frac{\partial{X_{i+1}}}{\partial{X_i}}=dX_{i+1}W_i^T\\
&dW_i=\frac{\partial loss}{\partial X_{i+1}}\frac{\partial{X_{i+1}}}{\partial{W_i}}=X_i^TdX_{i+1}\\
&db_i=\frac{\partial loss}{\partial X_{i+1}}\frac{\partial{X_{i+1}}}{\partial{b_i}}=\sum dX_{i+1,j}
$$

注：这里的b($10$)在式子中size匹配不上，利用了numpy加法的维度拉伸性质($n\times 10$)。即对于每一个样本，只要是同一类别，偏置相同。

### 8.SoftMax分类器以及交叉熵损失

SoftMax将ANN最后一层10个分类的节点进行处理，得到一个概率估计。

SoftMax：对于第j个样本，标签为j，最后一层为x，则对于类别i使用SoftMax进行概率估计，可以保证所有的$P_i$在$[0,1]$进行分布：
$$
P_i=\frac{\exp(x_i)}{\sum_{i=0}^{9} \exp(x_i)}
$$
KL散度：KL散度度量两个分布之间的关系。此处为，对于标签而言，正确的分布是标签项为1，其余所有分类都为0。而使用SoftMax得到的预测分布，需要使用这两个分布的交叉熵来表示损失函数：
$$
\begin{align}D_{KL}(p||q)&=\sum_{i=1}^{n}p_{x_i}\ln(\frac{p_{x_i}}{q_{x_i}})\\&=\sum_{i=1}^{n}p_{x_i}\ln{p_{x_i}}-\sum_{i=1}^{n}p_{x_i}\ln{q_{x_i}}\end{align}
$$
cross entropy：根据上面KL散度的表达式，由于前半部分是定值（恒为0），取后半部分作为交叉熵损失。将分类任务代入交叉熵，并且在末端加入正则化表达式得到对于这n个样本进行估计的总损失函数：
$$
Loss=\sum_{j=0}^{n-1}(-y_j\ln P_{y_j}/n)+normalization
$$
可以证明交叉熵损失具有凸性，正则化为二次型也具有凸性。

反向传播：
$$
i=y_j:\frac{\partial Loss}{\partial x_i}=(\frac{\exp(x_i)}{\sum_{i=0}^{9} \exp(x_i)}-1)/n=(p_{y_j}-1)/n\\
i\neq y_j:\frac{\partial Loss}{\partial x_i}=\frac{\exp(x_i)}{\sum_{i=0}^{9} \exp(x_i)}/n=p_i/n
$$
cross entropy为什么比 multi-SVM loss(hinge loss)和mean squared error在分类问题上更优秀？

> hinge loss在达到一定边界时损失为0，而cross entropy可以一直优化。
>
> 而对于MSE而言，MSE无差别地关注全部类别上预测概率和真实概率的差。交叉熵关注的更多的是正确类别的预测概率，并且交叉熵的损失数值大小范围没有受到限制，作图对比更加明显。对于两个分布差异的刻画，交叉熵体现得更加的优秀。
>
> 总结：分类问题大多用 cross entropy，回归问题大多用 mean squared error。

### 附.卷积演示

MNIST数据集演示:
![demonstration](D:\Work\Paper\week report\20级-本-刘洋岑-11月7日交流会准备材料\demonstration.png)

CIFAR-10数据集演示：

![convdemon](D:\Work\Paper\week report\20级-本-刘洋岑-11月7日交流会准备材料\convdemon.png)

## 2.代码实现以及细节展现

基本信息：代码使用numpy实现最简单的CNN框架，并且在CIFAR-10数据集（5个validation_batch分别有10000个样本，一个test_batch10000个样本）进行实践。

代码部分：分为my_CNN.py（主体框架）与my_layers.py（仿照Torch进行层间操作）。

硬件设备：Lenovo轻薄本4核CPU（跑的轮数很低，效果较差）。

模型结构：input(n,3,32,32) - conv1(n,10,32,32) - bn1 - relu1 - pool1(n,10,16,16) - conv2(n,20,16,16) - bn2 - relu2 - pool2(n,20,8,8) - conv3(n,10,8,8) - bn3 - relu3 - conv3(n,10,4,4) - spread(n,160) - layer1(n,20) - bn1 - dropout1 - relu1 - layer2(n,20) - bn2 - dropout2 - relu2 - layer3 - bn3(n,10) - dropout3 - relu3 - softmax(n,10)

### 2.1.my_layers.py部分

#### 2.1.1.Affine layers

仿射：

```python
def affine_forward(x, w, b):  # (batch_size,n1) (n1,n2) (n2)
    out = np.dot(x, w) + b  # (batch_size,n2)
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):  # (batch_size,n2)
    x, w, b = cache
    dx, dw, db = np.dot(dout, w.T), np.dot(x.T, dout), np.sum(dout, axis=0)
    return dx, dw, db
```

#### 2.1.2 Activation layers

Sigmoid型：

```python
def sigmoid_forward(x):  # (batch_size,n)
    cache = x
    out = np.ones(np.shape(x)) / (1 + np.exp(-x))
    return out, cache


def sigmoid_backward(dout, cache):
    out = np.ones(np.shape(cache)) / (1 + np.exp(-cache))
    dcache = dout * out * (1 - out)
    return dcache
```

ReLU型：

```python
def relu_forward(x):  # (batch_size,n)
    cache = x
    out = x
    out[x < 0] = 0
    return out, cache


def relu_backward(dout, cache):
    dcache = dout
    dcache[cache < 0] = 0
    return dcache
```

### 2.1.3.Dropout layers

Dropout层：

```python
def dropout_forward(x, dropout_param):
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])
    mask = np.random.random(np.shape(x)) - p
    mask = np.where(mask < 0, 1, 0)
    out = x * mask
    cache = (mask, dropout_param)
    return out, cache


def dropout_backward(dout, cache):
    mask, dropout_param = cache
    dx = dout * mask
    return dx
```

### 2.1.4.Convlution layers

卷积层：

```python 
def conv_forward_naive(x, w, b, conv_param):  # b:(F)
    stride, pad = conv_param['stride'], conv_param['pad']
    N, C, H, W = np.shape(x)  # pictures
    F, _, HH, WW = np.shape(w)  # filters
    hh = int(1 + (H + 2 * pad - HH) / stride)
    ww = int(1 + (W + 2 * pad - WW) / stride)
    new_x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), "constant", constant_values=(0))
    out = np.zeros((N, F, hh, ww))
    for i in range(N):
        for f in range(F):
            for j in range(hh):
                for k in range(ww):
                    out[i, f, j, k] = np.sum(
                        new_x[i, :, j * stride:j * stride + HH, k * stride:k * stride + WW] * w[f]) + b[f]
    cache = (x, w, b, conv_param)
    return out, cache  # out:(N, F, h, w)


def conv_backward_naive(dout, cache):  # dout:(N, F, h, w)
    x, w, b, conv_param = cache
    stride, pad = conv_param['stride'], conv_param['pad']
    N, C, H, W = np.shape(x)  # pictures
    F, C, HH, WW = np.shape(w)  # filters
    hh = int(1 + (H + 2 * pad - HH) / stride)
    ww = int(1 + (W + 2 * pad - WW) / stride)
    new_x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), "constant", constant_values=(0))
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    dx = np.zeros_like(new_x)
    for i in range(N):
        for f in range(F):
            for j in range(hh):
                for k in range(ww):  # (j,k)->(j*stride:j*stride+HH, k*stride:k*stride+WW)
                    dx[i, :, j * stride:j * stride + HH, k * stride:k * stride + WW] += w[f] * dout[i, f, j, k]
                    db[f] += dout[i, f, j, k]
                    dw[f] += new_x[i, :, j * stride:j * stride + HH, k * stride:k * stride + WW] * dout[i, f, j, k]
    dx = dx[:, :, pad:pad + H, pad:pad + W]
    return dx, dw, db
```

### 2.1.5.Max pool layers

池化层：

```python 
def max_pool_forward_naive(x, pool_param):  # tacit consent that it could be divided with no remainder
    N, C, H, W = np.shape(x)
    stride, HH, WW = pool_param['stride'], pool_param['height'], pool_param['width']
    hh = int(1 + (H - HH) / stride)
    ww = int(1 + (W - WW) / stride)
    out = np.zeros((N, C, hh, ww))
    for i in range(N):
        for c in range(C):
            for j in range(hh):
                for k in range(ww):
                    out[i, c, j, k] = np.max(x[i, c, j * stride:j * stride + HH, k * stride:k * stride + WW])
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    x, pool_param = cache
    N, C, H, W = np.shape(x)
    stride, HH, WW = pool_param['stride'], pool_param['height'], pool_param['width']
    hh = int(1 + (H - HH) / stride)
    ww = int(1 + (W - WW) / stride)
    dx = np.zeros(np.shape(x))
    for i in range(N):
        for c in range(C):
            for j in range(hh):
                for k in range(ww):
                    max_val = np.max(x[i, c, j * stride:j * stride + HH, k * stride:k * stride + WW])
                    for l in range(j * stride, j * stride + HH):
                        for m in range(k * stride, k * stride + WW):
                            if x[i, c, l, m] == max_val:
                                dx[i, c, l, m] = dout[i, c, j, k]
    return dx
```

### 2.1.6.Norm layers

均一化层:

```python
def batchnorm_forward(x, gamma, beta, bn_param):
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)
    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))
    out, cache = None, None
    if mode == 'train':
        sample_mean = np.mean(x, axis=0, keepdims=True)
        sample_var = np.var(x, axis=0, keepdims=True)
        x_normalized = (x - sample_mean) / np.sqrt(sample_var + eps)
        out = gamma * x_normalized + beta
        cache = (x_normalized, gamma, beta, sample_mean, sample_var, x, eps)
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
    elif mode == 'test':
        x_normalized = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_normalized + beta
    return out, cache


def batchnorm_backward(dout, cache):
    dx, dgamma, dbeta = None, None, None
    x_normalized, gamma, beta, sample_mean, sample_var, x, eps = cache
    N, D = x.shape
    dx_normalized = dout * gamma
    x_mu = x - sample_mean
    sample_std_inv = 1.0 / np.sqrt(sample_var + eps)
    dsample_var = -0.5 * np.sum(dx_normalized * x_mu, axis=0, keepdims=True) * sample_std_inv ** 3
    dsample_mean = -1.0 * np.sum(dx_normalized * sample_std_inv, axis=0, keepdims=True) - 2.0 * dsample_var * np.mean(x_mu, axis=0, keepdims=True)
    dx1 = dx_normalized * sample_std_inv
    dx2 = 2.0 / N * dsample_var * x_mu
    dx = dx1 + dx2 + 1.0 / N * dsample_mean
    dgamma = np.sum(dout * x_normalized, axis=0, keepdims=True)
    dbeta = np.sum(dout, axis=0, keepdims=True)
    return dx, dgamma, dbeta
```

### 2.1.7.Loss function

L-2 norm loss：

```python
def onehot_loss(X, y):
    dx = 2 * X
    loss = np.sum(X * X)
    for i in range(X.shape[0]):
        loss += -X[i][y[i]] ** 2 + (1 - X[i][y[i]]) ** 2
        dx[i][y[i]] = 2 * (X[i][y[i]] - 1)
    return loss / X.shape[0], dx
```

Cross entropy loss:

```python
def softmax_loss(X, y):  # y: (batch_size) the label
    loss = 0
    x = X - np.max(X, axis=1).reshape(X.shape[0], 1)
    p = np.exp(x)
    p = p / np.sum(p, axis=1).reshape(p.shape[0], 1)
    dx = p
    for i in range(x.shape[0]): # loss += -np.log(p[i][y[i]])
        dx[i][y[i]] = p[i][y[i]] - 1
    return loss / X.shape[0], dx
```



### 2.2.my_CNN.py部分

ANN的初始化：

```python
class FullyConnectedNet(object):
    def __init__(self, input_dim=160, num_classes=10, dropout_keep_ratio=1, normalization=None, reg=0, weight_scale=1e-3, seed=None, activition="ReLU", losstype="softmax_loss"):
```

CNN的初始化：

```Python
class ConvNet(object):
    def __init__(self, input_dim=(3, 32, 32), connect_conv=0, use_batchnorm=False,
                 weight_scale=1e-1):
```

主体框架：

```Python
class NeuralNetwork():
    def __init__(self):
        self.cnn = ConvNet()
        self.ann = FullyConnectedNet(input_dim=160, activition='ReLU', losstype='softmax_loss')

    def forward(self, X):
        ann_input = self.cnn.forward(X)
        return self.ann.forward(ann_input)

    def loss(self, X, y=None, mode='train'):
        self.forward(X)
        return self.ann.loss(y)

    def backward(self):
        dann = self.ann.backward()
        self.cnn.backward(dann)

    def train(self, X, y, iter_times=100, batch_size=20, draw=False):  # train the networks
        num = X.shape[0]
        loss_history = []
        for i in range(iter_times):
            select = np.random.choice(num, batch_size, replace=False).astype(np.int32) # select the mini_batch samples
            select = np.arange(batch_size)
            x_batch = np.zeros((batch_size, 3, 32, 32))
            y_batch = np.zeros(batch_size)
            for k in range(batch_size):
                x_batch[k] = X[select[k]]
                y_batch[k] = y[select[k]]
            y_batch = y_batch.astype(np.int32)
            loss_history.append(self.loss(x_batch, y_batch))
            self.backward()
        if draw == True: # draw the figure of the loss curve
            plt.ylabel("loss")
            plt.xlabel("iter_time")
            plt.plot(np.arange(iter_times), loss_history)
            plt.show()
```



## 3.测试经历

### 3.1.调参的困难

最开始由于我的实验设备是CPU，不能跑太大的mini-batch，猜测在小mini-batch上得到的mean和方差可能与大数据集相差较大。故没有均一化。

使用交叉熵函数导致训练过程中损失波动极大，使用ReLU激活函数时，稍微训练几轮，很快就梯度爆炸或者梯度消失，参数就到$10^{32}$或者$10^{-32}$之类的比较夸张的数量级。使用Sigmoid激活函数，很快便陷入了Sigmoid函数常见的梯度消失问题。

对于两种损失函数类型，固定卷积层之后，花费了大量的时间进行参数初始化以及学习率的调试，数量级终于大致趋于稳定，但是出现了第二个问题。

### 3.2.结果总是相同

观察了loss函数，有下降的趋势，认为梯度正确。

<img src="D:\Work\Paper\week report\20级-本-刘洋岑-11月7日交流会准备材料\loss.png" alt="loss" style="zoom: 50%;" />

然而发现预测每一个函数都是同一个结果，即训练集中出现次数最多的那一个类别。迫不得已加入了BN层，但是问题仍然没有解决。于是对于一个固定的mini-batch进行测试，batch_size设置为5，运行了100轮解决了问题。问题为迭代次数不够。在MNIST数据集未出现这个问题，猜测原因如下：MNIST数据集简单，而且在神经网络迭代较少次数的情况下，后面几层的数据有着明显的差异。而在CIFAR-10数据集，迭代次数少的时候，较深的几层数据接近。这时最后一层到SoftMax的数值主要取决于训练集出现次数最多的一个类别。

在CIFAR-10数据集上，对于小样本能够过拟合。调试参数：iteration_time=100，batch_size=100，得到42%正确率。但是去掉卷积层的反向传播，仅仅使用全连接层，正确率几乎不变，打印卷积核，猜测是由于训练迭代次数太少，卷积核太少，导致卷积层作用不大。

## 4.附录：PyTorch框架实现CNN参考

```Python
class CNNnet(torch.nn.Module):
    def __init__(self):
        super(CNNnet, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=16,
                            kernel_size=3,
                            stride=2,
                            padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 3, 2, 1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 2, 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 2, 2, 0),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.mlp1 = torch.nn.Linear(2 * 2 * 64, 100)
        self.mlp2 = torch.nn.Linear(100, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.mlp1(x.view(x.size(0), -1))
        x = self.mlp2(x)
        return x
    
model = CNNnet()

loss_func = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(),lr=0.001)

loss_func = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.001)

loss_count = []
for epoch in range(2):
    for i, (x, y) in enumerate(train_loader):
        batch_x = Variable(x)
        batch_y = Variable(y)
        out = model.foward(batch_x)
        loss = loss_func(out, batch_y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_count.append(loss)
```
