from my_layers import *
import cv2
import matplotlib.pyplot as plt

'''
conv1-relu1-pool1-conv2-relu2-pool2-conv3-relu3-conv3-layer1-dropout1-relu1-layer2-dropout2-relu2-layer3-
dropout3-relu3-softmax
'''


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def show_picture(x):  # show a picture
    image_m = x
    r = image_m[0, :, :]
    g = image_m[1, :, :]
    b = image_m[2, :, :]
    img23 = cv2.merge([r, g, b])
    plt.figure()
    plt.imshow(img23)
    plt.show()


length = 3072
# 加载测试集
dict_data = []  # [b'batch_label', b'labels', b'data', b'filenames']
dict_data.append(unpickle("data_batch_1"))
dict_data.append(unpickle("data_batch_2"))
dict_data.append(unpickle("data_batch_3"))
dict_data.append(unpickle("data_batch_4"))
dict_data.append(unpickle("data_batch_5"))

X = np.zeros((5, 10000, 3, 32, 32))  # transform all the pictures into numpy array
y = np.zeros((5, 10000))  # transform all the labels into numpy array
for i in range(5):
    for j in range(len(dict_data[i][b'data'])):
        X[i][j] = np.array(dict_data[i][b'data'][j]).reshape((3, 32, 32)) / 255
    y[i] = np.array(dict_data[i][b'labels']).astype(np.int32)


class FullyConnectedNet(object):  # layer1-dropout1-relu1-layer2-dropout2-relu2-layer3-dropout3-relu3-softmax
    def __init__(self, input_dim=160, num_classes=10, dropout_keep_ratio=1, normalization=None,
                 reg=0, weight_scale=1e-3, seed=None, activition="ReLU", losstype="softmax_loss"
                 ):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.normalization = normalization
        self.reg = reg
        self.weight_scale = weight_scale
        self.activition = activition
        self.params = {}
        self.params['W1'] = 0.01 * weight_scale * np.random.normal(loc=1, scale=1, size=(self.input_dim, 10))
        self.params['b1'] = np.zeros(10)
        self.params['W2'] = 0.01 * weight_scale * np.random.normal(loc=1, scale=1, size=(10, 10))
        self.params['b2'] = np.zeros(10)
        self.params['W3'] = weight_scale * np.random.normal(loc=1, scale=1, size=(10, self.num_classes))
        self.params['b3'] = np.zeros(self.num_classes)
        self.losstype = losstype
        self.dropout_param = {}
        self.dropout_param['seed'] = seed
        self.dropout_param['p'] = dropout_keep_ratio

    def forward(self, X, mode='test', dropout_ratio=1):
        self.dropout_param['mode'] = mode
        self.dropout_param['p'] = dropout_ratio
        self.input = X  # N*160
        if self.activition == 'ReLU':
            self.layer1, self.cachelayer1 = affine_forward(self.input, self.params['W1'], self.params['b1'])  # N*30
            self.drop1, self.cachedrop1 = dropout_forward(self.layer1, self.dropout_param)
            self.relu1, self.cacherelu1 = relu_forward(self.drop1)
            self.layer2, self.cachelayer2 = affine_forward(self.relu1, self.params['W2'], self.params['b2'])  # N*30
            self.drop2, self.cachedrop2 = dropout_forward(self.layer2, self.dropout_param)
            self.relu2, self.cacherelu2 = relu_forward(self.drop2)
            self.layer3, self.cachelayer3 = affine_forward(self.relu2, self.params['W3'], self.params['b3'])  # N*10
            self.drop3, self.cachedrop3 = dropout_forward(self.layer3, self.dropout_param)
            self.relu3, self.cacherelu3 = relu_forward(self.drop3)
            self.out = np.argmax(self.relu3, axis=1)
            self.weight_sum = np.sum(self.params['W1']) + np.sum(self.params['W2']) + np.sum(self.params['W3'])

        if self.activition == 'Sigmoid':
            self.layer1, self.cachelayer1 = affine_forward(self.input, self.params['W1'], self.params['b1'])  # N*30
            self.drop1, self.cachedrop1 = dropout_forward(self.layer1, self.dropout_param)
            self.sig1, self.cachesig1 = sigmoid_forward(self.drop1)
            self.layer2, self.cachelayer2 = affine_forward(self.sig1, self.params['W2'], self.params['b2'])  # N*30
            # print("layer2", "\n", self.layer2, end="\n" * 2)
            self.drop2, self.cachedrop2 = dropout_forward(self.layer2, self.dropout_param)
            # print("layer2", "\n", self.layer2, end="\n" * 2)
            self.sig2, self.cachesig2 = sigmoid_forward(self.drop2)
            self.layer3, self.cachelayer3 = affine_forward(self.sig2, self.params['W3'], self.params['b3'])  # N*10
            self.drop3, self.cachedrop3 = dropout_forward(self.layer3, self.dropout_param)
            self.sig3, self.cachesig3 = sigmoid_forward(self.drop3)
            self.out = np.argmax(self.sig3, axis=1)
            self.weight_sum = np.sum(self.params['W1']) + np.sum(self.params['W2']) + np.sum(self.params['W3'])
        # print(self.input[:, :10], end="\n" * 2)
        print("layer1", "\n", self.layer1, end="\n" * 2)
        # print("W2", "\n", self.params['W2'], end='\n' * 2)
        # print("layer2", "\n", self.layer2, end="\n" * 2)
        print("W3", "\n", self.params['W3'], end="\n" * 2)
        print("layer3", '\n', self.layer3, end="\n" * 2)
        print("out", '\n', self.out, end="\n" * 2)
        return self.out

    def backward(self, mode='test', learning_rate=0.2):
        self.dropout_param['mode'] = mode
        if self.activition == 'ReLU':
            drelu3 = relu_backward(self.dout, self.cacherelu3)
            ddrop3 = dropout_backward(drelu3, self.cachedrop3)
            dlayer3 = affine_backward(ddrop3, self.cachelayer3)
            self.params['W3'] -= learning_rate * dlayer3[1] + self.reg * self.params['W3']
            self.params['b3'] -= learning_rate * dlayer3[2] + self.reg * self.params['b3']
            drelu2 = relu_backward(dlayer3[0], self.cacherelu2)
            ddrop2 = dropout_backward(drelu2, self.cachedrop2)
            dlayer2 = affine_backward(ddrop2, self.cachelayer2)
            self.params['W2'] -= learning_rate * dlayer2[1] + self.reg * self.params['W2']
            self.params['b2'] -= learning_rate * dlayer2[2] + self.reg * self.params['b2']
            drelu1 = relu_backward(dlayer2[0], self.cacherelu1)
            ddrop1 = dropout_backward(drelu1, self.cachedrop1)
            dlayer1 = affine_backward(ddrop1, self.cachelayer1)
            self.params['W1'] -= learning_rate * dlayer1[1] + self.reg * self.params['W1']
            self.params['b1'] -= learning_rate * dlayer1[2] + self.reg * self.params['b1']

        if self.activition == 'Sigmoid':
            # print("out", ":", self.out, end='\n' * 2)
            # print('dout', "\n", self.dout, end='\n' * 2)
            dsig3 = sigmoid_backward(self.dout, self.cachesig3)
            ddrop3 = dropout_backward(dsig3, self.cachedrop3)
            dlayer3 = affine_backward(ddrop3, self.cachelayer3)
            self.params['W3'] -= learning_rate * dlayer3[1] + self.reg * self.params['W3']
            self.params['b3'] -= learning_rate * dlayer3[2] + self.reg * self.params['b3']
            # print("dlayer3", "\n", dlayer3[0], end="\n" * 2)
            # print("dW3", "\n", dlayer3[1], end="\n" * 2)
            dsig2 = sigmoid_backward(dlayer3[0], self.cachesig2)
            ddrop2 = dropout_backward(dsig2, self.cachedrop2)
            dlayer2 = affine_backward(ddrop2, self.cachelayer2)
            self.params['W2'] -= learning_rate * dlayer2[1] + self.reg * self.params['W2']
            self.params['b2'] -= learning_rate * dlayer2[2] + self.reg * self.params['b2']
            dsig1 = relu_backward(dlayer2[0], self.cachesig1)
            ddrop1 = dropout_backward(dsig1, self.cachedrop1)
            dlayer1 = affine_backward(ddrop1, self.cachelayer1)
            self.params['W1'] -= learning_rate * dlayer1[1] + self.reg * self.params['W1']
            self.params['b1'] -= learning_rate * dlayer1[2] + self.reg * self.params['b1']
        return dlayer1[0]

    def loss(self, y, mode='test'):
        if self.activition == 'ReLU':
            input = self.relu3
        if self.activition == 'Sigmoid':
            input = self.sig3
        if self.losstype == 'softmax_loss':
            loss, self.dout = softmax_loss(input, y)
        if self.losstype == 'onehot_loss':
            loss, self.dout = onehot_loss(input, y)
        return loss + self.reg * self.weight_sum


class ConvNet(object):  # conv1-relu1-pool1-conv2-relu2-pool2-conv3-relu3-conv3-FullConnectedLayer-softmax
    def __init__(self, input_dim=(3, 32, 32), connect_conv=0, use_batchnorm=False,
                 weight_scale=1e-1):
        self.use_connect_conv = connect_conv > 0
        self.use_batchnorm = use_batchnorm
        self.input_dim = input_dim
        self.W1 = weight_scale * np.random.normal(loc=0.5, scale=1, size=(10, self.input_dim[0], 3, 3))
        self.b1 = np.zeros(10)
        self.W2 = weight_scale * np.random.normal(loc=0.5, scale=1, size=(10, 10, 3, 3))
        self.b2 = np.zeros(10)
        self.W3 = weight_scale * np.random.normal(loc=0.5, scale=1, size=(10, 10, 3, 3))
        self.b3 = np.zeros(10)
        self.convparam = {}  # 'stride' 'pad'
        self.convparam['stride'] = int(1)
        self.convparam['pad'] = int(1)
        self.poolparam = {}  # stride height width
        self.poolparam['stride'] = int(2)
        self.poolparam['height'] = int(2)
        self.poolparam['width'] = int(2)

    def forward(self, X):
        self.input = X  # 3*32*32
        self.conv1, self.cacheconv1 = conv_forward_naive(self.input, self.W1, self.b1, self.convparam)  # 10*32*32
        self.relu1, self.cacherelu1 = relu_forward(self.conv1)  # 10*32*32
        self.pool1, self.cachepool1 = max_pool_forward_naive(self.relu1, self.poolparam)  # 10*16*16
        self.conv2, self.cacheconv2 = conv_forward_naive(self.pool1, self.W2, self.b2, self.convparam)  # 10*16*16
        self.relu2, self.cacherelu2 = relu_forward(self.conv2)  # 10*16*16
        self.pool2, self.cachepool2 = max_pool_forward_naive(self.relu2, self.poolparam)  # 10*8*8
        self.conv3, self.cacheconv3 = conv_forward_naive(self.pool2, self.W3, self.b3, self.convparam)  # 10*8*8
        self.relu3, self.cacherelu3 = relu_forward(self.conv3)  # 10*8*8
        self.pool3, self.cachepool3 = max_pool_forward_naive(self.relu3, self.poolparam)  # 10*4*4
        self.out = self.pool3.reshape((X.shape[0], -1))  # 160
        '''
        plt.subplot(1,2,1)
        plt.imshow(self.input[0][0])
        plt.subplot(1,2,2)
        plt.imshow(self.input[0][1])
        plt.show()
        '''
        return self.out

    def backward(self, dout, learning_rate=0):
        dout = dout.reshape((dout.shape[0], 10, 4, 4))
        dpool3 = max_pool_backward_naive(dout, self.cachepool3)
        drelu3 = relu_backward(dpool3, self.cacherelu3)
        dconv3 = conv_backward_naive(drelu3, self.cacheconv3)
        self.W3 -= learning_rate * dconv3[1]
        self.b3 -= learning_rate * dconv3[2]
        dpool2 = max_pool_backward_naive(dconv3[0], self.cachepool2)
        drelu2 = relu_backward(dpool2, self.cacherelu2)
        dconv2 = conv_backward_naive(drelu2, self.cacheconv2)
        self.W2 -= learning_rate * dconv2[1]
        self.b2 -= learning_rate * dconv2[2]
        dpool1 = max_pool_backward_naive(dconv2[0], self.cachepool1)
        drelu1 = relu_backward(dpool1, self.cacherelu1)
        dconv1 = conv_backward_naive(drelu1, self.cacheconv1)
        self.W1 -= learning_rate * dconv1[1]
        self.b1 -= learning_rate * dconv1[2]


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
        # self.cnn.backward(dann)

    def train(self, X, y, iter_times=20, batch_size=4, draw=False):  # train the networks
        num = X.shape[0]
        loss_history = []
        for i in range(iter_times):
            select = np.random.choice(num, batch_size, replace=False).astype(np.int32)
            select = np.arange(batch_size)
            x_batch = np.zeros((batch_size, 3, 32, 32))
            y_batch = np.zeros(batch_size)
            for k in range(batch_size):
                x_batch[k] = X[select[k]]
                y_batch[k] = y[select[k]]
            y_batch = y_batch.astype(np.int32)
            loss_history.append(self.loss(x_batch, y_batch))
            '''
            print(self.ann.relu3[:4])
            print(self.ann.out[:4])
            print(y_batch[:4], end="\n" * 2)
            '''
            print("next")
            self.backward()
        if draw == True:
            plt.ylabel("loss")
            plt.xlabel("iter_time")
            plt.plot(np.arange(iter_times), loss_history)
            plt.show()

    def score(self, X, y):  # give an accuracy of a certain dataset
        num = X.shape[0]
        x_batch = np.zeros((num, 3, 32, 32))
        y_batch = np.zeros(num)
        for k in range(num):
            x_batch[k] = X[k]
            y_batch[k] = y[k]
        y_batch = y_batch.astype(np.int32)
        ans = self.forward(x_batch)
        print(ans, y_batch)
        return np.sum(ans == y_batch) / num


nn = NeuralNetwork()
nn.train(X[0][:4], y[0][:4], iter_times=4000, draw=False)  # 6 9
# print(nn.score(X[0][:20], y[0][:20]))
