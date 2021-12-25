import numpy as np
import cv2


class linear_classifier(object):
    def __init__(self):
        pass

    def loss(self, x_batch, y_batch, reg):
        pass

    def train(self, X, y, num_iters=5000, batch_size=500, reg=0.1, learning_rate=0.000000001):
        self.class_num = np.max(y) + 1
        self.num, self.dimension = np.shape(X)  # get the shape of X
        self.data = np.column_stack((X, np.ones(self.num)))  # read data
        self.dimension += 1  # add a constant bias
        self.label = y
        self.W = 0.00005 * np.random.random((self.dimension, self.class_num))
        loss_history = []
        for i in range(num_iters):
            select = np.random.choice(self.num, batch_size, replace=False)
            x_batch = self.data[select, :]  # get a batch of X
            y_batch = np.zeros((batch_size, 1))
            for tot in range(batch_size):
                y_batch[tot] = int(self.label[select[tot]])  # get a batch of y
            loss, grad = self.loss(x_batch, y_batch, reg)
            loss_history.append(loss)
            self.W -= grad * learning_rate
        plt.plot(np.arange(num_iters), loss_history)
        plt.title = "loss_function"
        plt.xlabel("iter_times")
        plt.ylabel("loss")
        plt.show()

    def predict(self, X):
        num, _ = np.shape(X)
        score = np.dot(X, self.W)
        label = np.zeros((num, 1))
        for i in range(num):
            label[i] = np.argmax(score[i])
        return label

    def score(self, X, y):
        self.data = np.column_stack((X, np.ones(self.num)))  # read data
        label = self.predict(self.data)
        num, _ = np.shape(self.data)
        acc = 0
        for i in range(num):
            if y[i] == label[i]:
                acc += 1
        return acc / num


class soft_max_classifier(linear_classifier):
    def loss(self, x_batch, y_batch, reg=1):
        loss = reg * np.sum(self.W * self.W)
        num, _ = np.shape(x_batch)  # self.num
        score = np.dot(x_batch, self.W)  # self.num * self.class_num
        p = np.exp(score)
        for i in range(num):
            p[i] /= np.sum(np.exp(score[i]))  # probability
        d_W = reg * self.W
        d_score = p
        for i in range(num):
            loss += -np.log(p[i][int(y_batch[i])])  # loss = - p * log(p)
            d_score[i][int(y_batch[i])] = p[i][int(y_batch[i])] - 1  #
        loss /= num
        d_W += np.dot(x_batch.T, d_score)
        return loss, d_W


class multi_svm_classifier(linear_classifier):
    def loss(self, x_batch, y_batch, reg=1):
        loss = reg * np.sum(self.W * self.W)
        num, _ = np.shape(x_batch)
        score = np.dot(x_batch, self.W)  # self.num * self.class_num
        modify = np.zeros((num, self.class_num))  # self.num * 1
        for i in range(num):
            modify[i] = score[i][int(y_batch[i])]
        score = score - modify + 10
        score = np.where(score > 0, score, 0)
        d_W = reg * self.W  # the partial
        for i in range(num):
            for j in range(self.class_num):
                if score[i][j] > 0 and y_batch[i] != j:
                    d_W[:, j] += x_batch[i, :].T  # partial
                if y_batch[i] == j:
                    d_W[:, j] -= (self.class_num - 1) * x_batch[i, :]  # correction
        d_W /= num
        loss += np.sum(score) / num
        return loss, d_W


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


import matplotlib.pyplot as plt

length = 3072
# 加载测试集
dict_data = []
dict_data.append(unpickle("data_batch_1"))
dict_data.append(unpickle("data_batch_2"))
dict_data.append(unpickle("data_batch_3"))
dict_data.append(unpickle("data_batch_4"))
dict_data.append(unpickle("data_batch_5"))

for i in range(5):
    for j in range(len(dict_data[i][b'data'])):
        dict_data[i][b'data'][j] = dict_data[i][b'data'][j].flatten()  # 3072

svm = multi_svm_classifier()
softmax = soft_max_classifier()
softmax.train(dict_data[0][b'data'], dict_data[0][b'labels'])
print(softmax.score(dict_data[3][b'data'], dict_data[3][b'labels']))
'''
i=3349 # show a picture
image_m = np.reshape(dict_data[0][b'data'][i], (3, 32, 32))
print(dict_data[0].keys())
print(dict_data[0][b'filenames'][i])
r = image_m[0, :, :]
g = image_m[1, :, :]
b = image_m[2, :, :]
img23 = cv2.merge([r, g, b])
plt.figure()
plt.imshow(img23)
plt.show()
'''
