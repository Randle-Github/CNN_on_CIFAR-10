import numpy as np


def affine_forward(x, w, b):  # (batch_size,n1) (n1,n2) (n2)
    out = np.dot(x, w) + b  # (batch_size,n2)
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):  # (batch_size,n2)
    x, w, b = cache
    dx, dw, db = np.dot(dout, w.T), np.dot(x.T, dout), np.sum(dout, axis=0)
    return dx, dw, db


def sigmoid_forward(x):  # (batch_size,n)
    cache = x
    out = np.ones(np.shape(x)) / (1 + np.exp(-x))
    return out, cache


def sigmoid_backward(dout, cache):
    out = np.ones(np.shape(cache)) / (1 + np.exp(-cache))
    dcache = dout * out * (1 - out)
    return dcache


def relu_forward(x):  # (batch_size,n)
    cache = x
    out = x
    out[x < 0] = 0
    return out, cache


def relu_backward(dout, cache):
    dcache = dout
    dcache[cache < 0] = 0
    return dcache


def onehot_loss(X, y):
    dx = 2 * X
    loss = np.sum(X * X)
    for i in range(X.shape[0]):
        loss += -X[i][y[i]] ** 2 + (1 - X[i][y[i]]) ** 2
        dx[i][y[i]] = 2 * (X[i][y[i]] - 1)
    return loss / X.shape[0], dx


def softmax_loss(X, y):  # y: (batch_size) the label
    loss = 0
    x = X - np.max(X, axis=1).reshape(X.shape[0], 1)
    p = np.exp(x)
    p = p / np.sum(p, axis=1).reshape(p.shape[0], 1)
    dx = p
    for i in range(x.shape[0]):
        # loss += -np.log(p[i][y[i]])
        dx[i][y[i]] = p[i][y[i]] - 1
    return loss / X.shape[0], dx


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


def batchnorm_forward(x, gamma, beta, bn_param):
    pass

def batchnorm_backward(dout, cache):
    pass


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
