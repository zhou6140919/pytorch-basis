import torch
from torch.autograd import Variable

LR = 1e-3
x = torch.tensor([1., 2., 3., 4., 5., 6., 7., 8., 9., 11.])
y = torch.tensor([3., 5., 7., 9., 11., 14., 15., 18., 20., 23.])

x = Variable(x, requires_grad=True)
y = Variable(y, requires_grad=True)


t0 = torch.FloatTensor(torch.rand(1, 10))
t1 = torch.FloatTensor(10)

def hypothesis(x):
    y_pred = t0 + t1 * x
    return y_pred

def costfunc(y_pred, y):
    loss = (y_pred - y).pow(2).sum()/10
    return loss

for i in range(200):
    y_pred = hypothesis(x)
    loss = costfunc(y_pred, y)
    t0_grad = (y_pred - y).sum()/5
    t1_grad = ((y_pred - y) * x.t()).sum()/5
    t0 -= LR * t0_grad
    t1 -= LR * t1_grad
    print(i, loss.item())





