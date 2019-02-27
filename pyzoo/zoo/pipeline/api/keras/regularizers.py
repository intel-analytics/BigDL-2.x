from bigdl.optim.optimizer import *


def l1(l1=0.01):
    return L1Regularizer(l1=l1)


def l2(l2=0.01):
    return L2Regularizer(l2=l2)


def l1l2(l1=0.01, l2=0.01):
    return L1L2Regularizer(l1=l1, l2=l2)