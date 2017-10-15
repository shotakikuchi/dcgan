# coding: utf-8

from chainer import Chain
import chainer.functions as F
import chainer.links as L

import numpy as np


class Discriminator(Chain):
    """入力された画像が偽物かどうかを判定する判別器
    """

    def __init__(self, ):
        super(Discriminator, self).__init__(
            c1=L.Convolution2D(1, 64, 3, stride=3, pad=1, ),
            c2=L.Convolution2D(64, 128, 2, stride=2, pad=1, ),
            c3=L.Convolution2D(128, 256, 2, stride=2, pad=1, ),
            c4=L.Convolution2D(256, 512, 2, stride=2, pad=1, ),

            l1=L.Linear(3 * 3 * 512, 2),

            bn1=L.BatchNormalization(128),
            bn2=L.BatchNormalization(256),
            bn3=L.BatchNormalization(512),
        )

    def __call__(self, x):
        """判別関数．
        return 二次元のVariable
        """
        h = F.relu(self.c1(x))
        h = F.relu(self.bn1(self.c2(h)))
        h = F.relu(self.bn2(self.c3(h)))
        h = F.relu(self.bn3(self.c4(h)))
        y = self.l1(h)
        return y
