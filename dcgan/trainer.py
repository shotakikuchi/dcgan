# coding: utf-8

import chainer.functions as F
import numpy as np
from chainer import Variable, optimizers

import matplotlib.pyplot as plt


class Trainer(object):
    def __init__(self, gen, dis):
        self.gen = gen
        self.dis = dis
        self.z_dim = gen.z_dim

        self.X = None
        self.epochs = None
        self.batchsize = None
        self.plotting = None
        self.loss = None

    def fit(self, X, epochs=10, batch_size=1000, plotting=True):
        """
        generator と discriminator のトレーニングを行うメソッド

        :param X: 正しい画像. shape = (n_samples, width, height, channel)
        :param int epochs:
            画像全体を何回訓練させるかを表す.
        :param int batch_size:
            学習時のバッチ数.
        :param bool plotting:
            訓練中に生成画像をプロットするかどうか.
            true のときプロットを行う.
        :return:
        """

        self.X = X
        self.epochs = epochs
        self.batchsize = batch_size
        self.plotting = plotting

        n_train = X.shape[0]
        o_gen = optimizers.Adam()
        o_dis = optimizers.Adam()

        o_gen.setup(self.gen)
        o_dis.setup(self.dis)

        self.loss = []
        for epoch in range(epochs):
            perm = np.random.permutation(n_train)
            sum_loss_of_dis = np.float32(0)
            sum_loss_of_gen = np.float32(0)

            for i in range(int(n_train // batch_size)):
                # print('iter {i}'.format(**locals()))

                z = np.random.uniform(-1, 1, (batch_size, self.z_dim))
                z = z.astype(dtype=np.float32)
                z = Variable(z)

                x = self.gen(z)
                y1 = self.dis(x)

                # 答え合わせ
                # ジェネレーターとしては0と判別させたい（騙すことが目的）
                loss_gen = F.softmax_cross_entropy(
                    y1, Variable(np.zeros(batch_size, dtype=np.int32)))

                # 判別機としては1(偽物)と判別したい
                loss_dis = F.softmax_cross_entropy(
                    y1, Variable(np.ones(batch_size, dtype=np.int32)))

                # load true data form dataset
                idx = perm[i * batch_size:(i + 1) * batch_size]
                x_data = self.X[idx]
                x_data = Variable(x_data)

                y2 = self.dis(x_data)

                # 今度は正しい画像なので、0（正しい画像）と判別したい
                loss_dis += F.softmax_cross_entropy(
                    y2, Variable(np.zeros(batch_size, dtype=np.int32)))

                self.gen.zerograds()
                loss_gen.backward()
                o_gen.update()

                self.dis.zerograds()
                loss_dis.backward()
                o_dis.update()

                sum_loss_of_dis += loss_dis.data
                sum_loss_of_gen += loss_gen.data

            print(
                'epoch-{epoch}\tloss\tdiscreminator-{sum_loss_of_dis:.3f}\tgenerator-{sum_loss_of_gen:.3f}'.format(
                    **locals()))

            self.loss.append([sum_loss_of_gen, sum_loss_of_dis])

            if plotting:
                plt.figure(figsize=(12, 12))
                n_row = 3
                s = n_row ** 2
                z = Variable(np.random.uniform(-1, 1, 100 * s).reshape(-1, 100).astype(np.float32))
                x = self.gen(z)
                y = self.dis(x)
                y = F.softmax(y).data
                x = x.data.reshape(-1, 28, 28)
                for i, xx in enumerate(x):
                    plt.subplot(n_row, n_row, i + 1)
                    plt.imshow(xx, interpolation="nearest", cmap="gray")
                    plt.axis('off')
                    plt.title('True Prob {0:.3f}'.format(y[i][0]))
                plt.tight_layout()
                plt.savefig('epoch-{epoch}.png'.format(**locals()), dip=100)
                plt.close('all')

        print(self.loss)
