from dcgan.trainer import Trainer
from dcgan.generator import Generator
from dcgan.discreminator import Discriminator

from sklearn.datasets.mldata import fetch_mldata
import numpy as np
import pandas as pd

import pickle

if __name__ == '__main__':
    gen = Generator(100)
    dis = Discriminator()

    data = fetch_mldata('mnist-original', data_home=".")
    X = data['data']
    n_train = X.shape[0]
    X = np.array(X, dtype=np.float32)
    X /= 255.
    X = X.reshape(n_train, 1, 28, 28)

    trainer = Trainer(gen, dis)

    trainer.fit(X, batch_size=1000, epochs=1000)

    df_loss = pd.DataFrame(trainer.loss)
    df_loss.to_csv('loss.csv')

    gen.to_cpu()
    dis.to_cpu()

    with open('generator.model', 'wb') as w:
        pickle.dump(gen, w)

    with open('discriminator.model', 'wb') as w:
        pickle.dump(dis, w)
