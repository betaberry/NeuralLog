import os
import sys


import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.metrics import classification_report

sys.path.append("../")

import pickle
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import SparseCategoricalCrossentropy
# from official.nlp import optimization


from neurallog.models import NeuralLog
from neurallog import data_loader
# from neurallog.utils import classification_report


# 我的
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.optimizers import Adam

embed_dim = 768  # Embedding size for each token
max_len = 75
# max_len = 50

class BatchGenerator(Sequence):

    def __init__(self, X, Y, batch_size):
        self.X, self.Y = X, Y
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.X) / float(self.batch_size)))

    def __getitem__(self, idx):
        # print(self.batch_size)
        dummy = np.zeros(shape=(embed_dim,))
        x = self.X[idx * self.batch_size:min((idx + 1) * self.batch_size, len(self.X))]
        X = np.zeros((len(x), max_len, embed_dim))
        Y = np.zeros((len(x), 2))
        item_count = 0
        for i in range(idx * self.batch_size, min((idx + 1) * self.batch_size, len(self.X))):
            x = self.X[i]
            if len(x) > max_len:
                x = x[-max_len:]
            x = np.pad(np.array(x), pad_width=((max_len - len(x), 0), (0, 0)), mode='constant',
                       constant_values=0)
            X[item_count] = np.reshape(x, [max_len, embed_dim])
            Y[item_count] = self.Y[i]
            item_count += 1
        return X[:], Y[:, 0]

# 太老了，用gpt生成了一个新点的，参见下一个函数
def train_generator(training_generator, validate_generator, num_train_samples, num_val_samples, batch_size,
                    epoch_num, model_name=None):
    epochs = epoch_num
    steps_per_epoch = num_train_samples
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1 * num_train_steps)

    init_lr = 3e-4
    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')

    loss_object = SparseCategoricalCrossentropy()

    model = NeuralLog(768, ff_dim=2048, max_len=75, num_heads=12, dropout=0.1)

    # model.load_weights("hdfs_transformer.hdf5")

    model.compile(loss=loss_object, metrics=['accuracy'],
                  optimizer=optimizer)

    print(model.summary())

    # checkpoint
    filepath = model_name
    checkpoint = ModelCheckpoint(filepath,
                                 monitor='val_accuracy',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max',
                                 save_weights_only=True)
    early_stop = EarlyStopping(
        monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto',
        baseline=None, restore_best_weights=True
    )
    callbacks_list = [checkpoint, early_stop]

    model.fit_generator(generator=training_generator,
                        steps_per_epoch=int(num_train_samples / batch_size),
                        epochs=epoch_num,
                        verbose=1,
                        validation_data=validate_generator,
                        validation_steps=int(num_val_samples / batch_size),
                        workers=16,
                        max_queue_size=32,
                        callbacks=callbacks_list,
                        shuffle=True
                        )
    return model


'''
training_generator: 一个 Python 生成器，它负责在每个训练步骤提供批量的训练数据。
validate_generator: 一个 Python 生成器，它负责在每个验证步骤提供批量的验证数据。
num_train_samples: 训练样本的总数。
num_val_samples: 验证样本的总数。
batch_size: 每个批次的样本数。
epoch_num: 训练过程中的总周期数。
model_name: 用于保存训练后的最佳模型的文件名。
'''
def train_generator_2(training_generator, validate_generator, num_train_samples, num_val_samples, batch_size,
                    epoch_num, model_name=None):
    epochs = epoch_num
    steps_per_epoch = num_train_samples // batch_size
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1 * num_train_steps)

    # 创建一个学习率衰减计划
    lr_schedule = PolynomialDecay(
        initial_learning_rate=3e-4,
        decay_steps=num_train_steps,
        end_learning_rate=0.0,
        power=1.0,
        cycle=False
    )

    # 创建AdamW优化器
    optimizer = Adam(learning_rate=lr_schedule)

    loss_object = SparseCategoricalCrossentropy()

    # 假设NeuralLog是一个预定义的模型结构
    model = NeuralLog(768, ff_dim=2048, max_len=75, num_heads=12, dropout=0.1)

    # 如果有预训练的权重，可以取消注释下面的代码加载它们
    # model.load_weights("hdfs_transformer.hdf5")

    model.compile(loss=loss_object, metrics=['accuracy'], optimizer=optimizer, run_eagerly=True)

    print(model.summary())

    # 保存最好的模型
    filepath = model_name if model_name else "best_model.hdf5"
    checkpoint = ModelCheckpoint(filepath,
                                 monitor='val_accuracy',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max',
                                 save_weights_only=True)
    early_stop = EarlyStopping(
        monitor='val_loss', patience=5, verbose=1, mode='auto',
        restore_best_weights=True
    )
    callbacks_list = [checkpoint, early_stop]

    history = model.fit(
        x=training_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validate_generator,
        validation_steps=num_val_samples // batch_size,
        verbose=1,
        callbacks=callbacks_list,
        workers=16,
        max_queue_size=32,
        shuffle=True
    )
    return model



def train(X, Y, epoch_num, batch_size, model_file=None):
    X, Y = shuffle(X, Y)
    n_samples = len(X)
    train_x, train_y = X[:int(n_samples * 90 / 100)], Y[:int(n_samples * 90 / 100)]
    val_x, val_y = X[int(n_samples * 90 / 100):], Y[int(n_samples * 90 / 100):]

    training_generator, num_train_samples = BatchGenerator(train_x, train_y, batch_size), len(train_x)
    validate_generator, num_val_samples = BatchGenerator(val_x, val_y, batch_size), len(val_x)

    print("Number of training samples: {0} - Number of validating samples: {1}".format(num_train_samples,
                                                                                       num_val_samples))

    model = train_generator_2(training_generator, validate_generator, num_train_samples, num_val_samples, batch_size,
                            epoch_num, model_name=model_file)

    return model


def test_model(model, x, y, batch_size):
    # 来自sklearn
    x, y = shuffle(x, y)
    # 把x，y的数量调整为batch_size的整数倍
    x, y = x[: len(x) // batch_size * batch_size], y[: len(y) // batch_size * batch_size]
    # batch生成器
    test_loader = BatchGenerator(x, y, batch_size)
    prediction = model.predict_generator(test_loader, steps=(len(x) // batch_size), workers=16, max_queue_size=32,
                                         verbose=1)
    prediction = np.argmax(prediction, axis=1)
    y = y[:len(prediction)]
    report = classification_report(np.array(y), prediction)
    print(report)



def get_file_name(path):
    base_name = os.path.basename(path)
    file_name, _ = os.path.splitext(base_name)
    return file_name

if __name__ == '__main__':
    # log_file = "../logs/BGL-small.log"
    # log_file = "../logs/BGL-small.log"
    # log_file = "../logs/Spirit1G.log"
    log_file = "../logs/Spirit-small.log"
    (x_tr, y_tr), (x_te, y_te) = data_loader.load_supercomputers(
        log_file, train_ratio=0.8, windows_size=20,
        step_size=20, e_type='bert', mode='balance')

    epoch = 2
    batch_size = 64 # 默认256

    file_name = get_file_name(log_file)

    model = train(x_tr, y_tr, epoch, batch_size, file_name + "_transformer.hdf5")
    test_model(model, x_te, y_te, batch_size=1024)
