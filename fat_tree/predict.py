import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import os
import re
import numpy as np
import tensorflow as tf
from data_generator import input_fn

import sys

sys.path.append('../')
from delay_model import RouteNet_Fermi

for N in [16, 64, 128]:
    print(f"Predicting for N = {N}...")
    TEST_PATH = f'/data/TON23/fat{N}/test'

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model = RouteNet_Fermi()

    loss_object = tf.keras.losses.MeanAbsolutePercentageError()

    model.compile(loss=loss_object,
                  optimizer=optimizer,
                  run_eagerly=False)

    best = None
    best_mre = float('inf')

    ckpt_dir = f'./ckpt_dir_{N}'

    for f in os.listdir(ckpt_dir):
        if os.path.isfile(os.path.join(ckpt_dir, f)):
            reg = re.findall("\d+\.\d+", f)
            if len(reg) > 0:
                mre = float(reg[0])
                if mre <= best_mre:
                    best = f.replace('.index', '')
                    best = best.replace('.data', '')
                    best = best.replace('-00000-of-00001', '')
                    best_mre = mre

    print("BEST CHECKOINT FOUND : {}".format(best))

    model.load_weights(os.path.join(ckpt_dir, best))

    ds_test = input_fn(TEST_PATH, shuffle=False)
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    predictions = model.predict(ds_test, verbose=1)

    np.save(f'predictions_delay_{N}.npy', np.squeeze(predictions))
