import os
import re
import json
import tensorflow as tf
from data_generator import input_fn

import sys

sys.path.append('../')
from delay_model import RouteNet_Fermi

seed = 0
results = {}

for i in range(5):
    for num_samples in [10000, 5000, 500, 100, 50, 25]:
        print("STARTING ITERATION {} WITH NUM SAMPLES {}".format(i, num_samples))
        TEST_PATH = '../data/all_mixed/test'

        ckpt_dir = './ckpt_dirs/ckpt_dir_all_{}_{}'.format(i, num_samples)

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        model = RouteNet_Fermi()

        loss_object = tf.keras.losses.MeanAbsolutePercentageError()

        model.compile(loss=loss_object,
                      optimizer=optimizer,
                      run_eagerly=False)

        best = None
        best_mre = float('inf')

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

        print("BEST CHECKOINT FOUND FOR ITERATION {} AND NUM SAMPLES {}: {}".format(i, num_samples, best))

        ds_test = input_fn(TEST_PATH, shuffle=False)
        ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
        model.load_weights(os.path.join(ckpt_dir, best))

        eval = model.evaluate(ds_test, return_dict=True, use_multiprocessing=True, workers=10)

        if num_samples not in results:
            results[num_samples] = {}
        results[num_samples][i] = eval
        with open('results_all.json', 'w') as fp:
            json.dump(results, fp)
        seed += 1
