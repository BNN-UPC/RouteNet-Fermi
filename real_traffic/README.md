# How to reproduce the experiments
This model has been trained to predict the delay. In this directory, you will find the needed files to train/validate and predict the metrics.

## Dependencies

**Recommended: Python 3.7**

Please, ensure you use Python 3.7. Otherwise, we do not guarantee the correct installation of dependencies.

You can install all the dependencies by running the following commands.
```
pip install -r requirements.txt
```

## Download the data
You can download the datasets for this particular experiment here:
- [Real Traces Dataset](https://bnn.upc.edu/download/dataset-v6-real-traces/)

Otherwise, you can download the datasets using the following command:
```
wget -O real_traces.zip https://bnn.upc.edu/download/dataset-v6-real-traces/
```

Note that this dataset is a zip file, so you need to decompress it first. Also, these experiments suppose that the data 
is in a directory file called `data` located at the root of the repository. If you want to change this, you can change the
path sent as input to the `input_fn` function.

## Training and Validation
If you want to train the model, you can use the following command:
```
python main.py
```

If everything has been done correctly, you should see the following output:

```
Starting training from scratch...
Epoch 1/200
36/4000 [..............................] - ETA: 3:06 - loss: 156.5287
```

The model will train during 50 epochs. You can change this number by changing the `epochs` parameter in the `fit` function.

Finally, the models will be saved in a `ckp_dir` directory. You can change this path by changing the `ckp_dir` variable. 

## Prediction
We have already provided the trained models for each one of the experiments. If you want, you can skip the Training and Validation
part and just use the following command to predict the metrics:
```
python predict.py
```
This script will, first, load the best model located in the `ckp_dir` directory. Then, it will evaluate the metrics for
each one of the sample datasets and finally, store the results in a `predictions.npy` file.

Again, if you configured everything correctly, you should see something like this:
```
BEST CHECKOINT FOUND: 44-4.62
    114/Unknown - 6s 33ms/step - loss: 3.5782
```

## License
See [LICENSE](LICENSE) for full of the license text.

```
Copyright Copyright 2023 Universitat Politècnica de Catalunya

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```