# RouteNet-Fermi

This is the official implementation of RouteNet-Fermi, a pioneering GNN architecture designed to model computer 
networks. RouteNet-Fermi supports complex traffic models, multi-queue scheduling policies, routing policies and 
can provide accurate estimates in networks not seen in the training phase.

**If you decide to apply the concepts presented or base on the provided code, please do refer our paper.**

```
@article{ferriol2022routenet,
  title={RouteNet-Fermi: Network Modeling with Graph Neural Networks},
  author={Ferriol-Galm{\'e}s, Miquel and Paillisse, Jordi and Su{\'a}rez-Varela, Jos{\'e} and Rusek, Krzysztof and Xiao, Shihan and Shi, Xiang and Cheng, Xiangle and Barlet-Ros, Pere and Cabellos-Aparicio, Albert},
  journal={arXiv preprint arXiv:2212.12070},
  year={2022}
}
```

## Quick Start
### Project structure

The project is divided into three main blocks: scheduling, traffic models, and scalability. Each block has its own 
directory and contains its own files. In each directory we can find four main files:
- `main.py`: contains the code for the training/validation of the model.
- `ckpt_dir`: contains the trained models for the specific experiment.
- `predict.py`: contains the code for the predictions. It automatically loads the best-trained model and 
saves its predictions in a `predictions.npy` file that can be read using NumPy.

The project also contains some auxiliary files like `datanetAPI.py` used to read the different datasets and the 
`data_generator.py` that is used to convert the samples provided by the dataset API into the graph that is taken as
input by the model, and a `generate_dataframe.py` which saves the data in a Pandas Dataframe used to compute the scores.

You can find more information about the reproducibility of the experiments inside each one of the directories 
([Traffic Models](/traffic_models/README.md), [Scheduling](/scheduling/README.md),  [Scalability](/scalability/README.md),
[All Mixed](/all_mixed/README.md), [Testbed](/testbed/README.md), [Real Traffic](/real_traffic/README.md), [FatTree](/fat_tree/README.md)).

## Main Contributors
#### M. Ferriol-Galmés, J. Paillisé-Vilanovs, J. Suárez-Varela, P. Barlet-Ros, A. Cabellos-Aparicio.

[Barcelona Neural Networking center](https://bnn.upc.edu/), Universitat Politècnica de Catalunya

#### Do you want to contribute to this open-source project? Please, read our guidelines on [How to contribute](CONTRIBUTING.md)

## License
See [LICENSE](LICENSE) for full of the license text.

```
Copyright 2023 Universitat Politècnica de Catalunya

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
