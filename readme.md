
# D3A-TS Denoising-Driven Data Augmentation in Time Series

It has been demonstrated that the amount of data is crucial in data-driven machine learning methods. Data is always valuable, but in some tasks, it is almost like gold. This occurs in tasks where data is scarce or very expensive to obtain, such as predictive maintenance, where faults are rare. In this context, a mechanism to generate synthetic data is very useful.While fields such as Computer Vision or Natural Language Processing have extensively explored synthetic data generation with promising results, other domains like time-series have received less attention. This work specifically focuses on studying and analyzing the use of different techniques for data augmentation in time-series for classification and regression problems. The proposed approach involves the use of diffusion probabilistic models, which have recently achieved successful results in the field of Image Processing, for data augmentation in time-series. Additionally, it suggests the use of a set of meta-attributes to condition the data augmentation process. The results highlight the high utility of this methodology in creating synthetic data to train classification and regression models. To contrast the results, six different datasets from diverse domains were employed, showcasing versatility in terms of input size and output types. 

This repository contains all the source code needed to reproduce the experiments or review the results obtained (results folder).


## Installation

After clone this repository create a virtual enviroment with conda or virtualenv and install the requirements: 

```
pip install -r requirements
```

You are ready to start executing commands.


## Repository Structure

The structure of this repository is organized as follows:
```
d3a-ts/
    │
    ├─ data/                            # Links to datasets used in this work
    │
    ├─ results/                         # Results files, logs, and configuration of the experiments
    │
    +─ src/                            
       │ 
       ├─ d3a
       │   │
       │   ├─ meta.py                   # Meta attribute generation
       │   │
       │   ├─ net.py                    # Implementation of the diffusion and autoencoders for data augmentation
       │   │
       │   ├─ search.py                 # Command to execute the experiments
       │   │
       │   +─ notebooks        
       │        │
       │        ├─ Figures.ipynb        # Creation of some figures used in the paper
       │        │
       │        ├─ Results review.ipynb # Creation of graph results included in the paper
       │        │
       │        +─ bayesiantests.py     # Code to execute the Bayesian tests
       │
       ├─ data                          # Code to load and create the data generations of each dataset
       │
       +─ nets                          # Network architectures used in the paper
```

## Executing experiments

To run the experiments, utilize the search.py command:

```
usage: search.py \[-h\] -d DIR -g {0,1}

optional arguments:
  -h, --help            show this help message and exit
  -d DIR, --dir DIR     Directory where found the params file
  -g {0,1}, --gpu {0,1}
                        GPU to use \[0, 1\]
```

The command requires a directory path where the *params.json* file is located, defining the parameters for the experiments. For instance:

```
python d3a/search.py -d ../results/pronostia/128/nr_cond_bilstm_experiment_1 -g 0
```

In this example, the command uses GPU 0 and expects to find the *params.json* file in the directory  *../results/pronostia/128/nr_cond_bilstm_experiment_1*.

If the directory already contains result files, it will load the results without training the corresponding model. To train new models, create a new directory containing only the params.json file.

The params.json file follows the structure outlined below:

```
{
    "model": "\[bilstm|mscnn\]",
    "learning_rate": \[float\],
    "name": "\[experiment name\]",
    "package": "data.\[data package name\]",
    "generator": "\[generator class name\]",
    "save_memory": \[true|false\],
    "net_config": {
      "net": \[dict\],
      "noise_rates": \[list\]

    },
    "denoising_net": \[dict\],
    "s2a_net": \[dict\]
 
}
```

This file defines the architecture used to train the final classifier or regressor (*model* and *net_config*), as well as how to read the data (*package* and *generator*). It specifies the denoising model architecture (denoising_net) and the architecture of the network to estimate the attributes (*s2a_net*).

## Methodology

This work involves generating meta-attributes $\overline{a}^{(i)}$ for each sample $x^{(i)}$. For every segment of 32 data points, 15 meta-attributes are extracted, resulting in a vector $\overline{a}^{(i)}$ with 60 elements (30 for the shares dataset). This process is computationally intensive.

To manage the complexity, the meta-attributes vectors $\overline{a}^{(i)}$ are approximated using a fully connected neural network $\mathcal{A}_{\psi}$ with three hidden layers (32, 64, and 128 neurons) and hyperbolic tangent (tanh) activation function.

The model $\mathcal{A}_{\psi}$ is utilized during the training of the denoising model $\mathcal{H}_{\phi}$ to remove noise from a noisy sample. The architecture of $\mathcal{H}_{\phi}$ consists of 4 downsampling blocks and 4 upsampling blocks, with two additional blocks applied after downsampling. Each block applies one-dimensional operations, and the time step and attribute embeddings are integrated using sum and concatenation operations. Lastly, two residual blocks are applied. Embeddings are created using a fully connected layer and reshaped to match the dimensions of the former one-dimensional convolutional layer.

![Denoising model architecture](https://filedn.eu/loXpIrTnEJNpqcMnQVXex9m/arch_net.svg)

In the final step, the model $\mathcal{H}_{\phi}$ is employed to augment data during the training of the classification or regression task. Notably, this ultimate model is exclusively trained using synthetic data. Training variations include 1, 2, and 3 denoising steps, with various noise rates introduced to the samples before denoising. Both the number of denoising steps and the noise rate serve as hyper-parameters. The training procedure employs early stopping as the stopping criterion, halting training when the model's performance on a validation set ceases to improve after a predefined number of steps. The process is illustrated in the following figure:

![Training graph](https://filedn.eu/loXpIrTnEJNpqcMnQVXex9m/training.svg)

The graph illustrates the training process. $\mathcal{A}_{\psi}$ represents the network used to predict the meta-attributes vector $\overline{a}$ from the training raw data. $\mathcal{M}$ denotes the process that introduces normal noise to the raw samples, while $\mathcal{H}_{\phi}$ represents the denoising network responsible for generating the synthetic samples. The model $\mathcal{F}$ is then trained using these synthetic samples and validated against the raw samples from the test set.

## Results

The table compares the best mean results obtained by applying denoising conditioned data augmentation with the mean performance of models trained using raw data. The number in brackets refers to the number of denoising steps applied.

| Dataset        | Net    | Raw                 | AE                      | DPM                     |
|:---------------|:-------|:--------------------|:------------------------|:------------------------|
| ecg5k          | bilstm | 0.5211 ±   0.1178   | 0.3609 ±   0.0157 \[1\]   | **0.3235 ±   0.0085** \[1\]   |
| ecg5k          | mscnn  | 0.7591 ±   0.3981   | 0.5204 ±   0.0325 \[3\]   | **0.4272 ±   0.0205** \[1\]   |
| human_activity | bilstm | 1.2370 ±   0.2486   | 1.1023 ±   0.0016 \[1\]   | **1.0897 ±   0.0011** \[2\]   |
| human_activity | mscnn  | 1.2764 ±   0.1213   | **1.2015 ±   0.0314** \[1\]   | 1.2758 ±   0.0120 \[2\]   |
| ncmapss        | bilstm | 252.9270 ±  13.9745 | **242.1542 ±   0.0000** \[2\] | 246.7824 ±  16.9870 \[2\] |
| ncmapss        | mscnn  | 459.5337 ± 165.5178 | 324.1614 ±  16.9442 \[2\] | **266.0854 ±  13.5774** \[1\] |
| pronostia      | bilstm | 0.0720 ±   0.0063   | **0.0648 ±   0.0030** \[3\]   | 0.0662 ±   0.0007 \[1\]   |
| pronostia      | mscnn  | 0.0614 ±   0.0044   | **0.0487 ±   0.0030** \[1\]   | 0.0522 ±   0.0029 \[3\]   |
| shares         | bilstm | 0.3480 ±   0.0266   | 0.3435 ±   0.0142 \[3\]   | **0.2947 ±   0.0495** \[1\]   |
| shares         | mscnn  | 1.2505 ±   0.8296   | 0.3113 ±   0.0157 \[3\]   | **0.2153 ±   0.0175** \[3\]   |
| wine           | bilstm | 0.4097 ±   0.0203   | 0.3850 ±   0.0074 \[3\]   | **0.3432 ±   0.0043** \[1\]   |
| wine           | mscnn  | 1.3288 ±   0.5098   | **0.3298 ±   0.0269** \[2\]   | 0.4529 ±   0.0000 \[2\]   |

The Bayesian signed-rank test provides confirmation that utilizing Diffusion Probabilistic Models (DPM) conditioned with the proposed meta-attributes in this study is highly beneficial for data augmentation in time-series, particularly in the contexts of both classification and regression:

![enter image description here](https://filedn.eu/loXpIrTnEJNpqcMnQVXex9m/denoise_vs_raw.svg)


## ACKNOWLEDGMENT

This work has been supported by Grant PID2019-109152GBI00/AEI/10.13039/501100011033 (Agencia Estatal de Investigacion), Spain and by the Ministry of Science and Education of Spain through the national program "Ayudas para contratos para la formacion de investigadores en empresas (DIN2019)", of State Programme of Science Research and Innovations 2017-2020.