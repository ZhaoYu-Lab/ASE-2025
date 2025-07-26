# ASE-2025
This is the data and code for Coding-Fuse submit to ASE-2025. 

## Instructions
This is the experimental process code and data of the proposed method Coding-Fuse and FMF method, which can be found in the code directory.

If you need to repeat the performance values ​​of the experimental results, we recommend running the test code directly, in each setcode parameter --test part. But before that, you need to (1) first copy the dataset folder to the directory at the same level as code in each fusion folder. (2) Secondly, import the code pre-trained models in the code folder, and we have prepared the file name for you, prepared code PTMs can be found at https://zenodo.org/records/15543665 (3) Finally, you need to import our trained test model (can be found at https://zenodo.org/records/15543870, https://zenodo.org/records/15544997 and https://zenodo.org/records/15546108) in saved_models in the code folder. You can get the same performance values ​​as us.

If you need to perform the entire experiment, that is, the entire process from training to testing, we recommend that you directly execute all the commands of setcode, but before that you also need to perform the first two steps above.

## Details
### Dataset Partition
code clone detection: Train-901028, Valid-415416, Test-415416

code smell detection: Train-1089, Valid-350, Test-350

technical debt detection: Train-24034, Valid-7674, Test-6652
### Finetune 

For code clone detection, use 10% Train and 10% Valid for fine-tuning, for code smell detection and technical debt detection, use full Train and Valid for fine-tuning.

### Hyperparameters
The training and testing commands and hyperparameters can be found in the setcode files in each folder. Training hyperparameters such as learning rate, batch size, and random seed are aligned with FMF and OSM in the same task and scenario to ensure fair and optimal comparisons.

### Non-CLS representation 
To confirm the impact of non-CLS representation, we conduct a preliminary experiment through adopting average-pooling representation method (since it takes too much time to run identifiers’ representations) that captures both identifiers and code tokens [https://doi.org/10.1016/j.eswa.2025.126762] as non-CLS representation. 

We find the identified potential best layer remained unchanged on the TDD task of CodeBERT, as shown below, the potentially best performing layer is still layer 7, followed by layer 6, indicating that specific feature extraction methods would not impact Coding-Fuse on locating the best layer. 

And we also find the performance of TDD has been **improved** by Coding-Fuse with average-pooling representation on CodeBERT-6+BERT-5 (i.e., F1: 0.786 -> 0.814, AUC-PR: 0.815 ->0.825).

| Layer | Average-pooling |       CLS       |
|:-----:|:---------------:|:---------------:|
|   1   |    0.17480215   |   0.225054745   |
|   2   |   0.194034255   |    0.27882957   |
|   3   |   0.224214224   |   0.294902097   |
|   4   |   0.252783222   |   0.326180597   |
|   5   |   0.287419558   |   0.350668591   |
| **6** | **0.289376665** |  **0.36186484** |
| **7** | **0.293355693** | **0.374065596** |
|   8   |   0.276642499   |   0.352729261   |
|   9   |   0.266328177   |   0.323606751   |
|   10  |   0.255506349   |   0.295021258   |
|   11  |   0.251451717   |   0.290002731   |
|   12  |   0.245410281   |   0.332676078   |

### Greenness Metrics
To comprehensively evaluate the greenness, we adopt four widely used metrics: training throughput, GPU memory usage during training, inference latency, and final model storage size. These metrics reflect both the efficiency of the training process and the environmental cost of real-world deployment. These four are also commonly used metrics in related fields, which can achieve exhaustive evaluations of greenness in actual deployment.

Training **throughput** measures how quickly data can be processed during training. Higher throughput reduces training time, leading to lower energy consumption and carbon emissions. The unit is example/second in this paper.

**GPU** memory usage indicates the hardware footprint during training. Models requiring less memory can be trained on smaller, more energy-efficient GPUs, thereby reducing both electricity and cooling costs. The unit is MB in this paper.

Test **latency** is critical in deployment, especially on edge devices or in real-time systems. Lower latency implies faster response and reduced power usage. The unit is second in this paper.

Model **storage** size determines the resources needed for storing and transferring the model. Smaller models are easier to deploy across platforms and consume less disk space and bandwidth. The unit is MB in this paper.




If you have any questions, you can leave a message.

