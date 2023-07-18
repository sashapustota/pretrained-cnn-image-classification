<!-- ABOUT THE PROJECT -->
## About the project
This repository contains a scripts that trains an image classifier on the dataset of indian fashion garments using a pretrained CNN ```InceptionV3``` from [Keras](https://keras.io/api/applications/inceptionv3/) built with ```tensorflow```.

<!-- Data -->
## Data
The [dataset](https://www.kaggle.com/datasets/validmodel/indo-fashion-dataset) consists of approximately 160K images of 15 different indian fashion garments. Please make sure to download and unzip the data in the ```data``` folder.

<!-- USAGE -->
## Usage
To use the code you need to adopt the following steps.

**NOTE:** There may be slight variations depending on the terminal and operating system you use. The following example is designed to work using the Visual Studio Code version 1.76.0 (Universal). The terminal code should therefore work using a unix-based bash. The avoid potential package conflicts, the ```run.sh``` bash files contains the steps necesarry to create a virtual environment for the project. The code has been thoroughly tested and verified to work on a Mac machine running macOS Ventura 13.1. However, it should also be compatible with other Unix-based systems such as Linux. If you encounter any issues or have questions regarding compatibility on other platforms, please let me know.

1. Clone repository
2. Run ``run.sh``

### Clone repository

Clone repository using the following lines in the unix-based bash:

```bash
git clone https://github.com/sashapustota/pretrained-cnn-image-classification
cd pretrained-cnn-image-classification
```

### Run ```run.sh```

The ``run.sh`` script is used to automate the installation of project dependencies and configuration of the environment. By running this script, you ensure consistent setup across different environments and simplify the process of getting the project up and running.

```bash
bash run.sh
```

The script performs the following steps:

1. Creates a virtual environment for the project
2. Activates the virtual environment
3. Installs the required packages
4. Runs the main.py script
5. Deactivates the virtual environment

The ```main.py``` script perform the following steps:

1. Load the data
2. Preprocess the data
3. Load the model
4. Fit the models
5. Save a classification report, as well as plots of **loss** and **accuracy** metrics during the training of the model.



## Results

A classification report and the plots are saved to the master folder.

## Customizing model parameters

The ```main.py``` script is designed to run the models with the default parameters. However, it is possible to customize the parameters by changing the values in the scripts or by passing the parameters as arguments in the terminal.

The following parameters can be customized:

* ```--batch_size -b``` - The number of samples per gradient update. If unspecified, ```batch_size``` will default to 32.
* ```--train_samples -trs``` - The total number of samples in the training dataset. If unspecified, ```train_samples``` will default to 16000.
* ```--val_samples -vs``` - The total number of samples in the validation dataset. If unspecified, ```validation_samples``` will default to 4000.
* ```--test_samples -tes``` - The total number of samples in the test dataset. If unspecified, ```test_samples``` will default to 1000.
* ```--epochs -e``` - The number of epochs to train the model. If unspecified, ```epochs``` will default to 15.


To pass the parameters as arguments in the terminal, simply run the following lines in your terminal:

```bash
source ./inceptionv3_classifier_venv/bin/activate
python3 src/main.py -b <your value> -trs <your value> -vs <your value> -tes <your value> -e <your value>
deactivate
```

<!-- REPOSITORY STRUCTURE -->
## Repository structure
This repository has the following structure:
```
│   README.md
│   requirements.txt
│   run.sh
│
├──data
│
└──src
      main.py

```
<!-- REPOSITORY STRUCTURE -->
## Findings

When running the models with the parameters specified in the scripts, the following results were obtained:

```
lr-classifier
|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| blouse       | 0.863     | 0.955  | 0.906    | 66.0    |
| dhoti_pants  | 0.830     | 0.494  | 0.619    | 79.0    |
| dupattas     | 0.816     | 0.588  | 0.684    | 68.0    |
| gowns        | 0.591     | 0.394  | 0.473    | 66.0    |
| kurta_men    | 0.683     | 0.889  | 0.772    | 63.0    |
| leggings     | 0.642     | 0.743  | 0.689    | 70.0    |
| lehenga      | 0.904     | 0.835  | 0.868    | 79.0    |
| mojaris_men  | 0.932     | 0.809  | 0.866    | 68.0    |
| mojaris_women| 0.803     | 0.930  | 0.862    | 57.0    |
| nehru_jackets| 0.734     | 0.935  | 0.823    | 62.0    |
| palazzos     | 0.867     | 0.743  | 0.800    | 70.0    |
| petticoats   | 0.792     | 0.838  | 0.814    | 68.0    |
| saree        | 0.753     | 0.887  | 0.815    | 62.0    |
| sherwanis    | 0.914     | 0.525  | 0.667    | 61.0    |
| women_kurta  | 0.486     | 0.852  | 0.619    | 61.0    |
| accuracy     | 0.756     | 0.756  | 0.756    | 0.756   |
| macro avg    | 0.774     | 0.761  | 0.752    | 1000.0  |
| weighted avg | 0.778     | 0.756  | 0.751    | 1000.0  |

```

![Accuracy Plot](https://github.com/sashapustota/pretrained-cnn-image-classification/blob/main/plots/accuracy.png)

![Loss Plot](https://github.com/sashapustota/pretrained-cnn-image-classification/blob/main/plots/loss.png)
