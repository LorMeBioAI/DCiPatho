# DCiPatho
- [DCiPatho](#dcipatho)
  - [Introduction](#introduction)
  - [Features](#features)
  - [DCiPatho Configuration](#dcipatho-configuration)
    - [Train and Evaluation Settings](#train-and-evaluation-settings)
    - [Prediction Settings](#prediction-settings)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Dataset Download](#dataset-download)
  - [Mini-BacRefSeq Dataset](#mini-bacrefseq-dataset)
  - [Full BacRefSeq Dataset](#full-bacrefseq-dataset)
  - [Pathogenic Potential Prediction - Basic Demo](#pathogenic-potential-prediction---basic-demo)
    - [Prerequisites](#prerequisites)
    - [Instructions](#instructions)
  - [Detail of DCiPatho\_predict.py](#detail-of-dcipatho_predictpy)
    - [Evaluation (Optional)](#evaluation-optional)
  - [Training Demo on Mini-BacRefSeq Dataset](#training-demo-on-mini-bacrefseq-dataset)
    - [Prerequisites](#prerequisites-1)
  - [License](#license)

DCiPatho is a deep learning approach for predicting the pathogenic potentials of genomic and metagenomic sequences. It utilizes a combination of deep cross neural networks, residual neural networks, and deep neural networks with integrated features of 3-to-7 k-mer. This approach outperforms traditional machine learning approaches that rely solely on single k-mer features and databases of known organisms, making it particularly effective for identifying unknown, unrecognized, and unmapped pathogen sequences.

## Introduction

Pathogen identification is crucial from a One Health perspective. Traditional methods of pathogen detection rely on databases of known organisms, which limit their performance when faced with novel or uncharacterized sequences. DCiPatho addresses this limitation by employing machine learning techniques to infer pathogenic phenotypes from single Next-Generation Sequencing (NGS) reads, even in the absence of biological context.
See more detail in https://academic.oup.com/bib/advance-article/doi/10.1093/bib/bbad194/7186278.

## Features

- Deep cross neural networks: DCiPatho utilizes deep cross neural networks to capture complex interactions and relationships between features, enhancing the predictive power of the model.
- Residual neural networks: The inclusion of residual neural networks helps alleviate the vanishing gradient problem and allows for the effective training of deep architectures.
- Integrated k-mer features: DCiPatho leverages integrated features derived from 3-to-7 k-mer representations, enabling a more comprehensive analysis of genomic and metagenomic sequences.
- Improved classification performance: The combination of deep cross, residual, and deep neural networks, along with integrated k-mer features, results in superior performance for the classification of pathogenic bacteria.



## DCiPatho Configuration

The `DCiPatho_config.py` provides configuration settings for training, evaluation, and prediction. Adjust the settings in the `Config` class to match your specific dataset and requirements.

The `Config` class provides the configuration settings for training, evaluation, and prediction in the DCiPatho deep learning approach. Here are the details of the available settings:

### Train and Evaluation Settings

- `patho_path`: The path to the npy or csv data file containing pathogenic sequences.
- `nonpatho_path`: The path to the npy or csv data file containing non-pathogenic sequences.
- `hidden_layers`: A list specifying the number of hidden layers in the ResNet module.
- `deep_layers`: A list specifying the number of hidden layers in the DeepNet module.
- `num_cross_layers`: The number of CrossNet layers.
- `end_dims`: A list specifying the dimensions of the CrossNet output layers.
- `out_layer_dims`: The dimension of the final output layer.
- `val_size`: The proportion of data to be used for validation during training (e.g., 0.2 represents 20%).
- `fold`: The number of folds for cross-validation.
- `test_size`: The proportion of validation data to be used for testing (e.g., 0.5 represents 50%).
- `random_state`: The random seed for reproducibility.
- `num_epoch`: The number of training epochs.
- `patience`: The number of epochs to wait for early stopping if the validation loss does not improve.
- `batch_size`: The batch size for training.
- `Dropout`: The dropout rate for regularization.
- `lr`: The learning rate for optimization.
- `l2_regularization`: The L2 regularization parameter.
- `device_id`: The ID of the GPU device to be used (if available).
- `use_cuda`: A flag indicating whether to use GPU acceleration (True) or CPU (False).
- `save_model`: A flag indicating whether to save the trained model.
- `output_base_path`: The base path for saving the model.
- `best_model_name`: The file path for saving the best model during training.

### Prediction Settings

- `raw_fasta_path`: The path to the folder containing the raw fasta files to be predicted.
- `combined_fasta_path`: The path to the combined fasta file (optional).
- `ks`: A list specifying the k-mer lengths to be used for frequency calculation.
- `num_procs`: The number of processes to be used for parallel computation.
- `freqs_file`: The file path for saving the calculated k-mer frequencies.
- `save_res_path`: The file path for saving the prediction results.

Please adjust these settings accordingly before running the DCiPatho code.


## Requirements

DciPatho is developed in Python 3 with modules and external tools.

```
numpy~=1.19.5
torch~=1.10.0+cu102
pandas~=1.1.5
scikit-learn~=0.24.2
```
## Installation

Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Dataset Download

To access the dataset files, please follow the instructions below:

1. Visit the following link: [Dataset Download Link](https://zenodo.org/record/7571307#.Y9H0e3ZBxPa)

2. Download the file named `toy_dataset_for_DCiPatho.zip` from the provided link. This file contains the frequency features extracted on the Mini-BacRefSeq dataset.

3. Download the file named `NCBI_22June_RefSeq_32927_Complete_1NP_2P_taxnonmy.csv`.

If you wish to download the FASTA files from the BacRefSeq dataset, you can run the following command:

```python
python NCBI_download.py
```

## Mini-BacRefSeq Dataset

The Mini-BacRefSeq dataset consists of 1,506 complete genomes, including 707 pathogenic bacterial strains from 540 species (including animal, human, and plant pathogens) and 799 nonpathogenic bacterial strains from 687 species.

## Full BacRefSeq Dataset

The BacRefSeq dataset contains complete genomes of 32,927 bacteria, labeled as either pathogenic or nonpathogenic bacterial strains. The labeling was done based on the genus level, with 22,046 genomes labeled as pathogenic bacterial strains from 1,269 genera, and 10,882 genomes labeled as nonpathogenic bacterial strains from 6,568 genera. The dataset includes multiple sequences of chromosomes and plasmids for the complete genome sequences.


## Pathogenic Potential Prediction - Basic Demo

This repository provides a basic demo for predicting pathogenic potentials using the DCiPatho model. Follow the steps below to quickly perform the prediction:

### Prerequisites
Before proceeding with the prediction, ensure that you have downloaded the `DCiPatho_best_k3-7_model.pt` file. If you don't have the file, you can download it from https://zenodo.org/record/7571307#.Y9H0e3ZBxPa.

### Instructions
1. Open the `DCiPatho_config.py` file and make the following modifications:
   - Set `self.best_model_name` to the file path of your `DCiPatho_best_k3-7_model.pt` file.
   - Set `self.raw_fasta_path` to the path of your input FASTA file.
2. Save the `DCiPatho_config.py` file after making the modifications.
3. Run the `DCiPatho_predict.py`.

This code will initiate the prediction process using the provided model and input data. The prediction results will be saved in the specified `save_res_path` directory.


Note: If you don't have the `DCiPatho_best_k3-7_model.pt` file, you will need to download it before proceeding with the prediction.


## Detail of DCiPatho_predict.py

The provided code performs prediction using a trained DCiPatho model. Here is an overview of the code:

1. The `load_model` function loads the trained DCiPatho model from the specified path, considering whether to use GPU acceleration or CPU.
2. The `predict` function performs the prediction using the loaded model.
3. It combines the raw fasta files into a single combined fasta file and calculates the k-mer frequencies for the combined file using the `cal.cal_main` function.
4. The data is preprocessed for prediction using the `data_preprocess_for_predict` function.
5. The preprocessed data is fed into the model to obtain predicted probabilities (`y_pred_probs`) for each input sequence.
6. The predicted probabilities are converted to binary predictions (`y_pred`) using a threshold of 0.5.
7. The predicted results are stored in a list and printed.
8. If specified, the predicted results are saved to a CSV file (`config.save_res_path`).
9. If ground truth labels (`y_test`) are provided, evaluation metrics such as accuracy, F1 score, ROC AUC score, and Matthews correlation coefficient (MCC) are computed and printed.

To run the prediction, ensure that you have set the appropriate configuration settings in the `Config` class and provide the necessary paths for raw fasta files and the trained model.
### Evaluation (Optional)

If you have ground truth labels available, you can evaluate the model's performance using evaluation metrics such as accuracy, F1 score, ROC AUC score, and Matthews correlation coefficient (MCC). Set the `y_test` parameter in the `predict()` function to the ground truth labels and uncomment the evaluation code.


## Training Demo on Mini-BacRefSeq Dataset
### Prerequisites
Before proceeding with the prediction, ensure that you have downloaded the `toy_dataset_for_DCiPatho.zip` file and unzip them. If you don't have the file, you can download it from https://zenodo.org/record/7571307#.Y9H0e3ZBxPa.
To train the DCiPatho model on the Mini-BacRefSeq dataset, follow these steps:

1. Open the `DCiPatho_config.py` file and set the following parameters:
   - Set `config.patho_path` to the file path of the `toy_patho_freq.csv` file.
   - Set `config.nonpatho_path` to the file path of the `toy_nonpatho_freq.csv` file.
   - Adjust other parameters in the configuration file as needed for your training setup.
   
2. Save the `DCiPatho_config.py` file after making the modifications.

3. Run the `DCiPatho_main.py` script using the following command:

```python
python DCiPatho_main.py
```

This command will start the training process using the specified configuration and the Mini-BacRefSeq dataset. The model will be trained and evaluated on the dataset.

Note: Make sure you have the necessary dependencies installed and that the dataset files (`toy_patho_freq.csv` and `toy_nonpatho_freq.csv`) are correctly specified in the configuration file.



## License

This project is licensed under the [MIT License](LICENSE).

Feel free to modify and use the code as per your requirements.

