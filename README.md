# DCiPatho

DCiPatho is a tool for rapid identification of pathogens from sequencing data. This tool is based on the k-mer frequence data.

## Introduction

Pathogen identification is important for one health perspective. Traditional approaches to open-view pathogen detection depend on databases of known organisms, which limits their performance on unknown, unrecognized, and unmapped sequences. In contrast, machine learning methods can infer pathogenic phenotypes from single NGS reads, even though the biological context is unavailable. We present **DCiPatho**, a deep learning approach to predict the pathogenic potentials of genomic and metagenomic sequences. It is a kind of deep cross neural network. We show that combination of cross, residual and deep neural networks with integrated features of 3-to-7 k-mer outperform on single k-mer features with traditional machine learning approaches. We demonstrate **DCiPatho** model has superior performance in classifying pathogenic bacteria.

## Requirements

DciPatho is developed in Python 3 with modules and external tools.

```
numpy~=1.19.5
torch~=1.10.0+cu102
pandas~=1.1.5
scikit-learn~=0.24.2
```
## Download dataset 
First download all the files from this link:
https://zenodo.org/record/7571307#.Y9H0e3ZBxPa

### Toy Dataset: Mini-BacRefSeq
 The Mini-BacRefSeq is comprised of 1,506 complete genomes (Supplementary Table 2), including 707 pathogenic bacterial strains (540 species, including animal, human, and plant pathogens) and 799 nonpathogenic bacterial strains (687 species).
 The file name of the frequency features extracted on the Mini-BacRefSeq dataset are:
 ```
 toy_dataset_for_DCiPatho.zip
 ```


### Full Dataset: BacRefSeq 
The complete genomes of the 32,927 bacteria were labelled as pathogenic or nonpathogenic bacterial strains. Based on the genus level, 22,046 genomes were labelled as pathogenic bacterial strains (1269 genera), and 10882 genomes were as labelled nonpathogenic bacterial strains (6568 genera). Multiple sequences of chromosomes and plasmids were included for the complete genome sequences.
Please see Excel NCBI_22June_RefSeq_32927_Complete_1NP_2P_taxnonmy.csv for details of the dataset.
If you want to download these fasta files from the dataset, please run:
```
NCBI_download.py
```
### Our best DCiPatho model for prediction
The model is named for ```DCiPatho_best_k3-7_model.pt```

For more information on its use, please see: #Basic demo for prediction.






## Basic demo for prediction
If you have already downloaded our DCiPatho_best_k3-7_model.pt,then:
you can predict pathogenic potentials with the built-in models out of the box, first, change `pred_input_path` in `DCiPatho_config.py` to the directory containing the .fasta file you need to classify, and set `pred_output_path` to the directory where the prediction results need to be output, then:

```
# run predcit  

python DCiPatho.py
```



## Basic demo for training on Mini-BacRefSeq dataset

You can set the parameter of training and model in DCiPatho_config.py file. Then you can train the DCiPatho model and evaluate it on mini-RefSeq dataset by running:

``````python
python DCiPatho_main.py
``````

