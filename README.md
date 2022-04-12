# DCiPatho

DCiPatho is a tool based on the k-mer frequence data for fast classification of pathogens from sequencing data.

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

## Examples

Set parameters in DCiPatho_config.py file.

Use python DCiPatho.py to train and evaluate model.

**Input data**:


You can download the k-mer frequencies of MAGs dataset in the link below, then unzip them to example_data file.

Link：https://pan.baidu.com/s/1cVXoRre9eDuslTNVqD-EAQ 
code：dci1

**Output data**:

The probability of input sequences as pathogen.

