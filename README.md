# FGKT
Fine-Grained Knowledge Tracing model (FGKT) 

# Dataset
In 'data' folder, we have provided the processed datasets. 
If you would like to access the raw datasets, the raw datasets are placed in the following links:
* Statics : [address](https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=507)
* Synthetic-5 : [address](https://github.com/chrispiech/DeepKnowledgeTracing/tree/master/data/synthetic)
* ASSISTments2009  : [address](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data)
* ASSISTments2015 : [address](https://sites.google.com/site/assistmentsdata/datasets/2015-assistments-skill-builder-data)
* ASSISTments Competition : [address](https://sites.google.com/view/assistmentsdatamining/dataset)

# Setups

__Service__: 
* Linux

__Environment__:

* python 3+
* sklearn  0.21.3
* tqdm 4.54.1
* torch 1.7.0
* numpy 1.19.2

# Running FGKT
Here are some examples for using FGKT model (on ASSISTments2009):
`python main.py --dataset assist2015 `
