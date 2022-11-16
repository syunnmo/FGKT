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
Here are some examples for using FGKT model (on ASSISTments Competition):  
```
  python main.py --dataset assist2017  
```
If you do not want to use the default parameters for your experiments, you can change the model parameters in the following way:  
```
  python main.py --dataset assist2017 --gpu 0 --patience 5 --lr 0.001 --num_heads 1  --mode 3 --exercise_embed_dim 128 --batch_size 32
```
Explanation of parameters:  
* gpu: Specify the GPU to be used, e.g '0,1,2,3'. If CPU is used then fill in -1.
* patience: Maximum number of times if validation loss does not improve.
* lr: The learning rate
* num_heads: Number of head attentions.
* mode: The selection of integration function.
* exercise_embed_dim: The number of question embedding dimensions.
* batch_size: The number of batch size.

