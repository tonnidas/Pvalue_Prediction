# P Value Prediction


# How to run:
* Generate embedding on the entire filtered dataset: run `embedding_generator.py` file with the choice of appropriate embedding.
* Split the dataset into two parts: one having the p-values for all nodes and another not having the p-values for all nodes, run `pval_split.py` file.
* Generate the p-values of nodes that do not have p-values, run `main.py` file.
* running commands: 

```
$ python embedding_generator.py --folder=Alzheimers_Disease_Graph --dataset=Alzheimer --embedding=GCN
$ python pval_split.py --dataset=Alzheimer --embedding=GCN 
$ python main.py --dataset=Alzheimer --emb=GCN --modelName=NeuralNet_hyper --case=2
```




