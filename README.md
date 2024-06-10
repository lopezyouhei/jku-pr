# Practical Work
This is a self-supervised learning (SSL) project from the Master's program in Artificial Intelligence at the Johannes Kepler University (JKU).

## About
Investigation on how the number of classes affects the performance of SSL pre-training. For this a NNCLR head was trained atop of ImageNet-1k features from a MAE L/16 model. This setting is the same as the second stage of MAE-CT ([link](https://arxiv.org/abs/2304.10520)), where only the 
NNCLR head is trained. The hyperparameters selected for training the NNCLR head were found through grid search based on hyperparameters from MAE-CT and NNCLR ([link](https://arxiv.org/abs/2104.14548)). Class reduction was performed in steps of 100. All models were trained with roughly 
~13M samples (10 epochs without class reduction). The models were then evaluated at 3 layers: 
- linear: activations after first nn.Linear layer of NNCLR head projector
- projector: projector output
- predictor: predictor output

k-means clustering (accuracy & silhouette score), k-NN (accuracy) and UMAP were used to evaluate the models. For UMAP, superclasses referred to as main_5, dog_15 and wen_10 were used rather than using all 1000 classes (synset classes can be found in PR/config/class_group.py).
All results can be found under the results folder. For a more thorough analysis of the results please refer to the report in the main directory.

## Creating the environment
After cloning the project, change to the project directory (jku-pr), then create a new environment as following:
```terminal
conda env create -f environment.yml
```
