# Twitter 'Halal' Sentiment Analysis
### Sentiment Polarity Classification on Twitter Data for "Halal" Keyword

This repository holds the work for sentiment polarity classification of Twitter data gathered by University of Malaya Halal research group.  
This project implements various deep learning architectures using 2 features sets:  
 * Word2Seq = A simple sequence-based features matrix  (research article in progress)
 * Word2Vec = Based on the pre-trained Google's Word2Vec-300 dimensions with 3 billions running words

The list of models available are:
-   Word2Seq Convolutional Neural Network
-   Word2Seq Long Short Term Memory
-   Word2Seq Convolutional Neural Network + Long Short Term Memory
-   Word2Seq Convolutional Neural Network + Bi-directional Recurrent Neural Network + Bi-directional Long Short Term Memory
-   Word2Vec Convolutional Neural Network
-   Word2Vec Long Short Term Memory
-   Word2Vec Convolutional Neural Network + Long Short Term Memory
-   Word2Vec Convolutional Neural Network + Bi-directional Recurrent Neural Network + Bi-directional Long Short Term Memory

The data are private datasets and can be made available upon further requests.  


### Libraries and Engines
This project uses:
 * **Ubuntu 18.04** as the main OS because deep learning frameworks and architectures works better with Linux rather than Windows (performance wise).
 * **NVIDIA RTX 2080 (Gigabyte)** as the main compute engine for the neural network.
 * **NVIDIA GPU Cloud (NGC) Container** for easy deployment of highly optimized Docker images for deep learning projects.    
 * **Docker v18.09.0** for hosting the NGC images.
 * **NVIDIA Docker v2.0.3** for the customized docker manager for NGC images.  
 * **NVIDIA CUDA Toolkit v10.0** for the latest CUDA toolkit available to support next-gen Turing GPU.
 * **Tensorflow:18.10-py3 Docker Image** from NGC (nvcr.io) for the latest version of highly optimized Tensorflow image using GPU learning.  
 * **Tensorflow 1.12** as the main backend of the neural network framework.  
 * **Keras v2.2.4** as the main high-level neural network API.  

If it is still unclear, this project uses **Graphical Processing Unit (GPU)** based learning using Tensorflow as the backend.  

### Overview

This repository contains all the **Jupyter Notebook** scripts for:
 * Training all the models
 * Prediction of the sentiment polarity

This project aims to implement all **8** deep learning models to perform sentiment polarity classification of the Twitter data.  
The outcome of the predictions are **8** sets of :
 * Predicted sentiment
 * Predicted positive sentiment probability
 * Predicted negative sentiment probability
 
With all **8** sets of metrics are available, the weighted average of each metric is calculated:
 * Weighted average of positive sentiment probability  
 
<a href="https://www.codecogs.com/eqnedit.php?latex=posProb&space;_{\omega}&space;=&space;\frac{\sum&space;posProb}{n}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?posProb&space;_{\omega}&space;=&space;\frac{\sum&space;posProb}{n}" title="posProb _{\omega} = \frac{\sum posProb}{n}" /></a>
 * Weighted average of negative sentiment probability

<a href="https://www.codecogs.com/eqnedit.php?latex=negProb&space;_{\omega}&space;=&space;\frac{\sum&space;negProb}{n}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?negProb&space;_{\omega}&space;=&space;\frac{\sum&space;negProb}{n}" title="posProb _{\omega} = \frac{\sum negProb}{n}" /></a>  

Therefore, upon having the weighted sentiment probabilities, the sentiment polarity is identified using a set of rules:  

 * <a href="https://www.codecogs.com/eqnedit.php?latex=\boldsymbol{if}&space;\;&space;posProb&space;_{\omega}&space;\boldsymbol{>}&space;negProb&space;_{\omega}&space;\;\;&space;\boldsymbol{then}&space;\;\;&space;sentiment&space;_{\omega}&space;=&space;Positive" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\boldsymbol{if}&space;\;&space;posProb&space;_{\omega}&space;\boldsymbol{>}&space;negProb&space;_{\omega}&space;\;\;&space;\boldsymbol{then}&space;\;\;&space;sentiment&space;_{\omega}&space;=&space;Positive" title="\boldsymbol{if} \; posProb _{\omega} \boldsymbol{>} negProb _{\omega} \;\; \boldsymbol{then} \;\; sentiment _{\omega} = Positive" /></a>
 * <a href="https://www.codecogs.com/eqnedit.php?latex=\boldsymbol{else\;if}&space;\;&space;posProb&space;_{\omega}&space;\boldsymbol{==}&space;negProb&space;_{\omega}&space;\;\;&space;\boldsymbol{then}&space;\;\;&space;sentiment&space;_{\omega}&space;=&space;Neutral" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\boldsymbol{else\;if}&space;\;&space;posProb&space;_{\omega}&space;\boldsymbol{==}&space;negProb&space;_{\omega}&space;\;\;&space;\boldsymbol{then}&space;\;\;&space;sentiment&space;_{\omega}&space;=&space;Neutral" title="\boldsymbol{else\;if} \; posProb _{\omega} \boldsymbol{==} negProb _{\omega} \;\; \boldsymbol{then} \;\; sentiment _{\omega} = Neutral" /></a>
 * <a href="https://www.codecogs.com/eqnedit.php?latex=\boldsymbol{else}&space;\;\;&space;sentiment&space;_{\omega}&space;=&space;Negative" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\boldsymbol{else}&space;\;\;&space;sentiment&space;_{\omega}&space;=&space;Negative" title="\boldsymbol{else} \;\; sentiment _{\omega} = Negative" /></a>

This reduces the possibilities of having **ties** condition if only the **predicted sentiments** are taken into account. However, **ties** condition is still possible but the probability is very unlikely to occur.  
The **processes**, **training patterns**, **results** and **outputs** for each training and prediction session are available inside the notebook. Feel free to explore.  
Below are the weighted outcomes of the sentiment.

| Number of Weighted Positive Sentiment |Number of Weighted Negative Sentiment|
|--|--|
| 90910  |14632|

### Model Training Results

The graph below visualizes the testing set accuracies of both feature sets across all models.  
The highest accuracy achieved is the **Word2Vec CNN + LSTM** model while the lowest accuracy achieved is the **Word2Seq LSTM** model.  
However, all of the models are of equal important in this project as each of them have different implementation and personality.  

![](g1.png?raw=true "")

Thank you.  
