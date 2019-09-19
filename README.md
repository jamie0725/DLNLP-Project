# Deep Learning for Natural Language Processing

### About
This repository contains code for mini-project of MSc. course Deep Learning for Natural Language Processing.

### Description
We cover the classical NLP problem of question classification, which consists of two parts.

**Classification Task**

* We choose Facebook's FastText as our baseline, and further implement two neural models, namely LSTM and TextCNN.

* The three models are compared in terms of the overall classification accuracy, and the precision, recall and F1-score values for each category.

**Adding Unsupervised Component for Better Interpretability**

* A layer of binary latent variables is added to our neural models that select what parts of the input expose features for classification.


### Dataset
The dataset for we use can be found [here](https://cogcomp.seas.upenn.edu/Data/QA/QC/).


### Prerequisites
1. Install the conda environment by running `conda env create -f environment.yml`. 

2. Then activate it by `conda activate dlnlp`.

3. Download the pre-trained [Word2Vec word embeddings](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit) into folder _preprocessing_.

### Running Instructions
* Preprocess the dataset and extract word embeddings by running `python -m dataset`.

* Test the FastText model by running `python FastText/model.py --mode=eval`.

* Train the LSTM model by running `python -m LSTM`.

* Train the TextCNN model by running `python -m TextCNN`.