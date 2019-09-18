# Deep Learning for Natural Language Processing

### About
This repository contains code for mini-project of MSc. course Deep Learning for Natural Language Processing.

### Description
- The dataset for learning question classification can be found at
https://cogcomp.seas.upenn.edu/Data/QA/QC/


- The project requirement can be found at
https://canvas.uva.nl/courses/10769/pages/project-requirements?module_item_id=359508

### Configuration
Install the conda environment by running `conda env create -f environment.yml`. 

Then activate it by `conda activate dlnlp`.

Download the pretrained [Word2Vec word embeddings](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit) into folder **preprocessing**.

### Run the Code
Preprocessing the dataset and generate wordembeddings by running
`python -m dataset`.

Test FastText model by running
`python -m FastText`

Test LSTM model by running
`python -m LSTM`


### Division of labor
- Data preprocessing: Vincent
- LSTM model: Robbie
- FastText model: Kai
- TextCNN model: Philip & Others