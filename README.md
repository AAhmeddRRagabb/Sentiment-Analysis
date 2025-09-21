# Sentiment Analysis Project
Alhamdulillah, this project focuses on practicing the development of sequence models for text data. It implements a TensorFlow model that classifies user reviews as positive or negative. Throughout the project, I applied a variety of data science and machine learning concepts, including data exploration, cleaning, text preprocessing, and model building. The model demonstrated strong performance across all datasets (training, validation, and test), with only minor overfitting observed on the training set. The project leverages the pre-trained GloVe embeddings from `glove.6B.50d.txt`.


## Project Workflow
The project was structured into four main stages:
- Data Understanding: Leveraged Pandas to load, explore, and navigate the dataset, examining its dimensions and overall structure.
- Data Cleaning: Utilized NumPy and Pandas to clean the text by standardizing formatting, parsing HTML tags, and removing extraneous characters.
- Data Preprocessing: Tokenized the text, built a vocabulary, and padded sequences to prepare the data for modeling.
- Model Development: Employed NumPy and TensorFlow to load pre-trained embeddings, convert text sequences into indices, build an LSTM (Long Short-Term Memory) network, train the model, and evaluate its performance


## Results
- The initial model achieved 99% accuracy on training data and 93% on validation data, showing slight overfitting. To address this, a regularized version of the model using L2 regularization was developed, improving generalization marginally.
- On the test set:
  * Model v1: 87% accuracy
  * Model v2: 88% accuracy


## Project Structure
├── 📓 sentiment_analysis.ipynb: Main notebook: cleaning, processing & modeling  
├── 🛠️ utils/: Helper functions for data & modeling  
├── 🤖 models: # Trained LSTM models  
├── 📊 data/: Dataset + GloVe vectors + word_to_idx map  
├── 🌐 app.py: Flask web app to serve predictions  
└── 🎨 templates/: Frontend (index.html) for UI


## Notes
- The dataset was obtained from the Kaggle [IMDB Movie Reviews](https://www.kaggle.com/datasets/vishakhdapat/imdb-movie-reviews) dataset.
- The GloVe file can not be uploaded here. It can be accessed through [GloVe Embeddings](https://nlp.stanford.edu/projects/glove/)
---
***Alhamdulilllah***






