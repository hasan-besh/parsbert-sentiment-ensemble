
# Persian Sentiment Analysis using ParsBERT & Deep Ensemble Learning

## Overview
This project performs **sentiment analysis on Persian text** using a deep learning ensemble built on **ParsBERT embeddings**.  
It first uses **ParsBERT** to extract semantic representations of Persian sentences, then trains multiple neural architectures (LSTM, GRU, CNN, and Dense) on these embeddings.  
Finally, an **ensemble model** combines their predictions to improve accuracy and generalization across different sentiment categories.

## Key Features
- **ParsBERT embeddings:** Leverages transformer-based embeddings for rich Persian text representation.  
- **Multiple deep learning models:** Implements and trains LSTM, GRU, CNN, and Dense networks.  
- **Ensemble learning:** Combines all models’ outputs to enhance robustness and accuracy.  
- **Comprehensive preprocessing:** Uses `Hazm` and `PersianStemmer` for Persian text normalization, stemming, and cleaning.  
- **Performance evaluation:** Measures accuracy, precision, recall, F1-score, and visualizes results with PR, ROC, and confusion matrix plots.

## Model Pipeline
1. **Data Preparation** – Load and merge Persian text datasets.  
2. **Text Preprocessing** – Normalize, clean, and tokenize sentences.  
3. **Embedding Extraction** – Generate dense vectors using `ParsBERT`.  
4. **Model Training:**  
   - LSTM Network  
   - GRU Network  
   - CNN Model  
   - Fully Connected (Dense) Model  
5. **Ensemble Aggregation** – Combine model predictions via voting or averaging.  
6. **Evaluation & Visualization** – Compute metrics and visualize PR/F1/Confusion curves.

## Project Structure
pars_bert_embedding.ipynb # Notebook for training and ensemble evaluation

perl
Copy code

## 🛠 Requirements
Install dependencies:
```bash
transformers
tensorflow
torch
scikit-learn
hazm
PersianStemmer
clean-text[gpl]
nltk
pandas
numpy
matplotlib
tqdm
 How to Run
Clone the repository:

bash
Copy code
git clone https://github.com/<hasan-besh>/parsbert-sentiment-ensemble.git
cd parsbert-sentiment-ensemble
Open and run the notebook:

bash
Copy code
jupyter notebook pars_bert_embedding.ipynb
The notebook will:

Preprocess Persian text data

Generate ParsBERT embeddings

Train LSTM, GRU, CNN, and Dense models

Build an ensemble and evaluate performance

 Evaluation Metrics
Accuracy

Precision / Recall / F1-score

Confusion Matrix

Precision–Recall and ROC Curves

 Applications
Sentiment analysis of Persian reviews, comments, or tweets

Emotion classification in Persian NLP systems

Building smart Persian chatbots or feedback analyzers

License
This project is released under the MIT License.
