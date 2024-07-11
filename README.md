# Deep BiLSTM Project

## Table of Contents
1. [Introduction](#introduction)
2. [Architecture Overview](#architecture-overview)
3. [Setup and Installation](#setup-and-installation)
4. [Usage](#usage)
5. [Project Structure](#project-structure)
6. [Acknowledgements](#acknowledgements)

## Introduction
The Deep BiLSTM Project is a text classification model that uses deep bidirectional LSTM layers to classify sarcasm in text data. The model is trained on a dataset of headlines and labels indicating whether the headline is sarcastic or not.

## Architecture Overview
The model architecture consists of the following components:
<p align="center">
Input Layer<br>
     |<br>
Embedding Layer<br>
     |<br>
Bidirectional LSTM Layer 1 (Forward + Backward)<br>
     |<br>
Bidirectional LSTM Layer 2 (Forward + Backward)<br>
     |<br>
... (Additional Bidirectional LSTM Layers) ...<br>
     |<br>
Bidirectional LSTM Layer N (Forward + Backward)<br>
     |<br>
Fully Connected (Dense) Layer 1<br>
     |<br>
Fully Connected (Dense) Layer 2<br>
     |<br>
Output Layer (Sigmoid for binary classification)
</p>

## Setup and Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/juliemai16/deep_bi_lstm_project.git
    cd deep_bi_lstm_project
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. **Train the Model**:
    ```bash
    python src/train.py --url <data_url> --headline_title <headline_key> --label_title <label_key> --units <units> --embedding_size <embedding_size> --max_length <max_length> --num_layers <num_layers> --train_size <train_size> --batch_size <batch_size> --epochs <epochs> --learning_rate <learning_rate>
    ```

2. **Classify Text**:
    ```bash
    python classify.py --model_path ./data/model/model_weights.h5 --tokenizer_path ./data/model/tokenizer.json --text "Your text to classify"
    ```

## Project Structure
```
deep_bi_lstm_project/
│
├── data/
│ ├── model/ # Directory to save model and tokenizer
│ ├── plots/ # Directory to save plots
│
├── src/
│ ├── data_loader.py # Contains the function to load data
│ ├── preprocessing.py # Contains the function to preprocess data
│ ├── custom_layers.py # Contains the custom LSTM layer
│ ├── model.py # Contains the model definition
│ ├── visualize_results.py # Script to visualize training results
│
├── train.py # Script to train the model
├── classify.py # Script to classify new text
├── requirements.txt # Required packages
├── README.md # Project documentation
│
```

## Acknowledgements
This project is inspired by various tutorials and examples from the ProtonX community.
