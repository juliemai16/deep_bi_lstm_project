import argparse, os
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.preprocessing import preprocess_data
from src.model import DeepBidirectionalLSTM

def load_model(model_path, units, embedding_size, vocab_size, input_length, num_layers):
    model = DeepBidirectionalLSTM(units, embedding_size, vocab_size, input_length, num_layers)
    
    # Dummy call to initialize variables
    dummy_input = tf.zeros((1, input_length), dtype=tf.int32)
    model(dummy_input)
    
    model.load_weights(os.path.join(model_path, 'model_weights.h5'))
    return model

def classify_text(model, tokenizer, text, max_length):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, truncating='post', padding='post')
    prediction = model.predict(padded_sequence)
    return prediction[0][0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify text using a trained Deep Bidirectional LSTM model.')
    parser.add_argument('--model_path', type=str, default='./data/model', help='Path to the trained model')
    parser.add_argument('--text', type=str, required=True, help='Text to classify')
    parser.add_argument('--units', type=int, default=128)
    parser.add_argument('--embedding_size', type=int, default=100)
    parser.add_argument('--max_length', type=int, default=25)
    parser.add_argument('--num_layers', type=int, default=1)

    args = parser.parse_args()

    # Load tokenizer
    import pickle
    with open(os.path.join(args.model_path, 'tokenizer.pickle'), 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    vocab_size = len(tokenizer.word_index) + 1

    print(f"Classifying text: {args.text}")
    model = load_model(args.model_path, args.units, args.embedding_size, vocab_size, args.max_length, args.num_layers)
    prediction = classify_text(model, tokenizer, args.text, args.max_length)
    print(f"Prediction: {'Sarcastic' if prediction > 0.5 else 'Not Sarcastic'} (Probability: {prediction})")
