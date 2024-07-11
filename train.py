import argparse, os
import tensorflow as tf
import pickle
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.model import DeepBidirectionalLSTM
from src.visualize_results import plot_graphs

def train_model(url, headline_title, label_title, units, embedding_size, input_length, num_layers, train_size, batch_size, epochs, learning_rate):
    print("Step 1: Loading data")
    # Load and preprocess data
    dataset, label_dataset = load_data(url, headline_title, label_title)
    
    print("Step 2: Splitting data into training and testing sets")
    size = int(len(dataset) * train_size)
    
    train_sentence = dataset[:size]
    test_sentence = dataset[size:]
    
    train_label = label_dataset[:size]
    test_label = label_dataset[size:]
    
    print("Step 3: Preprocessing data")
    padded_train_sequences, padded_test_sequences, tokenizer = preprocess_data(train_sentence, test_sentence, max_length=input_length)

    vocab_size = len(tokenizer.word_index) + 1

    print("Step 4: Initializing the Deep Bidirectional LSTM model")
    # Initialize and compile model
    model = DeepBidirectionalLSTM(units, embedding_size, vocab_size, input_length, num_layers)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='binary_crossentropy', metrics=['acc'])

    print("Step 5: Training the model")
    # Train model
    history = model.fit(padded_train_sequences, train_label, validation_data=(padded_test_sequences, test_label), batch_size=batch_size, epochs=epochs)

    if not os.path.exists(args.model_save_dir):
      os.makedirs(args.model_save_dir)

    print("Step 6: Saving the model and tokenizer (./data/model)")
    model.save_weights(os.path.join(args.model_save_dir, 'model_weights.h5'))
    with open(os.path.join(args.model_save_dir, 'tokenizer.pickle'), 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Step 7: Visualizing training results (./data/plots)")
    plot_save_dir = './data/plots'
    plot_graphs(history, 'acc', save_dir=plot_save_dir)
    plot_graphs(history, 'loss', save_dir=plot_save_dir)

    print("Training complete and model saved.")

# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a Deep Bidirectional LSTM model.')
    parser.add_argument('--url', type=str, default='https://storage.googleapis.com/learning-datasets/sarcasm.json')
    parser.add_argument('--headline_title', type=str, default='headline')
    parser.add_argument('--label_title', type=str, default='is_sarcastic')
    parser.add_argument('--units', type=int, default=128)
    parser.add_argument('--embedding_size', type=int, default=100)
    parser.add_argument('--max_length', type=int, default=25)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--train_size', type=float, default=0.8)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--model_save_dir', type=str, default='./data/model')

    args = parser.parse_args()

    print("Starting the training process with the following parameters:")
    print(f"URL: {args.url}")
    print(f"Headline title: {args.headline_title}")
    print(f"Label title: {args.label_title}")
    print(f"Units: {args.units}")
    print(f"Embedding size: {args.embedding_size}")
    print(f"Max length: {args.max_length}")
    print(f"Number of layers: {args.num_layers}")
    print(f"Train size: {args.train_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Model saved: {args.model_save_dir}")

    train_model(
        url=args.url,
        headline_title=args.headline_title,
        label_title=args.label_title,
        units=args.units,
        embedding_size=args.embedding_size,
        input_length=args.max_length,
        num_layers=args.num_layers,
        train_size=args.train_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate
    )
