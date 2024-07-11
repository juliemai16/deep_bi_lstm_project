from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def preprocess_data(train_sentences, test_sentences, vocab_size=10000, max_length=25):
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_sentences)

    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    padded_train_sequences = pad_sequences(train_sequences, maxlen=max_length, truncating="post", padding="post")

    test_sequences = tokenizer.texts_to_sequences(test_sentences)
    padded_test_sequences = pad_sequences(test_sequences, maxlen=max_length, truncating="post", padding="post")

    return padded_train_sequences, padded_test_sequences, tokenizer
