#!/usr/bin/env python
"""
one_to_many_textgen.py: one to many RNN config for generating text with TF 2.0
"""
import sys
import os
import re
import numpy as np
import shutil
import pathlib
import tensorflow as tf
import kr_helper_funcs as kru

SEED = 42
kru.seed_all(SEED)

print(f"Using Tensorflow {tf.__version__}")

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
assert os.path.exists(DATA_DIR)
CHECKPOINT_DIR = DATA_DIR / "checkpoints"


def get_data(urls):
    texts = []

    for i, url in enumerate(urls):
        p = tf.keras.utils.get_file("ex1-{:d}.txt".format(i), url, cache_dir=".")
        text = open(p, "r").read()
        # remove byte order & new lines
        text = text.replace("\ufeff", "")
        text = text.replace("\n", " ")
        # replace multiple spaces with single space
        text = re.sub(f"\s+", " ", text)
        texts.extend(text)
    return texts


texts = get_data(
    [
        "http://www.gutenberg.org/cache/epub/28885/pg28885.txt",
        "https://www.gutenberg.org/files/6/6.txt",
        "https://www.gutenberg.org/files/7/7.txt",
        "https://www.gutenberg.org/files/8/8.txt",
        "https://www.gutenberg.org/files/9/9.txt",
        "https://www.gutenberg.org/files/10/10-0.txt",
        "https://www.gutenberg.org/files/11/11-0.txt",
        "https://www.gutenberg.org/files/12/12-0.txt",
    ]
)
# print(texts)

# create vocab
vocab = sorted(set(texts))
print("vocab size: {:d}".format(len(vocab)))

# create mapping from vocab chars to ints
char2idx = {c: i for i, c in enumerate(vocab)}
idx2char = {i: c for c, i in char2idx.items()}

# numericize the texts
texts_as_ints = np.array([char2idx[c] for c in texts])
data = tf.data.Dataset.from_tensor_slices(texts_as_ints)
# number of characters to show before asking for prediction
# sequences: [None, 100]
seq_length = 100
sequences = data.batch(seq_length + 1, drop_remainder=True)


def split_train_labels(sequence):
    input_seq = sequence[0:-1]
    output_seq = sequence[1:]
    return input_seq, output_seq


sequences = sequences.map(split_train_labels)
# set up for training batches: [None, 64, 100]
batch_size = 64
steps_per_epoch = len(texts) // seq_length // batch_size
dataset = sequences.shuffle(10000).batch(batch_size, drop_remainder=True)


class CharGenModel(tf.keras.Model):
    def __init__(self, vocab_size, num_timesteps, embedding_dim, **kwargs):
        super(CharGenModel, self).__init__(**kwargs)
        self.embedding_layer = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.rnn_layer = tf.keras.layers.GRU(
            num_timesteps,
            recurrent_initializer="glorot_uniform",
            recurrent_activation="sigmoid",
            stateful=True,
            return_sequences=True,
        )
        self.dense_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, x):
        x = self.embedding_layer(x)
        x = self.rnn_layer(x)
        x = self.dense_layer(x)
        return x


def loss(labels, predictions):
    return tf.losses.sparse_categorical_crossentropy(
        labels,
        predictions,
        from_logits=True,
    )


vocab_size = len(vocab)
embedding_dim = 256
model = CharGenModel(vocab_size, seq_length, embedding_dim)
model.build(input_shape=(batch_size, seq_length))
model.compile(optimizer=tf.optimizers.Adam(), loss=loss)


def generate_text(
    model,
    prefix_string,
    char2idx,
    idx2char,
    num_chars_to_generate=1000,
    temperature=1.0,
):
    inp = [char2idx[s] for s in prefix_string]
    inp = tf.expand_dims(inp, 0)
    text_generated = []
    model.reset_states()
    for i in range(num_chars_to_generate):
        preds = model(inp)
        preds = tf.squeeze(preds, 0) / temperature
        # predict char returned by model
        pred_id = tf.random.categorical(preds, num_samples=1)[-1, 0].numpy()
        text_generated.append(idx2char[pred_id])
        # pass the prediction as the next inp to the model
        inp = tf.expand_dims([pred_id], 0)
    return prefix_string + "".join(text_generated)


num_epochs = 50
for i in range(num_epochs // 10):
    model.fit(
        dataset.repeat(),
        epochs=10,
        steps_per_epoch=steps_per_epoch,
        # callbacks=[checkpoint_callback, tensorboard_callback]
    )
    checkpoint_file = os.path.join(CHECKPOINT_DIR, "model_epoch_{:d}".format(i + 1))
    model.save_weights(checkpoint_file)
    # create generative model using the trained model so far
    gen_model = CharGenModel(vocab_size, seq_length, embedding_dim)
    gen_model.load_weights(checkpoint_file)
    gen_model.build(input_shape=(1, seq_length))
    print(f"After epoch: {(i+1):d}\n{'-' * 50}")
    print(generate_text(gen_model, "Alice ", char2idx, idx2char))
    print("*" * 50)
