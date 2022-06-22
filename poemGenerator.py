import argparse
import pickle
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# load tokenizer
file = open("tokenizer.obj", 'rb')
tokenizer = pickle.load(file)

# load model
model = tf.keras.models.load_model('my_model')


def generatePoem(length, seed):
    tokens = tokenizer.texts_to_sequences([seed])[0]
    length - len(tokens)
    tokens = pad_sequences([tokens], maxlen=15, padding='post')
    newWord = ""
    for _ in range(length-len(tokens)):
        predicton = model.predict(tokens, verbose=0)
        predicton = np.argmax(predicton, axis=-1)
        for word, index in tokenizer.word_index.items():
            if index == predicton:
                newWord = word
                break
        seed += " " + newWord
        tokens = tokenizer.texts_to_sequences([seed])[0]
        tokens = pad_sequences([tokens], maxlen=15, padding='post')

    return seed


if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("length", help="Total length of output text", type=int)
    parser.add_argument("seed", help="Starting text for the poem", type=str)
    args = parser.parse_args()
    print(generatePoem(args.length, args.seed))
