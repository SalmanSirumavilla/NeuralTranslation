# import the necessary packages
import tensorflow_text as tf_text
import tensorflow as tf
import argparse
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--sentence", required=True,
    help="input english sentence")
args = vars(ap.parse_args())
# convert the input english sentence to a constant tensor
sourceText = tf.constant([args["sentence"]])
# load the translator model from disk
print("[INFO] loading the translator model from disk...")
translator = tf.saved_model.load("translator")
# perform inference and display the result
print("[INFO] translating english sentence to french...")
result = translator.translate(sourceText)
translatedText = result["text"][0].numpy().decode()
print("[INFO] english sentence: {}".format(args["sentence"]))
print("[INFO] french translation: {}".format(translatedText))