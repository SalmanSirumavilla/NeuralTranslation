import tensorflow_text as tf_text
import tensorflow as tf
import random

# define a module level autotune
_AUTO = tf.data.AUTOTUNE

# loading data and doing basic processing
def load_data(fname):
        with open(r"D:\VIT\Papers\Implementation\NMT\bahdanau\fra.txt", "r", encoding="utf-8") as textFile:
            lines = textFile.readlines()
            pairs = [line.split("\t")[:-1] for line in lines]
            random.shuffle(pairs)
            source = [src for src, _ in pairs] 
            target = [trgt for _, trgt in pairs]
        return (source, target)

# splitting data into train, test and validation sets. Following 8:1:1 ratio    
def splitting_dataset(source, target):
        trainSize = int(len(source) * 0.8)
        valSize = int(len(source) * 0.1)
        (trainSource, trainTarget) = (source[: trainSize], target[: trainSize])
        (valSource, valTarget) = (source[trainSize : trainSize + valSize], target[trainSize : trainSize + valSize])
        (testSource, testTarget) = (source[trainSize + valSize :], target[trainSize + valSize :])
        return ((trainSource, trainTarget), (valSource, valTarget), (testSource, testTarget),)
        
# creating tensorflow dataset from input and output datasets
def make_dataset(splits, batchSize=32, train=False):
        (source, target) = splits 
        dataset = tf.data.Dataset.from_tensor_slices((source, target))
        if train:
            dataset = (dataset.shuffle(dataset.cardinality().numpy()).batch(batchSize).prefetch(_AUTO))
        else:
            dataset = (dataset.batch(batchSize).prefetch(_AUTO))
        return dataset
    
# lowering the input sentences and handling punctuations
def tf_lower_and_split_punct(text):
        # split accented characters
        text = tf_text.normalize_utf8(text, "NFKD")
        text = tf.strings.lower(text)
        text = tf.strings.regex_replace(text, "[^ a-z.?!,]", "")
        # add spaces around punctuation
        text = tf.strings.regex_replace(text, "[.?!,]", r" \0 ")
        text = tf.strings.strip(text)
        text = tf.strings.join(["[START]", text, "[END]"], separator=" ")
        return text

