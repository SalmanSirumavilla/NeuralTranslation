import tensorflow as tf
tf.random.set_seed(42)

# import the necessary packages
from bahdanau import config
from bahdanau.schedule import WarmUpCosine
from bahdanau.dataset import load_data
from bahdanau.dataset import splitting_dataset
from bahdanau.dataset import make_dataset
from bahdanau.dataset import tf_lower_and_split_punct
from bahdanau.models import Encoder
from bahdanau.models import Decoder
from bahdanau.translator import TrainTranslator
from bahdanau.translator import Translator
from bahdanau.loss import MaskedLoss
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow_text as tf_text
import matplotlib.pyplot as plt
import numpy as np
import os

# load data from disk
print(f"[INFO] loading data from {config.DATA_FILE_NAME}...")
(source, target) = load_data(fname=config.DATA_FILE_NAME)

# split the data into training, validation, and test set
(train, val, test) = splitting_dataset(source=source, target=target)

# build the TensorFlow data datasets of the respective data splits
print("[INFO] building TensorFlow Data input pipeline...")
trainDs = make_dataset(splits=train, batchSize=config.BATCH_SIZE,
    train=True)
valDs = make_dataset(splits=val, batchSize=config.BATCH_SIZE,
    train=False)
testDs = make_dataset(splits=test, batchSize=config.BATCH_SIZE,
    train=False)

print("[INFO] performing text vectorization...")
sourceTextProcessor = TextVectorization(
    standardize=tf_lower_and_split_punct,
    max_tokens=config.SOURCE_VOCAB_SIZE
)
sourceTextProcessor.adapt(train[0])
# create target text processing layer and adapt on the training
# target sentences
targetTextProcessor = TextVectorization(
    standardize=tf_lower_and_split_punct,
    max_tokens=config.TARGET_VOCAB_SIZE
)
targetTextProcessor.adapt(train[1])

# build the encoder and the decoder
print("[INFO] building the encoder and decoder models...")
encoder = Encoder(
    sourceVocabSize=config.SOURCE_VOCAB_SIZE,
    embeddingDim=config.ENCODER_EMBEDDING_DIM,
    encUnits=config.ENCODER_UNITS
)
decoder = Decoder(
    targetVocabSize=config.TARGET_VOCAB_SIZE,
    embeddingDim=config.DECODER_EMBEDDING_DIM,
    decUnits=config.DECODER_UNITS,
)
# build the trainer module
print("[INFO] build the translator trainer model...")
translatorTrainer = TrainTranslator(
    encoder=encoder,
    decoder=decoder,
    sourceTextProcessor=sourceTextProcessor,
    targetTextProcessor=targetTextProcessor,
)

# get the total number of steps for training.
totalSteps = int(trainDs.cardinality() * config.EPOCHS)
# calculate the number of steps for warmup.
warmupEpochPercentage = config.WARMUP_PERCENT
warmupSteps = int(totalSteps * warmupEpochPercentage)
# Initialize the warmupcosine schedule.
scheduledLrs = WarmUpCosine(
    lrStart=config.LR_START,
    lrMax=config.LR_MAX,
    warmupSteps=warmupSteps,
    totalSteps=totalSteps,
)
# configure the loss and optimizer
print("[INFO] compile the translator trainer model...")
translatorTrainer.compile(
    optimizer=Adam(learning_rate=scheduledLrs),
    loss=MaskedLoss(),
)
# build the early stopping callback
earlyStoppingCallback = EarlyStopping(
    monitor="val_batch_loss",
    patience=config.PATIENCE,
    restore_best_weights=True,
)

# train the model
print("[INFO] training the translator model...")
history = translatorTrainer.fit(
    trainDs,
    validation_data=valDs,
    epochs=config.EPOCHS,
    callbacks=[earlyStoppingCallback],
)
# save the loss plot
if not os.path.exists(config.OUTPUT_PATH):
    os.makedirs(config.OUTPUT_PATH)
plt.plot(history.history["batch_loss"], label="batch_loss")
plt.plot(history.history["val_batch_loss"], label="val_batch_loss")
plt.xlabel("EPOCHS")
plt.ylabel("LOSS")
plt.title("Loss Plots")
plt.legend()
plt.savefig(f"{config.OUTPUT_PATH}/loss.png")
# build the translator module
print("[INFO] build the inference translator model...")
translator = Translator(
    encoder=translatorTrainer.encoder,
    decoder=translatorTrainer.decoder,
    sourceTextProcessor=sourceTextProcessor,
    targetTextProcessor=targetTextProcessor,
)
# save the model
print("[INFO] serialize the inference translator to disk...")
tf.saved_model.save(
    obj=translator,
    export_dir="translator",
    signatures={"serving_default": translator.translate}
)

