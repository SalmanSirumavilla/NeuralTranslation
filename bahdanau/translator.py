import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import StringLookup
from tensorflow import keras

class TrainTranslator(keras.Model):
    def __init__(self, encoder, decoder, sourceTextProcessor,
            targetTextProcessor, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.sourceTextProcessor = sourceTextProcessor
        self.targetTextProcessor = targetTextProcessor
    def _preprocess(self, sourceText, targetText):
        # convert the text to token IDs
        sourceTokens = self.sourceTextProcessor(sourceText)
        targetTokens = self.targetTextProcessor(targetText)
        return (sourceTokens, targetTokens)
    def _calculate_loss(self, sourceTokens, targetTokens):
        # encode the input text token IDs
        (encOutput, encFwdState, encBckState, sourceMask) = self.encoder(sourceTokens=sourceTokens)
        # initialize the decoder's state to the encoder's final state
        decState = tf.concat([encFwdState, encBckState], axis=-1)
        (logits, attentionWeights, decState) = self.decoder(inputs=[targetTokens[:, :-1], encOutput, sourceMask], state=decState,)
        # calculate the batch loss
        yTrue = targetTokens[:, 1:]
        yPred = logits
        batchLoss = self.loss(yTrue=yTrue, yPred=yPred)
        # return the batch loss
        return batchLoss
        
        
    @tf.function(
        input_signature=[[
            tf.TensorSpec(dtype=tf.string, shape=[None]),
            tf.TensorSpec(dtype=tf.string, shape=[None])
        ]])
    def train_step(self, inputs):
        # grab the source and the target text from the inputs
        (sourceText, targetText) = inputs
        # pre-process the text into token IDs
        (sourceTokens, targetTokens) = self._preprocess(
            sourceText=sourceText,
            targetText=targetText
        )
        # use gradient tape to track the gradients
        with tf.GradientTape() as tape:
            # calculate the batch loss
            loss = self._calculate_loss(
                sourceTokens=sourceTokens,
                targetTokens=targetTokens,
            )
            # normalize the loss
            averageLoss = (
                loss / tf.reduce_sum(
                    tf.cast((targetTokens != 0), tf.float32)
                )
            )
        # apply an optimization step on all the trainable variables
        variables = self.trainable_variables 
        gradients = tape.gradient(averageLoss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        # return the batch loss
        return {"batch_loss": averageLoss}

    @tf.function(
    input_signature=[[
        tf.TensorSpec(dtype=tf.string, shape=[None]),
        tf.TensorSpec(dtype=tf.string, shape=[None])
    ]])
    def test_step(self, inputs):
        # grab the source and the target text from the inputs
        (sourceText, targetText) = inputs
        # pre-process the text into token IDs
        (sourceTokens, targetTokens) = self._preprocess(
            sourceText=sourceText,
            targetText=targetText
        )
        # calculate the batch loss
        loss = self._calculate_loss(
            sourceTokens=sourceTokens,
            targetTokens=targetTokens,
        )
        # normalize the loss
        averageLoss = (
            loss / tf.reduce_sum(
                tf.cast((targetTokens != 0), tf.float32)
            )
        )
        # return the batch loss
        return {"batch_loss": averageLoss}
        
class Translator(tf.Module):
    def __init__(self, encoder, decoder, sourceTextProcessor,
        targetTextProcessor):
        # initialize the encoder, decoder, source text processor, and
        # target text processor
        self.encoder = encoder
        self.decoder = decoder
        self.sourceTextProcessor = sourceTextProcessor
        self.targetTextProcessor = targetTextProcessor
        # initialize index to string layer
        self.stringFromIndex = StringLookup(
            vocabulary=targetTextProcessor.get_vocabulary(),
            mask_token="",
            invert=True
        )
        # initialize string to index layer
        indexFromString = StringLookup(
            vocabulary=targetTextProcessor.get_vocabulary(),
            mask_token="",
        )
        # generate IDs for mask tokens
        tokenMaskIds = indexFromString(["", "[UNK]", "[START]"]).numpy()
        tokenMask = np.zeros(
            [indexFromString.vocabulary_size()],
            dtype=np.bool
        )
        tokenMask[np.array(tokenMaskIds)] = True
        # initialize the token mask, start token, and end token
        self.tokenMask = tokenMask
        self.startToken = indexFromString(tf.constant("[START]"))
        self.endToken = indexFromString(tf.constant("[END]"))
        
    def tokens_to_text(self, resultTokens):
        # decode the token from index to string
        resultTextTokens = self.stringFromIndex(resultTokens)
        # format the result text into a human readable format
        resultText = tf.strings.reduce_join(inputs=resultTextTokens, axis=1, separator=" ")
        resultText = tf.strings.strip(resultText)
        return resultText
        
    def sample(self, logits, temperature):
        # reshape the token mask
        tokenMask = self.tokenMask[tf.newaxis, tf.newaxis, :]
        # set the logits for all masked tokens to -inf, so they are
        # never chosen
        logits = tf.where(
            condition=self.tokenMask,
            x=-np.inf,
            y=logits
        )
        # check if the temperature is set to 0
        if temperature == 0.0:
            # select the index for the maximum probability element
            newTokens = tf.argmax(logits, axis=-1)
        # otherwise, we have set the temperature
        else: 
            # sample the index for the element using categorical
            # probability distribution
            logits = tf.squeeze(logits, axis=1)
            newTokens = tf.random.categorical(logits / temperature,
                num_samples=1
            )
        # return the new tokens
        return newTokens
        
    @tf.function(input_signature=[tf.TensorSpec(dtype=tf.string,
        shape=[None])])
    def translate(self, sourceText, maxLength=50, returnAttention=True,
        temperature=1.0):
        # grab the batch size
        batchSize = tf.shape(sourceText)[0]
        # encode the source text to source tokens and pass them
        # through the encoder
        sourceTokens = self.sourceTextProcessor(sourceText)
        (encOutput, encFwdState, encBckState, sourceMask) = self.encoder(
            sourceTokens=sourceTokens)
        # initialize the decoder state and the new tokens
        decState = tf.concat([encFwdState, encBckState], axis=-1)
        newTokens = tf.fill([batchSize, 1], self.startToken)
        # initialize the result token, attention, and done tensor
        # arrays
        resultTokens = tf.TensorArray(tf.int64, size=1,
            dynamic_size=True)
        attention = tf.TensorArray(tf.float32, size=1,
            dynamic_size=True)
        done = tf.zeros([batchSize, 1], dtype=tf.bool)
        # loop over the maximum sentence length
        for i in tf.range(maxLength):
            # pass the encoded tokens through the decoder
            (logits, attentionWeights, decState) = self.decoder(
                inputs=[newTokens, encOutput, sourceMask],
                state=decState,
            )
            # store the attention weights and sample the new tokens
            attention = attention.write(i, attentionWeights)
            newTokens = self.sample(logits, temperature)
            # if the new token is the end token then set the done
            # flag
            done = done | (newTokens == self.endToken)
            # replace the end token with the padding
            newTokens = tf.where(done, tf.constant(0, dtype=tf.int64),
                newTokens)
            # store the new tokens in the result
            resultTokens = resultTokens.write(i, newTokens)
            # end the loop once done
            if tf.reduce_all(done):
                break
        # convert the list of generated token IDs to a list of strings
        resultTokens = resultTokens.stack()
        resultTokens = tf.squeeze(resultTokens, -1)
        resultTokens = tf.transpose(resultTokens, [1, 0])
        resultText = self.tokens_to_text(resultTokens)
        # check if we have to return the attention weights
        if returnAttention:
            # format the attention weights
            attentionStack = attention.stack()
            attentionStack = tf.squeeze(attentionStack, 2)
            attentionStack = tf.transpose(attentionStack, [1, 0, 2])
            # return the text result and attention weights
            return {"text": resultText, "attention": attentionStack}
        # otherwise, we will just be returning the result text
        else:
            return {"text": resultText}
        