import tensorflow as tf
from tensorflow.keras.layers import AdditiveAttention
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GRU
from tensorflow.keras import Sequential


# defining Encode class
class Encoder(Layer):
    def __init__(self, sourceVocabSize=10000, embeddingDim=512, encUnits=512, **kwargs):
        super().__init__(**kwargs)
        self.sourceVocabSize = sourceVocabSize
        self.embeddingDim = embeddingDim
        self.encUnits = encUnits
    def build(self, inputShape):
        # the embedding layer converts token IDs to embedding vectors
        self.embedding = Embedding(input_dim=self.sourceVocabSize, output_dim=self.embeddingDim, mask_zero=True,)
        # the GRU layer processes the embedding vectors sequentially
        self.gru = Bidirectional(GRU(units=self.encUnits, return_sequences=True, return_state=True,
                        recurrent_initializer="glorot_uniform",))
    def get_config(self):
        return {"inputVocabSize": self.inputVocabSize, "embeddingDim": self.embeddingDim, "encUnits": self.encUnits,}
    def call(self, sourceTokens, state=None):
        sourceVectors = self.embedding(sourceTokens)
        sourceMask = self.embedding.compute_mask(sourceTokens)
        (encOutput, encFwdState, encBckState) = self.gru(inputs=sourceVectors, initial_state=state, mask=sourceMask)
        return (encOutput, encFwdState, encBckState, sourceMask)
        
# defining Attention(Bahdanau) class
class BahdanauAttention(Layer):
    def __init__(self, attnUnits, **kwargs):
        super().__init__(**kwargs)
        self.attnUnits = attnUnits
    def build(self, inputShape):
        # the dense layers projects the query and the value
        self.denseEncoderAnnotation = Dense(units=self.attnUnits, use_bias=False,)
        self.denseDecoderAnnotation = Dense(units=self.attnUnits, use_bias=False,)
        self.attention = AdditiveAttention()
    def get_config(self):
        return {"attnUnits": self.attnUnits,}
    def call(self, hiddenStateEnc, hiddenStateDec, mask):
        # grab the source and target mask
        sourceMask = mask[0]
        targetMask = mask[1]
        # pass the query and value through the dense layer
        encoderAnnotation = self.denseEncoderAnnotation(hiddenStateEnc)
        decoderAnnotation = self.denseDecoderAnnotation(hiddenStateDec)
        # apply attention to align the representations
        (contextVector, attentionWeights) = self.attention(inputs=[decoderAnnotation, hiddenStateEnc, encoderAnnotation],
                                                mask=[targetMask, sourceMask], return_attention_scores=True)
        return (contextVector, attentionWeights)

#defining Luong attention
class LuongAttention(Layer):
    def __init__(self, attnUnits, **kwargs):
        super().__init__(**kwargs)
        self.attnUnits = attnUnits

    def build(self, inputShape):
        self.attention = Attention()

    def get_config(self):
        return {"attnUnits": self.attnUnits,}

    def call(self, hiddenStateEnc, hiddenStateDec, mask):
        sourceMask = mask[0]
        targetMask = mask[1]
        (contextVector, attentionWeights) = self.attention(inputs=[hiddenStateDec, hiddenStateEnc, hiddenStateEnc],
                                                mask=[targetMask, sourceMask], return_attention_scores=True)
        return (contextVector, attentionWeights)
            
#defining Decoder class
class Decoder(Layer):
    def __init__(self, targetVocabSize, embeddingDim, decUnits, **kwargs):
        super().__init__(**kwargs)
        self.targetVocabSize = targetVocabSize
        self.embeddingDim = embeddingDim
        self.decUnits = decUnits
    def get_config(self):
        return {"targetVocabSize": self.targetVocabSize, "embeddingDim": self.embeddingDim, "decUnits": self.decUnits,}
    def build(self, inputShape):
        self.embedding = Embedding(input_dim=self.targetVocabSize, output_dim=self.embeddingDim, mask_zero=True,)
        self.gru = GRU(units=self.decUnits, return_sequences=True, return_state=True, recurrent_initializer="glorot_uniform")            

        self.attention = BahdanauAttention(self.decUnits)
        
        self.fwdNeuralNet = Sequential([Dense(units=self.decUnits, activation="tanh", use_bias=False,),
                                        Dense(units=self.targetVocabSize,),])
    def call(self, inputs, state=None):
        targetTokens = inputs[0]
        encOutput = inputs[1]
        sourceMask = inputs[2]
        targetVectors = self.embedding(targetTokens)
        targetMask = self.embedding.compute_mask(targetTokens)
        (decOutput, decState) = self.gru(inputs=targetVectors,
                initial_state=state, mask=targetMask)
        # use the GRU output as the query for the attention over the
        # encoder output
        (contextVector, attentionWeights) = self.attention(
                hiddenStateEnc=encOutput,
                hiddenStateDec=decOutput,
                mask=[sourceMask, targetMask],
        )
        contextAndGruOutput = tf.concat(
                [contextVector, decOutput], axis=-1)

        logits = self.fwdNeuralNet(contextAndGruOutput)
        return (logits, attentionWeights, decState)