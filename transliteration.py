#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xml.etree.ElementTree as ET
import numpy as np
# In[3]:

UNIT = 256

import tensorflow as tf
class Vector:
    def __init__(self, corpus):
        self.word2id = {}
        self.id2word = {}
        self.pad = "<PAD>"
        self.sos = "<SOS>"
        self.eos = "<EOS>"
        self.unk = "<UNK>"
        
        self.ipad = 0
        self.isos = 1
        self.ieos = 2
        self.iunk = 3
        
        self.word2id[self.pad] = 0
        self.word2id[self.sos] = 1
        self.word2id[self.eos] = 2
        self.word2id[self.unk] = 3
        
        self.id2word[0] = self.pad
        self.id2word[1] = self.sos
        self.id2word[2] = self.eos
        self.id2word[3] = self.unk
        
        curr_id = 4
        self.chars = {}
        self.max_len = 0
        for word in corpus:
            self.max_len = max(self.max_len, len(word))
            word = word.lower()
            for char in word.strip():
                self.chars[char] = 1 + self.chars.get(char,0)
        
        self.chars={k:v for k,v in sorted(self.chars.items(),
                key=lambda item:item[1], reverse=True)}
        for key in self.chars.keys():
            self.word2id[key] = curr_id
            self.id2word[curr_id] = key
            curr_id+=1
        self.vocab_size = len(self.word2id)
    
    def encode(self, word):
        word = word.lower().strip()
        res = [self.word2id.get(char,self.iunk) for char in word]
        res.insert(0,self.isos)
        res.append(self.ieos)
        res = res + [self.ipad for i in range(len(res), self.max_len+3)]
        return res[:self.max_len]
    
    def decode(self, vector):
        res = []
        for i in vector:
            if i in [0,1]:
                continue;
            if i == 2:
                break;
            res.append(self.id2word.get(i, self.unk))
        return ''.join(res)
        
        


# In[4]:




def process_text(context, target):
    targ_in = target[:,:-1]
    targ_out = target[:,1:]
    return (context, targ_in), targ_out
  



# In[10]:



class Encoder(tf.keras.layers.Layer):
      def __init__(self, vocab_size, units):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.units = units
        
        # The embedding layer converts tokens to vectors
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.units,
                                                   mask_zero=True)
    
        # The RNN layer processes those vectors sequentially.
        self.rnn = tf.keras.layers.Bidirectional(
            merge_mode='sum',
            layer=tf.keras.layers.GRU(self.units,
                                # Return the sequence and state
                                return_sequences=True,
                                recurrent_initializer='glorot_uniform'))
    
      def call(self, x):
        
    
        # 2. The embedding layer looks up the embedding vector for each token.
        x = self.embedding(x)
        
        # 3. The GRU processes the sequence of embeddings.
        x = self.rnn(x)
        
        # 4. Returns the new sequence of embeddings.
        return x
    
      def convert_input(self, texts, en):
        texts = list(map(en.encode, texts))
        texts = tf.convert_to_tensor(np.array(texts))
        context = self(texts)
        return context
    




class CrossAttention(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(key_dim=units, num_heads=1, **kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

    def call(self, x, context):
        
        attn_output, attn_scores = self.mha(
            query=x,
            value=context,
            return_attention_scores=True)
        
        
        # Cache the attention scores for plotting later.
        attn_scores = tf.reduce_mean(attn_scores, axis=1)
        self.last_attention_weights = attn_scores
    
        x = self.add([x, attn_output])
        x = self.layernorm(x)
    
        return x




# In[17]:


class Decoder(tf.keras.layers.Layer):
    @classmethod
    def add_method(cls, fun):
        setattr(cls, fun.__name__, fun)
        return fun

    def __init__(self, vocab_size, units):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.units = units
        self.start_token = 1
        self.end_token = 2
    
        # 1. The embedding layer converts token IDs to vectors
        self.embedding = tf.keras.layers.Embedding(self.vocab_size,
                                                   units, mask_zero=True)
    
        # 2. The RNN keeps track of what's been generated so far.
        self.rnn = tf.keras.layers.GRU(units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
    
        # 3. The RNN output will be the query for the attention layer.
        self.attention = CrossAttention(units)
    
        # 4. This fully connected layer produces the logits for each
        # output token.
        self.output_layer = tf.keras.layers.Dense(self.vocab_size)


# In[18]:


@Decoder.add_method
def call(self,
         context, x,
         state=None,
         return_state=False):  
    
    
    # 1. Lookup the embeddings
    x = self.embedding(x)
    
    # 2. Process the target sequence.
    x, state = self.rnn(x, initial_state=state)
    
    # 3. Use the RNN output as the query for the attention over the context.
    x = self.attention(x, context)
    self.last_attention_weights = self.attention.last_attention_weights
    
    # Step 4. Generate logit predictions for the next token.
    logits = self.output_layer(x)
    
    if return_state:
        return logits, state
    else:
        return logits



@Decoder.add_method
def get_initial_state(self, context):
    batch_size = tf.shape(context)[0]
    start_tokens = tf.fill([batch_size, 1], self.start_token)
    done = tf.zeros([batch_size, 1], dtype=tf.bool)
    embedded = self.embedding(start_tokens)
    return start_tokens, done, self.rnn.get_initial_state(embedded)[0]


# In[22]:


@Decoder.add_method
def tokens_to_text(self, tokens, hi):
    tokens = tokens.numpy()
    words = [hi.decode(token) for token in tokens]
    return words


# In[23]:


@Decoder.add_method
def get_next_token(self, context, next_token, done, state, temperature = 0.0):
    logits, state = self(
        context, next_token,
        state = state,
        return_state=True) 
  
    if temperature == 0.0:
        next_token = tf.argmax(logits, axis=-1)
    else:
        logits = logits[:, -1, :]/temperature
        next_token = tf.random.categorical(logits, num_samples=1)

    # If a sequence produces an `end_token`, set it `done`
    done = done | (next_token == self.end_token)
    # Once a sequence is done it only produces 0-padding.
    next_token = tf.where(done, tf.constant(0, dtype=tf.int64), next_token)
  
    return next_token, done, state


# In[24]:


# In[25]:


class Translator(tf.keras.Model):
    @classmethod
    def add_method(cls, fun):
        setattr(cls, fun.__name__, fun)
        return fun

    def __init__(self, units,
               source_vectorizer,
               target_vectorizer):
        super().__init__()
        # Build the encoder and decoder
        self.source_vectorizer = source_vectorizer
        self.target_vectorizer = target_vectorizer
        encoder = Encoder(source_vectorizer.vocab_size, units)
        decoder = Decoder(target_vectorizer.vocab_size, units)

        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        context, x = inputs
        context = self.encoder(context)
        logits = self.decoder(context, x)

        #TODO(b/250038731): remove this
        try:
          # Delete the keras mask, so keras doesn't scale the loss+accuracy. 
            del logits._keras_mask
        except AttributeError:
            pass

        return logits


# In[26]:




# In[27]:


def masked_loss(y_true, y_pred):
    # Calculate the loss for each item in the batch.
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    loss = loss_fn(y_true, y_pred)

    # Mask off the losses on padding.
    mask = tf.cast(y_true != 0, loss.dtype)
    loss *= mask

    # Return the total.
    return tf.reduce_sum(loss)/tf.reduce_sum(mask)


# In[28]:


def masked_acc(y_true, y_pred):
    # Calculate the loss for each item in the batch.
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, y_true.dtype)
    
    match = tf.cast(y_true == y_pred, tf.float32)
    mask = tf.cast(y_true != 0, tf.float32)
    
    return tf.reduce_sum(match)/tf.reduce_sum(mask)


# In[29]:






@Translator.add_method
def translate(self,
              texts, *,
              max_length=50,
              temperature=0.0):
  # Process the input texts
    context = self.encoder.convert_input(texts, self.source_vectorizer)
    batch_size = tf.shape(texts)[0]

    # Setup the loop inputs
    tokens = []
    attention_weights = []
    next_token, done, state = self.decoder.get_initial_state(context)

    for _ in range(max_length):
        # Generate the next token
        next_token, done, state = self.decoder.get_next_token(
            context, next_token, done,  state, temperature)
        
        # Collect the generated tokens
        tokens.append(next_token)
        attention_weights.append(self.decoder.last_attention_weights)
    
        if tf.executing_eagerly() and tf.reduce_all(done):
            break

    # Stack the lists of tokens and attention weights.
    tokens = tf.concat(tokens, axis=-1)   # t*[(batch 1)] -> (batch, t)
    self.last_attention_weights = tf.concat(attention_weights, axis=1)  # t*[(batch 1 s)] -> (batch, t s)

    result = self.decoder.tokens_to_text(tokens,self.target_vectorizer)
    return result


# In[40]:




# In[ ]:




