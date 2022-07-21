---
layout: post
title: RNN attention
---
[Notes are based on this blogpost](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)

# Traditional seq-2-seq models
Consist of encoder and decoder.

### Encoder:
RNN Encoder take 2 inputs at each timepoint: a word embedding, and a hidden state

Note that the word embedding and hidden states are both vectors

For each sentence, we sequentially feed in word i and the hidden state from the previous timepoint as input to the next state. In this way I believe we can re-use the same weights for each recurrent unit.

The encoder captures the full sentence into a **context** vector to be fed into the decoder.

### Traditional Decoder:
Decoder aims to unroll the context vector into a sequence of outputs of the same size as the input (translation) in autoregressive nature, through the same structure as the encoder.

 It turns out formation of this context vector is the bottleneck for RNNs, making it challenging to deal with long sentences --> how can you capture complex large sentences with a single vector?

### Attention Decoder
Attention decoders let the decoder focus on relevant parts of the input sequence "as needed".
Rather than feed in a single context vector, we feed in all of the encoder's hidden states to the decoder. _i.e what the encoder thought after seeing word 1, word 2, etc_

For this sequence of hidden states from the encoder, where hidden state i corresponds closest to input i, the decoder assigns an attention score to each hidden state at every timestep.

Multiply each hidden state by its softmaxed score to get a weighted average. This way we look at all the inputs rather than just the final one.

This attention weighting is done at every timestep, so attention may differ depending on the hidden state of the decoder.

At each timestep in the decoder, we feed the attention-weighted hidden states concatenated with the decoder's hidden state at the previous timepoint to generate a new hidden state vector. we feed this hidden state vector into 1) a feedforward neural net to generate an output and 2) the next decoder unit.

This way, we can both consider what our decoder thinks about the previous output, alongside what the encoder thought about the input
