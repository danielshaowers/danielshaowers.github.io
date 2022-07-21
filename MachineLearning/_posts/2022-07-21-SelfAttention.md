---
layout: post
title: Self Attention
---


# Self-attention vs Traditional attention
Useful resource: [The illustrated transformer](https://jalammar.github.io/illustrated-transformer/)
Normal attention (with clam) examines the relationship of each patch towards the final prognosis.

In contrast, self-attention examines the relationship of each patch with each other.

## Attention heads
Attention heads let us break up the input to examine the full sequence in a parallel fashion.

#### Self-attention is like database lookup
Self-attention is analogous to a database lookup, but generalized to be non-discrete.

Each self-attention head consists of **key, query, value**
This idea is similar to a hash table:
 - The **key** is similar to a hash lookup of what has been memorized
 - The **query** is what is being looked up.
 - The **value** is the output

In traditional database, we simply feed in the query, which is equal to the key, and we receive a single scalar value. But in self-attention, the key and value are both embeddings!

We can intuitively think of the key as the *qualities* of an item, and the value as the *meaning* of the item.

Meanwhile the query is just the raw input. So we need a weight matrix to transform the query into the same embedding space as the keys.

Once we transform the query into the same vector space as the keys, we can look up the key which is most similar to the query. In a database where the keys and queries are identical, we would receive a single value.

In this generalized case where everything is a vector, there is no exact match. Instead we want a weighted average of values, based on the keys which are most similar to our query.

We measure similarity according to the **general dot product** of the transformed query.
$s_i = q^T W k_i$ , where $k_i$ is the $i$'th key, $s_i$ is the similarity between the query and key $k_i$  W is a matrix to transform the query into the key's embedding space.
$s_i$ is the similarity score which lets us get the result of feeding our query into the attention head.
We need to normalize $s_i$ to prevent any diminishing/exploding gradients

$$a_i = softmax(s_i) = \dfrac{exp(s_i)}{\sum_j s_j}$$ $$out = \sum_{i} a_iv_i$$
In the case of machine translation, we set $k_j = v_j$. My intuition is that, since we are translating one language to another, the ideal operation would be to feed in query (an english word), "look up" the embedding for the translated word (a single key), and directly output that key embedding as our value.
In contrast if we were trying to predict disease state then we wouldn't necessarily want $v=k$. Say we feed in a resnet image embedding as our query. We convert it to the key embedding space which represents a disease state (receptor status, lymph node involvement). Then we want to identify the prognosis, which has a value.

In this example, the _queries_ and _values_ are quite unique, but the overall prognosis is not, so we wouldnt want a unique value. I.e there are many different disease states which can lead to a prognosis of high recurrence likelihood.


In transformers, we feed in the full sentence at once. Unlike RNNs where we feed a single word at a time.

### Multihead attention in transformers
In _self attention_, we compute the attention by treating each word as a query, and every other word in the sentence as a value/key (again the keys=values).
***
So _self attention vs normal attention differs in what we define as a value/key_. **In self attention, we let entries in the sequence function as the values and keys**
***
Multihead attention breaks up the input sequence into $h$ sections. At each attention head, we compute the pairwise attention between each word in that head, and the embeddings of other words in that head. This gives us one attention-weighted output per word in the sequence, which is why an input of $\dfrac{n}{h}$ at each head gives $\dfrac{n}{h}$ outputs as well.
These weighted outputs essentially are word embeddings, contextualized by other words within the attention head.

Note that so far, we haven't achieved global interactions. The word at the start of the sentence never sees the last word in the sentence.

### AdaNorm
After a multihead attention layer return the contextualized word embeddings, we add a residual layer from the input to the multihead layer to the attention layer and normalize it.

### Feedforward and AdaNorm
Finally, we feed the result into a feedforward network to encode the sequential information into a smaller embedding, and normalize the result.

We then stack these encoder layers. This effectively increases the receptive field of our attention heads, since the next i'th input to the attention head is actually the i'th output of the previous layers' attention head!
