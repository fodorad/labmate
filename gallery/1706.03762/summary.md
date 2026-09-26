# Attention Is All You Need: No Recurrence Required

Summary of *Attention Is All You Need* (https://arxiv.org/abs/1706.03762)

## 1. Modeling Dependencies Without Recurrence

- Attention models dependencies regardless of distance in input or output sequences.
  - p. 2 (Introduction): "Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various tasks, allowing modeling of dependencies without regard to their distance in the input or output sequences [2, 19]."
- The Transformer relies entirely on attention, eschewing recurrence to draw global dependencies.
  - p. 2 (Introduction): "In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output."
- Future plans include extending the model to images, audio, and video modalities.
  - p. 10 (Conclusion): "We plan to extend the Transformer to problems involving input and output modalities other than text and to investigate local, restricted attention mechanisms to efficiently handle large inputs and outputs such as images, audio and video."

## 2. Sequential Bottlenecks in Previous Models

- Recurrent models preclude parallelization within training examples.
  - p. 2 (Introduction): "This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples."
- Previous convolutional models make it difficult to learn dependencies between distant positions.
  - p. 2 (Background): "This makes it more difficult to learn dependencies between distant positions [12]."
- A single convolutional layer does not connect all pairs of input and output positions.
  - p. 6 (Why Self-Attention): "A single convolutional layer with kernel width k < n does not connect all pairs of input and output positions. Doing so requires a stack of O(n/k) convolutional layers"

## 3. Pure Self-Attention Architecture Design

- The Transformer uses stacked self-attention and point-wise fully connected layers for both encoder and decoder.
  - p. 2 (Model Architecture): "The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder"
- It employs h=8 parallel attention heads, each with dimension dk=dv=64, to attend to different subspaces simultaneously.
  - p. 2 (Model Architecture): "In this work we employ h = 8 parallel attention layers, or heads. For each of these we use dk = dv = dmodel/h = 64."
- Self-attention connects all positions with constant sequential operations, unlike recurrent layers requiring O(n) operations.
  - p. 6 (Why Self-Attention): "a self-attention layer connects all positions with a constant number of sequentially executed operations, whereas a recurrent layer requires O(n) sequential operations."

## 4. State-of-the-Art Translation and Parsing Results

- The big Transformer model achieves a state-of-the-art BLEU score of 28.4 on WMT 2014 English-to-German translation.
  - p. 8 (Results): "On the WMT 2014 English-to-German translation task, the big transformer model (Transformer (big) in Table 2) outperforms the best previously reported models (including ensembles) by more than 2.0 BLEU, establishing a new state-of-the-art BLEU score of 28.4."
- On WMT 2014 English-to-French, the model reaches a BLEU score of 41.0 at less than one-quarter the training cost.
  - p. 8 (Results): "On the WMT 2014 English-to-French translation task, our big model achieves a BLEU score of 41.0, outperforming all of the previously published single models, at less than 1/4 the training cost of the previous state-of-the-art model."
