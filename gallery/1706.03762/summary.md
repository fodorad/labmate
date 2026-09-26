# Attention Is All You Need: No Recurrence Required

Summary of *Attention Is All You Need* (https://arxiv.org/abs/1706.03762)

## 1. The Parallelization Bottleneck

- Recurrent models preclude parallelization due to their sequential nature.
  - p. 2 (Introduction): "This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths"
- Transformers allow significantly more parallelization than recurrent models.
  - p. 2 (Introduction): "The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs."

## 2. Introducing the Transformer Architecture

- The authors propose the Transformer, a model architecture that relies entirely on attention mechanisms.
  - p. 2 (Introduction): "In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output."
- It eschews recurrence and is the first transduction model relying entirely on self-attention.
  - p. 2 (Background): "the Transformer is the first transduction model relying entirely on self-attention to compute representations"

## 3. How Self-Attention Works

- Self-attention relates different positions within a single sequence.
  - p. 2 (Background): "Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence"
- Scaled Dot-Product Attention uses dot products, scaling, and softmax to compute weights.
  - p. 2 (Model Architecture): "We compute the dot products of the query with all keys, divide each by √dk, and apply a softmax function to obtain the weights on the values."
- Multi-head attention allows the model to attend to different representation subspaces simultaneously.
  - p. 2 (Model Architecture): "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions."

## 4. Transformer Translation Results

- Reaches state-of-the-art translation quality after twelve hours on eight P100 GPUs.
  - p. 2 (Introduction): "can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs."
- Achieves better BLEU scores than previous models at lower training cost.
  - p. 7 (Training): "Table 2: The Transformer achieves better BLEU scores than previous state-of-the-art models on the English-to-German and English-to-French newstest2014 tests at a fraction of the training cost."

## 5. Big Transformer Results

- The big Transformer achieves a BLEU score of 28.4 on WMT English-to-German.
  - p. 8 (Results): "the big transformer model (Transformer (big) in Table 2) outperforms the best previously reported models (including ensembles) by more than 2.0 BLEU, establishing a new state-of-the-art BLEU score of 28.4."
- It scores 41.0 BLEU on English-to-French at less than one-quarter the training cost.
  - p. 8 (Results): "our big model achieves a BLEU score of 41.0, outperforming all of the previously published single models, at less than 1/4 the training cost of the previous state-of-the-art model."

## 6. Limitations and Future Directions

- Reduced effective resolution occurs due to averaging attention-weighted positions.
  - p. 2 (Background): "at the cost of reduced effective resolution due to averaging attention-weighted positions"
- Future work extends the Transformer to other modalities and investigates local attention mechanisms.
  - p. 10 (Conclusion): "We plan to extend the Transformer to problems involving input and output modalities other than text and to investigate local, restricted attention mechanisms"

## 7. Why This Matters for ML Engineers

- Self-attention enables greater parallelization than recurrent layers.
  - p. 6 (Why Self-Attention): "a self-attention layer connects all positions with a constant number of sequentially executed operations, whereas a recurrent layer requires O(n) sequential operations."
- Attention heads often learn syntactic and semantic structures.
  - p. 6 (Why Self-Attention): "many appear to exhibit behavior related to the syntactic and semantic structure of the sentences."
