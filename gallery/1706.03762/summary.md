# Attention Is All You Need: The End of Recurrence

Summary of *Attention Is All You Need* (https://arxiv.org/abs/1706.03762)

## 1. The Parallelization Bottleneck

- Recurrent models preclude parallelization within training examples due to their sequential nature.
  - p. 2 (Introduction): "This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths"
- Transformers allow significantly more parallelization, reaching state of the art in twelve hours on eight P100 GPUs.
  - p. 2 (Introduction): "The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs."

## 2. Transformer Architecture Details

- The Transformer eschews recurrence entirely, relying solely on attention mechanisms.
  - p. 2 (Introduction): "In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output."
- It is the first transduction model relying entirely on self-attention.
  - p. 2 (Background): "the Transformer is the first transduction model relying entirely on self-attention"

## 3. Transformer Encoder Structure

- The Transformer uses stacked self-attention and point-wise fully connected layers for both encoder and decoder.
  - p. 2 (Model Architecture): "The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder"
- The encoder stacks N=6 identical layers, each with multi-head self-attention.
  - p. 2 (Model Architecture): "The encoder is composed of a stack of N = 6 identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism"
- Self-attention keys, values, and queries originate from the previous layer's output.
  - p. 2 (Model Architecture): "In a self-attention layer all of the keys, values and queries come from the same place, in this case, the output of the previous layer"

## 4. Multi-Head Self-Attention Mechanism

- Self-attention relates different positions of a single sequence to compute its representation.
  - p. 2 (Background): "Self-attention... is an attention mechanism relating different positions of a single sequence"
- Multi-head attention allows the model to attend to information from different representation subspaces simultaneously.
  - p. 2 (Model Architecture): "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions"
- Decoder self-attention is masked to prevent leftward information flow and preserve the auto-regressive property.
  - p. 2 (Model Architecture): "We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. We implement this ... by masking out"

## 5. Transformer Translation Results

- The Transformer reaches new state-of-the-art translation quality after training for twelve hours on eight P100 GPUs.
  - p. 2 (Introduction): "can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs."
- The big model achieves a BLEU score of 28.4.
  - p. 8 (Results): "establishing a new state-of-the-art BLEU score of 28.4"
- The big model achieves a BLEU score of 41.0, outperforming all previously published single models.
  - p. 8 (Results): "our big model achieves a BLEU score of 41.0, outperforming all of the previously published single models"

## 6. Beyond Translation: Parsing

- Transformer achieves 92.7 F1 in semi-supervised English constituency parsing.
  - p. 8 (Results): "Transformer (4 layers) semi-supervised 92.7"
- It outperforms all models except Recurrent Neural Network Grammar.
  - p. 8 (Results): "yielding better results than all previously reported models with the exception of the Recurrent Neural Network Grammar [8]."
- Transformer beats Berkeley-Parser using only 40K training sentences.
  - p. 8 (Results): "the Transformer outperforms the Berkeley-Parser [29] even when training only on the WSJ training set of 40K sentences."

## 7. Limitations and Future Work

- Reducing key size dk hurts quality, suggesting compatibility is difficult.
  - p. 8 (Results): "reducing the attention key size dk hurts model quality. This suggests that determining compatibility is not easy"
- Future work extends the Transformer to other modalities using local attention mechanisms.
  - p. 10 (Conclusion): "We plan to extend the Transformer to problems involving input and output modalities other than text and to investigate local, restricted attention mechanisms"

## 8. The Transformer Takeaway

- Achieves new state-of-the-art results on WMT 2014 English-to-German and English-to-French tasks.
  - p. 10 (Conclusion): "On both WMT 2014 English-to-German and WMT 2014 English-to-French translation tasks, we achieve a new state of the art."
- Trains significantly faster than architectures based on recurrent or convolutional layers.
  - p. 10 (Conclusion): "For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers."
