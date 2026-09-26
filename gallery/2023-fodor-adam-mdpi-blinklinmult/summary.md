# Detect blinks faster with linear transformers

Summary of *BlinkLinMulT: Transformer-Based Eye Blink Detection* (https://adamfodor.com/pdf/2023_Fodor_Adam_MDPI_BlinkLinMulT.pdf)

## 1. The Blink Detection Challenge

- Detecting eye blinks is challenging but reveals deep fakes and driver drowsiness.
  - p. 1 (Introduction): "Detecting eye blinks is a challenging problem that can be used to solve a number of facial analysis tasks; it can reveal deep fake manipulation, detect driver drowsiness"
- Existing methods struggle with short blinks due to frame-wise classification.
  - p. 1 (Introduction): "Most approaches depend on face recognition and frame-wise eye state classiﬁcation... However, these methods may struggle to handle cases where blinks occur for only a few frames"

## 2. Introducing BlinkLinMulT

- The authors propose a fast transformer-based framework for eye blink detection.
  - p. 1 (Introduction): "In this work, we propose a fast transformer-based framework for eye blink detection that can effectively combine low- and high-level feature sequences"
- It combines low- and high-level feature sequences including head pose angles.
  - p. 1 (Introduction): "In this work, we propose a fast transformer-based framework for eye blink detection that can effectively combine low- and high-level feature sequences"
  - p. 1 (Introduction): "To our knowledge, this is the ﬁrst work to use transformer architecture and implement an efﬁcient fusion of input features, including head pose angles."

## 3. Linear Attention Architecture

- BlinkLinMulT uses a modified multi-modal transformer with linear attention to process RGB texture and landmarks.
  - p. 1 (Introduction): "We present a modiﬁed multi- modal transformer with linear attention (LinMulT) [21], which considers multiple inputs, such as RGB texture, iris and eye landmarks"
- The model replaces quadratic attention modules with linear versions for easier training while maintaining similar performance.
  - p. 4 (Materials and Methods): "we replaced the quadratic attention modules with linear versions [28], providing similar performance [21] while being easier to train."
- A backbone CNN extracts features from eye patches, which a linear transformer uses to estimate blinks for each eye independently.
  - p. 4 (Materials and Methods): "a backbone CNN extracts deep hidden representations of the RGB eye patches. These sets of features are utilized as inputs to a linear multimodal transformer that estimates blinks for each eye independently."

## 4. Superior Cross-Dataset Results

- BlinkLinMulT matches or beats state-of-the-art methods across various datasets.
  - p. 7 (Results and Discussion): "We show that our method generalizes well to different scenarios, and the performance of our network is comparable to or surpasses the performance of state-of-the-art methods"
- It achieves a perfect 1.0 F1 score for blink detection on the TalkingFace dataset.
  - p. 7 (Results and Discussion): "the problematic extreme blinks when the person looks down are detected in TalkingFace, achieving a 1.0 F1 score in the blink presence detection task."

## 5. Robustness via Union Training

- Union training improved F1 scores on CEW, ZJU, and RT-BENEimg benchmarks.
  - p. 7 (Results and Discussion): "Training on the union of the datasets improved on previous results measured in Section 4.2: the F1 scores changed from 0.995 to 0.997 on CEW, from 0.927 to 0.933 on ZJU, and from 0.883 to 0.913 RT-BENEimg."
- Eye state recognition and blink detection outperformed single dataset baselines.
  - p. 7 (Results and Discussion): "Table 6 shows that the eye state recognition and blink presence detection performance are consistently improved compared to those experiments, where only a single dataset is used for training."

## 6. Handling Extreme Head Poses

- Model performance drops at higher yaw angles.
  - p. 7 (Results and Discussion): "the performance of the model drops at higher yaw angles compared to the frontal faces"
- Performance slightly decreases at extreme orientations.
  - p. 15 (Appendix A): "Extreme yaw and pitch angles, when the participant is looking sideways or down, might cause slightly decreased performance."

## 7. Unified Training for Generalization

- The BlinkLinMulT model is trained once on the union of all datasets.
  - p. 7 (Results and Discussion): "While, in the literature, the proposed models are retrained for each dataset, our network is trained once and performs well on all the public blink benchmark datasets."
- This approach improves robustness and results across all individual unseen datasets.
  - p. 14 (Conclusions): "We performed cross-dataset evaluations to assess the robustness of BlinkLinMulT on unseen samples, demonstrating that a single network trained on a union of datasets improves results across all datasets indi-vidually."
