# Detect blinks faster with linear attention transformers

Summary of *BlinkLinMulT: Transformer-Based Eye Blink Detection* (https://adamfodor.com/pdf/2023_Fodor_Adam_MDPI_BlinkLinMulT.pdf)

## 1. Eye Blink Detection for Safety and Security

- Detecting eye blinks solves facial analysis tasks like deep fake detection and driver drowsiness monitoring.
  - p. 1 (Introduction): "Detecting eye blinks is a challenging problem that can be used to solve a number of facial analysis tasks; it can reveal deep fake manipulation, detect driver drowsiness, measure the level of attention and eye fatigue during task performance, and health disorders, among others."
- Methods are categorized into feature-based, appearance-based, and motion-based approaches.
  - p. 1 (Introduction): "Three categories emerged in the last decade: (i) feature-based methods use predetermined higher-level features like iris and eye landmark distances, (ii) appearance-based methods utilize low-level eye representations (e.g., RGB texture) and then adopt learning algorithms to classify eye states, and (iii) motion-based methods use input representations that encode motion information or a sequence of eye-related features to determine eye states."
- The approach is evaluated on blink presence detection and eye state recognition tasks using multiple public benchmark datasets.
  - p. 1 (Introduction): "The proposed approach is evaluated on blink presence detection and eye state recognition tasks and multiple public benchmark datasets."

## 2. Challenges in Handling Noisy Data and Short Blinks

- Existing face recognition and frame-wise classification methods struggle with short blinks or noisy data.
  - p. 1 (Introduction): "Most approaches depend on face recognition and frame-wise eye state classiﬁcation, even with the rise of transformer-based sequence models in many other research areas. However, these methods may struggle to handle cases where blinks occur for only a few frames or where input data are noisy or incomplete."
- The CEW dataset lacks pre-defined splits, requiring a 10-fold cross-validation scheme to measure performance.
  - p. 6 (Datasets): "The CEW dataset does not have pre-deﬁned training and test images; therefore, similar to others in the literature, we employed a 10-fold cross-validation (CV) scheme to measure the performance of backbone models."
- Extreme head pose angles cause most errors due to eyelid visibility and landmark precision issues.
  - p. 7 (Results and Discussion): "Most of the errors originated from this extreme head pose because the eyelids are more visible, they move over a shorter distance, and the landmark-based features are less precise considering the available resolution."

## 3. BlinkLinMulT: Linear Attention for Multi-Modal Input

- Modified multi-modal transformer considers RGB texture, iris and eye landmarks, ear, and head pose angles.
  - p. 1 (Introduction): "We present a modiﬁed multi-modal transformer with linear attention (LinMulT) [21], which considers multiple inputs, such as RGB texture, iris and eye landmarks, ear, and head pose angles."
- Cross-modal transformers translate between landmark features and RGB texture embeddings.
  - p. 4 (Materials and Methods): "Cross-modal transformers translate one information source (e.g., β, here associated with the landmark features) to another one (e.g., α, here associated with the RGB texture)"
- Linear multimodal transformer estimates blinks for each eye independently.
  - p. 4 (Materials and Methods): "These sets of features are utilized as inputs to a linear multimodal transformer that estimates blinks for each eye independently."

## 4. Superior F1 Scores on RT-BENEimg and TalkingFace

- ResNet50 achieved a 0.912 F1 score on RT-BENEimg, outperforming DenseNet121's 0.883.
  - p. 7 (Results and Discussion): "DenseNet121 performed well with 0.883, but ResNet50 outperformed the other models by a considerable margin with a 0.912 F1 score and 0.946 AP."
- Training on the union of datasets improved F1 scores: CEW rose to 0.997, ZJU to 0.933, and RT-BENEimg to 0.913.
  - p. 7 (Results and Discussion): "Training on the union of the datasets improved on previous results measured in Section 4.2: the F1 scores changed from 0.995 to 0.997 on CEW, from 0.927 to 0.933 on ZJU, and from 0.883 to 0.913 RT-BENEimg."
- A model achieved a perfect 1.0 F1 score on the TalkingFace dataset for blink presence detection.
  - p. 7 (Results and Discussion): "in the last case, where all feature sequences are used, the problematic extreme blinks when the person looks down are detected in TalkingFace, achieving a 1.0 F1 score in the blink presence detection task."
