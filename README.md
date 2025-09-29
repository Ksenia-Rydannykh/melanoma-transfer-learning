# Melanoma Detection using Transfer Learning with MobileNetV2

## Code and Notebook  

The full implementation, including data preprocessing, model training, and evaluation, is available in this Jupyter Notebook:  
👉 [Melanoma_TransferLearning_MobileNetV2.ipynb](Melanoma_TransferLearning_MobileNetV2.ipynb)

## Introduction  

This project applies **transfer learning with MobileNetV2** to the task of melanoma detection. The goal is to adapt pretrained ImageNet features for reliable classification of skin lesions, improving early diagnosis while keeping the model lightweight and efficient.  

The work is based on the [Kaggle Melanoma Cancer Dataset](https://www.kaggle.com/datasets/bhaveshmittal/melanoma-cancer-dataset/data), which contains **13,900 dermoscopic images (224×224 px)** of benign and malignant cases. Its diversity makes it a strong benchmark for testing transfer learning approaches in medical image analysis.  

## Model Architecture  
I implemented and evaluated three configurations of the **MobileNetV2** convolutional neural network, all based on a common architecture that utilized the pretrained MobileNetV2 backbone as a feature extractor. The base model was initialized with **ImageNet weights** and configured without the top classification layer (`include_top=False`). A custom classification head was appended to adapt the model for binary classification.  

**Architecture details**:  
- **Input Preprocessing:** Images resized to 224×224×3 and normalized to [0,1].  
- **Backbone:** Pretrained MobileNetV2 (`weights='imagenet'`, `include_top=False`).  
- **Custom Classification Head:**  
  - `GlobalAveragePooling2D`  
  - `Dense(128, activation='relu')`  
  - `Dropout(0.3)`  
  - `Dense(1, activation='sigmoid')`  

**Training setup:**  
- Optimizer: Adam (learning rate = 0.001)  
- Loss function: Binary Cross-Entropy  
- Metric: Accuracy  

## Experiments  
Three experimental configurations were tested to evaluate the effect of transfer learning and fine-tuning:  

| Configuration                                | Epochs | Accuracy | F1-score | Recall (Malignant) |
|---------------------------------------------|--------|----------|----------|---------------------|
| Frozen MobileNetV2 base + custom head        | 10     | **88.05%** | 0.8757   | 0.89                |
| Partially fine-tuned MobileNetV2 (last 30 layers unfrozen) | 10     | **88.91%** | 0.8848   | **0.90**            |
| Fine-tuned MobileNetV2 with data augmentation | 30     | **88.05%** | 0.8680   | 0.83                |

### Analysis  

The **frozen base model with only a custom classification head** produced a strong baseline, successfully transferring general ImageNet features but showing limited adaptation to the unique visual characteristics of melanoma lesions.  

The **partially fine-tuned model (last 30 layers unfrozen)** provided the best results. Allowing deeper layers to adapt to lesion-specific features improved generalization and delivered the highest recall, which is especially critical in medical applications where false negatives must be minimized.  

The **fine-tuned model with additional data augmentation**, even when trained for longer, demonstrated steady learning but did not surpass the selectively fine-tuned configuration. The increased complexity and regularization from augmentation likely slowed convergence and reduced sensitivity to malignant cases.  

### Final Conclusion  

Among the three strategies, the **partially fine-tuned MobileNetV2** demonstrated the most effective trade-off between accuracy, F1-score, and recall. This highlights the importance of selective fine-tuning in transfer learning, as it allows the model to adapt deeper layers to domain-specific features without over-regularization. For melanoma detection tasks, where minimizing false negatives is critical, this configuration provides the most reliable results.  


## Contributions  
This project contributes by:  
- Applying and comparing three distinct MobileNetV2 configurations on a curated melanoma dataset.  
- Emphasizing **recall** as a critical metric for clinical tasks.  
- Analyzing the impact of **layer unfreezing, data augmentation, and training duration**.  
- Providing practical insights for deploying lightweight CNNs in sensitive domains like dermatology.  

