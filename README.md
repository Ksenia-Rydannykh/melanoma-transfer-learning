# Melanoma Detection using Transfer Learning with MobileNetV2

## Introduction  
Transfer learning has become a powerful approach in computer vision, especially for medical imaging tasks where annotated datasets are often limited. Instead of training models from scratch, pretrained convolutional neural networks (CNNs) can be fine-tuned to leverage knowledge gained from large-scale datasets like ImageNet. This strategy enables faster convergence, reduces computational costs, and often yields higher accuracy in specialized domains such as disease classification.  

In this project, transfer learning is applied using the MobileNetV2 architecture to the task of melanoma detection, where early and reliable diagnosis is crucial for improving patient outcomes.  

## Dataset  
The dataset used in this project is sourced from [Kaggle’s Melanoma Cancer Dataset](https://www.kaggle.com/datasets/bhaveshmittal/melanoma-cancer-dataset/data). It contains **13,900 high-quality dermoscopic images**, uniformly resized to **224×224 pixels**, representing both benign and malignant skin lesions.  

The dataset was curated to support research in dermatology and computer-aided diagnostics, with the goal of enabling the development of machine learning models that can distinguish between healthy and cancerous tissue. Its diversity of lesion appearances provides a realistic challenge for classification tasks, making it a suitable benchmark for evaluating the effectiveness of transfer learning approaches in medical image analysis.  

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
1. **Fully frozen base model**
   > All layers of the MobileNetV2 base were frozen, and only the classification head was trained. This setup achieved a test accuracy of 88.05%, F1-score of 0.8757, and Recall for Malignant class 0.89.
2. **Partially fine-tuned base model (last 30 layers unfrozen)**
   > In this configuration, the final 30 layers of MobileNetV2 were unfrozen to allow deeper feature adaptation. This improved performance slightly, achieving the highest test accuracy of 88.91%, F1-score of 0.8848, and Recall for Malignant class 0.90. Fine-tuning enabled the model to adjust more complex features to the specific characteristics of skin lesion images, helping it generalises better to unseen data.
3. **Fine-tuned base with data augmentation**
   > The third approach incorporated the same fine-tuning strategy but also applied real-time data augmentation (random flips, rotations, zoom, and translations) to increase the diversity and robustness of the training data. This setup reached a test accuracy of 88.05%, F1-score of 0.8680, and Recall for Malignant class 0.83. The model with data augmentation showed steady progress during training, and although we increased the number of epochs, it did not outperform the fine-tuned model—likely due to the added complexity and regularization introduced by augmentation, which can slow convergence.

Among the three configurations, the **partially fine-tuned model (last 30 layers unfrozen)** achieved the best overall performance. It slightly outperformed the others in accuracy, F1-score, and especially **recall for the Malignant class**, which is crucial in medical diagnosis to minimize false negatives.  



<p align="center">  
  <img src="training.png" width="500"/>  
</p>  

*Figure 1. MobileNetV2 Training and Validation Loss*  

## Contributions  
This project contributes by:  
- Applying and comparing three distinct MobileNetV2 configurations on a curated melanoma dataset.  
- Emphasizing **recall** as a critical metric for clinical tasks.  
- Analyzing the impact of **layer unfreezing, data augmentation, and training duration**.  
- Providing practical insights for deploying lightweight CNNs in sensitive domains like dermatology.  

