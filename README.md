# Deepfake Image Classifier  
A deep learning project designed to classify images as **real** or **fake** using state-of-the-art CNN architectures implemented in the **PyTorch** framework.  

## Overview  
This repository provides an implementation of a deepfake image classifier built and tested on a subset of the [Deepfake and Real Images Dataset](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images/data). The goal of this project is to identify the best-performing model for detecting deepfake images through experimentation with multiple pre-trained Convolutional Neural Networks (CNNs).  

## Dataset  
The dataset used for this project contains **10,000 images** split evenly into:  
- **5,000 fake images**  
- **5,000 real images**
- Fake Image:
  
![image](https://github.com/user-attachments/assets/c218c681-65e5-41b8-bed2-eabe99dcda82)

- Real Image:
  
![image](https://github.com/user-attachments/assets/dbd4d579-04ba-4fd6-ad81-61c6c6b14046)

These images were extracted from the larger dataset available [here](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images/data).  

### Dataset Preprocessing  
- Images were resized and normalized to fit the input dimensions of the models.  
- Data augmentation techniques were applied to improve model generalization.  
- The dataset was split into training, validation, and testing sets for evaluation.  

## Models Tested  
Five pre-trained CNN architectures were evaluated in this project:  
1. **VGG-16**  
2. **ResNet-50**  
3. **DenseNet-121**  
4. **MobileNet**  
5. **Inception**  

### Experimentation  
1. **Custom Classifier Layer**  
   - For each model, the final classifier layer was replaced with a fully connected layer for **binary classification** (real or fake).  
2. **Fine-Tuning**  
   - All models were fine-tuned on the dataset to further adapt to the task of deepfake classification.
3. **Learning Rate Optimization with Optuna**
   - The optimal learning rate for each model was determined using Optuna, a hyperparameter optimization framework.
   - Optuna was used to perform efficient search across a range of learning rates, selecting the one that maximized performance on the validation set.

## Results  
After evaluating the performance of each model using 3-fold cross-validation, the best-performing model was selected based on accuracy, precision, recall, and F1-score. 

## Framework  
This project is implemented using **PyTorch**, a powerful and flexible deep learning framework that facilitates:  
- Efficient model training and evaluation.  
- Easy integration with pre-trained models from `torchvision`.  
- Comprehensive tools for fine-tuning and transfer learning.  

### Results Visualization  
The training process generates plots for:  
- Accuracy and loss curves.  
- Confusion matrix.  
- ROC and precision-recall curves.  

## Contributing  
Contributions are welcome! Feel free to fork the repository, open issues, or submit pull requests to improve this project.  

## Acknowledgements  
- [Deepfake and Real Images Dataset](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images/data) for providing the data.  
- Pre-trained CNN models available through PyTorch's `torchvision`.
## Contact
If you have any questions, suggestions, or feedback about this project, feel free to reach out!

| **Name**              | **Email**                    | **LinkedIn**                                                      |
|------------------------|------------------------------|-------------------------------------------------------------------|
| Adham Hany Ahmed      | elkomyadham56@gmail.com      | [Adham Elkomy](https://www.linkedin.com/in/adham-hany-b18469277) |
| Ziad Ahmed Eliwa      | ziadeliwa93@gmail.com        | [Ziad Eliwa](https://www.linkedin.com/in/ziad-eliwa-b48754240/)  |
| Adham Adel Mohammed   | adhamadelmoh@gmail.com       | [Adham Adel](https://www.linkedin.com/in/ziad-eliwa-b48754240/)  |


