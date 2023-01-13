# Skin Disease Prediction through Image Analysis of Diverse Skin Tones
---
# Overview 
Dermatologists diagnose a variety of skin diseases, most commonly skin cancers and other inflammatory conditions. AI predictive modeling is a promising dermatology tool to aid in early detection of skin cancer and other skin diseases. Historically, most dermatology models have not been trained on images of diverse skin tones, leading to potential biases in algorithm performance. This project aimed to utilize Convolutional Neural Networks on a Stanford University “Diverse Dermatology Images” Dataset, to predict malignant skin diseases, inclusive of all skin tones. 

The data used from this project can be found from the following source: [Stanford AIMI Shared Datasets](https://stanfordaimi.azurewebsites.net/datasets/35866158-8196-48d8-87bf-50dca81df965). All image rights to the Diverse Dermatology Images Dataset (DDI) belong to Stanford University School of Medicine. If you would like to replicate this project, you will need to register as an individual user and comply with all terms of their Research Use Agreement. 

---
# Data Dictionary 
|Feature|Type|Dataset|Description|
|---|---|---|---|
|DDI_file|object|DDI|Image Path|
|skin_tone|integer|DDI|Skin tone present, where 12 indicates tone falls into Fitzpatrick range I-II, 34 indicates tone falls into Fitzpatrick range III-IV and 56 indicates tone falls into Fitzpatrick range V-VI|
|malignant|integer|DDI|Numerical binary representation of each class, where 0 is "benign" and 1 is "malignant"|
|disease|object|DDI|Type of disease present on image|

---
# Conclusions and Recommendations 
Predictive modeling can be a powerful tool to triaging or assessing risk for patients to see a dermatologist. While it is a great tool, it is important for models to be trained on images of diverse skin tones. There is a huge ethical component to AI in medicine. If a individual’s skin tone is not represented when training the model, there is inherent bias in how the model performs, and in the case of triaging patients, a greater disparity in access to medical care. It is extremely important that medical institutions implement models that improve medical equality.

In general models need to be kept current and doctors should still monitor the results. Medical predictive modeling is not meant to take the place of seeing a trained professional, but can be a powerful tool to aid in the quick diagnosis and risk assessment of various diseases.

This project highlights 7 Convolutional Neural Network (CNN) models with different hyperparameters tuned. The best model was model 6, accounting for correctly classifying 71% of malignant cases and 88% of benign cases (recall). In this model I used the ResNet50 pre trained keras neural network. I froze all base layers and added a dense output layer with 2 nodes and softmax activation. I used a SGD optimizer with a learning rate of 0.0001 and a momentum of 0.9. The loss function used was binary cross entropy and I monitored accuracy and recall, with the intent of optimizing recall for a medical classification problem. Data going into this model was pre-processed with a keras pre-processing ResNet50 function, and rescaled to 1/255. Train batches were augmented to mitigate overfitting.

The results of this model are summarized below:

- Averarge Accuracy: 83%
- Average Recall: 79%
- Average Precision: 78%
- Malignant Recall: 71%
- Benign Recall: 88%

There were other models that performed better on paper, but ultimately model 6 was selected because recall was being optimized, in an effort to minimize false negatives for a medical classification problem. The intention is to ensure no one gets turned away that needs to be treated.

Ultimately, I think this dataset needs more data to train with and scores would be higher. Additionally there is class imbalance between malignant and benign classes leading to higher benign recall scores. I will continue to improve this project by addressing the class imbalance through thresholding. Additionally, I want to continue to hyperparameter tune my best model and also look into zooming in to the image to only include the skin condition.

---
# Streamlit Application

https://user-images.githubusercontent.com/114830638/212193676-371726fa-8e67-4f0a-9b63-6c82389d5af3.mov

