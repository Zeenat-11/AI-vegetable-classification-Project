# GreenClassify: Deep Learning-Based Vegetable Image Classifier

GreenClassify is an AI-powered web application designed to classify different types of vegetables using a Convolutional Neural Network (CNN). It leverages deep learning and image processing to assist users—especially farmers, suppliers, and researchers—in identifying vegetables from images quickly and accurately.

---

##  Features

*  Classifies 12 types of vegetables:

  * Bean, Bitter Gourd, Bottle Gourd, Brinjal, Broccoli, Cabbage, Capsicum, Carrot, Cauliflower, Cucumber, Papaya, Potato
* Upload & Predict: Users can upload vegetable images and get real-time predictions
* Model Accuracy: Achieves high classification accuracy with data augmentation
* Flask-based Web App with a clean Bootstrap UI

---

##  Model Overview

GreenClassify uses a custom CNN architecture built using TensorFlow/Keras. It includes:

* 3 Convolutional layers with ReLU activation
* MaxPooling layers
* Dropout for regularization
* Fully connected Dense layers for classification

---

##  Dataset

* Custom dataset with images categorized into 12 vegetable classes.
* Augmented using `ImageDataGenerator` for robustness.
* Directory structure:

```
dataset/
│
├── train/
│   ├── Bean/
│   ├── Bitter_Gourd/
│   └── ...
│
└── test/
    ├── Bean/
    ├── Bitter_Gourd/
    └── ...
```

---

##  Installation

```bash
git clone https://github.com/<your-username>/greenclassify.git
cd greenclassify
pip install -r requirements.txt
```

---

##  How to Run

1. **Model Training (Optional)**

```python
python train_model.py
```

2. **Start Flask Web App**

```bash
python app.py
```

3. **Open in Browser**

Go to: `http://127.0.0.1:5000`

---

##  Sample Predictions

| Input Image                             | Prediction |
| --------------------------------------- | ---------- |
| ![Carrot](images/sample_carrot.jpg)     | Carrot     |
| ![Broccoli](images/sample_broccoli.jpg) | Broccoli   |

---

##  Project Structure

```
greenclassify/
│
├── static/
│   └── styles.css
│
├── templates/
│   └── index.html
│
├── dataset/
│   └── train/, test/
│
├── model/
│   └── vegetable_classifier.h5
│
├── app.py
├── train_model.py
└── README.md
```

---

## Class Mapping

```python
{
  0: 'Bean',
  1: 'Bitter_Gourd',
  2: 'Bottle_Gourd',
  3: 'Brinjal',
  4: 'Broccoli',
  5: 'Cabbage',
  6: 'Capsicum',
  7: 'Carrot',
  8: 'Cauliflower',
  9: 'Cucumber',
  10: 'Papaya',
  11: 'Potato'
}
```




