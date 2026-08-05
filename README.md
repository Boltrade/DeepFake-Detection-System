# Deepfake Detection from Images

A simple deep learning project to detect deepfakes using Convolutional Neural Networks (CNN). This model takes an image as input and predicts whether it's **Real** or **Fake**.

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/Boltrade/DeepFake-Detection-System
.git
cd deepfake-detection
```
### 2. Set Up Environment
Install required packages: **pip install -r requirements.txt**

### 3. Train the Model by downloading real and fake images samples(you can get from kaggle.com)
To train the model: **python train.py**

### 4. To predict the model
To predict whether an image is real or fake: **python predict.py path_to_image.jpg**

### Output will be:
Prediction: Real / Fake (with accuracy above 90%).
