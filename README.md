# 🕵️‍♂️ Deepfake Detection from Images

A simple deep learning project to detect deepfakes using Convolutional Neural Networks (CNN). This model takes an image as input and predicts whether it's **Real** or **Fake**.

---

## 📁 Project Structure
deepfake_detection/ │ ├── data/ # Dataset folder │ ├── real/ # Real images │ └── fake/ # Deepfake images │ ├── model/ # Folder to save trained model │ └── deepfake_cnn.pth │ ├── deepfake_model.py # CNN model definition ├── train.py # Training script ├── predict.py # Image prediction script ├── utils.py # Preprocessing utilities ├── requirements.txt # Required Python packages └── README.md # Project documentation

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/Boltrade/deepfake-detection.git
cd deepfake-detection
```
### 2. Set Up Environment
Install required packages: pip install -r requirements.txt

### 3. Prepare Dataset
Create the following structure inside the data/ folder:

##### data/
├── real/
│   ├── real1.jpg
│   └── ...
└── fake/
    ├── fake1.jpg
    └── ...

Make sure you have a good number of real and fake images.(You can get it from Kaggle).

### 4. Train the Model
To predict whether an image is real or fake: python predict.py path_to_image.jpg
##### Output will be:
Prediction: Real
##### or
Prediction: Fake
