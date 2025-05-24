# Rice-Leaf-Disease-Detection-
Rice Leaf Disease Detection is a machine learning project that identifies and classifies diseases in rice plant leaves using image analysis. It helps farmers and agronomists detect issues early, ensuring timely intervention. The system improves crop health monitoring and supports sustainable agriculture.
🌾 Rice Leaf Disease Detection
A computer vision project that detects and classifies common rice leaf diseases using deep learning techniques. The goal is to assist farmers and agricultural experts in identifying diseases early, minimizing crop loss, and improving yield quality through timely intervention.

📌 Project Overview
Rice is a staple food crop for more than half the world’s population, and diseases affecting rice plants can severely impact food security. This project utilizes Convolutional Neural Networks (CNNs) to automatically detect and classify rice leaf diseases from image data.

Objectives:
Detect and classify multiple types of rice leaf diseases from images.

Develop a robust deep learning model capable of generalizing on real-world data.

Support early and accurate disease diagnosis to assist agricultural decision-making.

🍃 Types of Rice Leaf Diseases
This project classifies images into the following categories:

Bacterial Leaf Blight

Brown Spot

Leaf Smut

Healthy Leaf (optional, based on dataset)

🗂️ Dataset
Source: [Kaggle / PlantVillage / Custom Collected Dataset]

Format: RGB images of rice leaves

Size: e.g., 120–2,000+ labeled images

Classes: 3–4 disease types (with or without healthy leaf class)

Note: All images were resized and augmented to improve model generalization.

🛠️ Technologies Used
Language: Python 3.x

Libraries:

TensorFlow, Keras, OpenCV, NumPy, Pandas

Matplotlib, Seaborn for visualization

scikit-learn for model evaluation

Model Architectures:

Custom CNN

Transfer Learning with MobileNetV2 and EfficientNetB0

🏗️ Project Structure
bash
Copy
Edit
Rice-Leaf-Disease-Detection/
│
├── data/                    # Image dataset (train/test/validation)
├── notebooks/               # Jupyter notebooks for analysis
├── models/                  # Trained models and logs
├── app/                     # Streamlit/Flask web app (optional)
├── requirements.txt         # Python package requirements
├── README.md                # Project overview and instructions
└── main.py                  # Training/testing entry point
🔍 Methodology
Data Preprocessing:

Resizing, normalization, image augmentation

Model Training:

CNN from scratch and pretrained models (MobileNetV2, EfficientNetB0)

Loss: categorical_crossentropy

Optimizer: Adam

Evaluation:

Accuracy, Precision, Recall, F1-Score

Confusion Matrix, ROC curves

📊 Sample Results
Model	Accuracy	Precision	Recall	F1 Score
Custom CNN	83%	0.82	0.83	0.82
MobileNetV2	88%	0.87	0.88	0.87
EfficientNetB0	91%	0.90	0.91	0.90

Results may vary depending on dataset size and training parameters.

🚀 How to Run
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/yourusername/Rice-Leaf-Disease-Detection.git
cd Rice-Leaf-Disease-Detection
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Train the Model
bash
Copy
Edit
python main.py
4. Run the Web App (Optional)
bash
Copy
Edit
streamlit run app/app.py
🧪 Visualizations
Confusion Matrix

Sample Predictions

Training & Validation Accuracy/Loss Curves

<p align="center"> <img src="images/sample_predictions.png" width="600" alt="Sample Predictions"> </p>
💡 Future Improvements
Deploy model to edge devices (e.g., mobile phones or Raspberry Pi).

Include a wider range of rice diseases and real-time video input.

Add multilingual support for farmers in rural areas.

Integrate weather and location-based disease forecasting.

🤝 Contributing
Pull requests are welcome! If you’d like to contribute to this project (e.g., better model, new UI, larger dataset), please fork the repo and submit a PR.

⚖️ License
This project is licensed under the MIT License.

🙏 Acknowledgements
Kaggle

PlantVillage Dataset

TensorFlow

Streamlit
