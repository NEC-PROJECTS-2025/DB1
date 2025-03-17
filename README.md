# Melanoma Skin Cancer Detection

## Project Overview
This project focuses on the detection of melanoma skin cancer using a hybrid deep learning and machine learning approach. The model utilizes deep learning architectures such as VGG16, VGG19, ResNet50, Capsule Networks (CapsNet), and Vision Transformers (ViT) to extract image features. These extracted features are then classified using ensemble machine learning classifiers, including Support Vector Classifier (SVC), XGBoost, Random Forest, K-Nearest Neighbors (KNN), and Logistic Regression, leveraging majority voting to enhance classification accuracy.

## Key Features
- **Deep Learning Models:** VGG16, VGG19, ResNet50, Capsule Networks, Vision Transformers (ViT)
- **Machine Learning Classifiers:** SVC, XGBoost, Random Forest, KNN, Logistic Regression
- **Ensemble Approach:** Majority voting to enhance classification performance
- **Dataset:** Kaggle's Melanoma Skin Cancer Dataset (10,000 high-resolution dermoscopic images)
- **Performance:** Achieved up to 92.4% accuracy using the ViT model

## Dataset
- **Source:** Kaggle - "Melanoma Skin Cancer Dataset of 10,000 Images"
- **Images:** 9,605 training images (5,773 benign, 3,832 malignant), 1,000 test images (500 benign, 500 malignant)
- **Preprocessing:** Image resizing, colorspace conversion, normalization, label encoding, train-test split

## Methodology
### Preprocessing Steps
1. **Resizing:** All images are resized to 128x128 pixels.
2. **Color Space Conversion:** Converted images from BGR to RGB format.
3. **Shuffling:** Randomized the image order to prevent bias.
4. **Normalization:** Scaled pixel values between 0 and 1.
5. **Label Encoding:** Converted categorical labels (benign/malignant) into numeric form.
6. **Train-Test Split:** 80% training, 20% testing.

### Model Implementation
1. **Feature Extraction:**
   - VGG16, VGG19, ResNet50, CapsNet, ViT are used for feature extraction.
2. **Classification:**
   - Machine learning models (SVC, XGBoost, Random Forest, KNN, Logistic Regression) classify extracted features.
3. **Ensemble Model:**
   - Majority voting is used to combine results from classifiers to improve accuracy.

## Performance Evaluation
- **Best Performing Model:** ViT with 92.4% accuracy
- **Ensemble Model Accuracy:** 92.3%
- **Evaluation Metrics:** Precision, Recall, F1 Score, Accuracy
- **Confusion Matrix Analysis:** ViT showed the lowest false positives and false negatives.

## Requirements
- Python 3.x
- TensorFlow/Keras
- OpenCV
- NumPy
- Scikit-learn
- Matplotlib
- Pandas
- XGBoost

## Installation & Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repository/melanoma-detection.git
   cd melanoma-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the training script:
   ```bash
   python train.py
   ```
4. Evaluate the model:
   ```bash
   python evaluate.py
   ```
5. Predict on new images:
   ```bash
   python predict.py --image-path test_image.jpg
   ```

## Future Enhancements
- Implement real-time melanoma detection using mobile applications.
- Fine-tune the ViT model for even higher accuracy.
- Integrate additional preprocessing techniques for better feature extraction.
- Deploy the model as a web-based application for easier accessibility.

## Authors
- Dr. S.V.N. Srinivasu (Professor, Dept of CSE, Narasaraopeta Engineering College)
- Bellamkonda Nanda Krishna
- Kurra Venkatesh
- Kalva Adi Babu
- Ch Rajani (Asst. Prof, Dept of CSE, Narasaraopeta Engineering College)

## Acknowledgment
Special thanks to the Kaggle community for providing the melanoma dataset and to all researchers who contributed to advancing AI in medical imaging.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


