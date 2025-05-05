# Epileptic_Seizure_Analysis
🧠 Epileptic Seizure Analysis Using Scalp EEG Signals with Deep Learning
📌 Project Overview
This research explores epileptic seizure detection and emotion classification in individuals—particularly those with depression—using scalp EEG signals. The goal is to improve early detection and classification using advanced machine learning (KNN) and deep learning (LSTM) models.

👥 Authors
Ms. B. Madhavi Devi, Assistant Professor

Rayedi Srilekha, BTech IV Year, Institute of Aeronautical Engineering

Gattineni Harshitha, BTech IV Year

Burri Sushma, BTech IV Year

🎯 Objectives
Detect epileptic seizures using EEG signal analysis.

Improve emotion classification accuracy in depressed patients using KNN and LSTM.

Propose a preprocessing and classification pipeline using real-time EEG recordings.

🧪 Methodology
1. Data Collection
EEG data was collected using Muse Headband from 2 individuals under positive, neutral, and negative conditions.

2. Preprocessing
Null values removed and categorical data encoded.

Bandpass filtering and spatial filtering applied to improve signal-to-noise ratio.

Dataset split into training and testing partitions.

3. Feature Selection
Hand-crafted features like mean, standard deviation, and skewness were extracted from EEG signals.

4. Classification Models
K-Nearest Neighbor (KNN): Classifies based on similarity to nearest neighbors.

Long Short-Term Memory (LSTM): Captures temporal patterns in EEG signal sequences.

📊 Results
KNN Accuracy: 94%

LSTM Accuracy: 95%

Metrics used: Accuracy, Precision, Recall, F1-Score

Confusion matrices and bar graphs used to visualize performance.

🔧 Technologies Used
Python

TensorFlow / Keras

Scikit-learn

NumPy / Pandas

Muse EEG Headband

📈 Visuals
EEG electrode placements

Classification outputs from KNN and LSTM

Confusion matrix and emotion bar graph

📚 References
The project is supported by a wide range of studies in EEG-based depression analysis, seizure prediction, and deep learning techniques (full reference list available in the paper).

🏁 Conclusion
This work shows promising results in detecting seizures and classifying emotional states using EEG signals and machine learning. The integration of spatial features and LSTM networks greatly improves detection accuracy, paving the way for real-time clinical applications.

