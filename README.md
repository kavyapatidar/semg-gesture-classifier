# sEMG Gesture Classification – Synapse NeuroTech Challenge

This repository contains a complete machine learning pipeline for **surface Electromyography (sEMG)–based hand gesture classification**, developed for **Synapse: The NeuroTech Challenge (PARSEC 6.0, IIT Dharwad)**.

The project focuses on **robust subject-independent gesture recognition**, addressing the inherent noise and variability present in biological signals.


## Project Overview

Surface EMG signals encode muscle activation patterns that precede limb movement. Decoding these signals enables applications such as prosthetic control and human–computer interaction. However, sEMG data is highly noisy and varies significantly across subjects and recording sessions.

This project implements and evaluates:
- Signal preprocessing and windowing
- Baseline machine learning models
- Feature-based classifiers
- End-to-end deep learning using CNNs
- Subject-independent (leave-one-subject-out) evaluation


## Dataset Description

- **Modality:** Surface Electromyography (sEMG)
- **Channels:** 8 forearm electrodes
- **Subjects:** 25
- **Gestures:** 5
- **Sessions:** 3 (recorded on different days)
- **Sampling Rate:** 512 Hz
- **Trial Duration:** 5 seconds per gesture
- **Trials per Gesture:** 7

### Gesture Classes
1. Open Hand  
2. Closed Hand  
3. Lateral Pinch  
4. Signalling Sign  
5. Rock Sign  


## Preprocessing Pipeline

1. **Band-pass Filtering (20–450 Hz)**  
   Removes motion artifacts and high-frequency noise while preserving motor unit action potentials.

2. **Windowing**  
   Signals are segmented into fixed-length overlapping windows to capture local temporal patterns and increase training samples.

3. **Normalization**  
   Channel-wise normalization reduces amplitude variability across subjects and recording sessions.


## Models Implemented

### Baseline Models
- Logistic Regression trained on raw sEMG windows
- Random Forest classifier using handcrafted EMG features

### Deep Learning Model
- 1D Convolutional Neural Network (CNN)
- End-to-end learning from raw sEMG signals
- Lightweight architecture suitable for real-time inference


## Evaluation Protocol

To ensure realistic performance estimation and avoid subject leakage, a **subject-independent (Leave-One-Subject-Out)** evaluation protocol is used. In each experiment, data from one subject is held out entirely for testing, while the model is trained on data from all remaining subjects. This setup reflects real-world deployment scenarios where the system must generalize to unseen users.


## Performance Summary

| Model                        | Evaluation Type            | Accuracy |
|------------------------------|----------------------------|----------|
| Logistic Regression          | Random Split               | ~22%     |
| Random Forest (EMG Features) | Random Split               | ~81%     |
| CNN (Raw sEMG)               | Random Split               | ~83%     |
| CNN (Raw sEMG)               | Subject-Independent (LOSO) | ~72–75%  |


## Installation and Setup 

Create Virtual Environment:
python -m venv venv


Activate Environment :
Windows

venv\Scripts\activate
Linux / macOS

source venv/bin/activate

Install Dependencies

pip install -r requirements.txt

Running the Project
Run Full Pipeline
python main.py


This will:

Build datasets

Train baseline, feature-based, and CNN models

Perform subject-independent evaluation

Print accuracy for each unseen subject

Note: Subject-independent CNN evaluation is computationally intensive and may take significant time on CPU.

**Key Contributions**

Robust subject-independent EMG gesture classification

Comparison of classical ML vs deep learning

Practical LOSO evaluation aligned with real-world use

Modular, extensible codebase

**Disclaimer**

This project was developed strictly for academic and research purposes as part of the Synapse NeuroTech Challenge. Dataset ownership and usage rights belong to the competition organizers.

Author

Kavya Patidar
B.Tech – Electronics & Advanced Communication
Maharaja Agrasen Institute of Technology

**Acknowledgements**

PARSEC 6.0, IIT Dharwad

Synapse: The NeuroTech Challenge

