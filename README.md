# Cardia4 - AI-Powered Healthcare Platform

## Overview
Cardia4 is a comprehensive AI-powered healthcare platform that combines multiple machine learning models and deep learning techniques to provide accurate cardiovascular risk assessments and patient care recommendations. The platform integrates various medical imaging analyses including X-ray, MRI, and ECG, along with traditional health metrics to deliver holistic health insights.

## Features

### 1. Risk Assessment System
- Machine Learning-based risk prediction
- Medical imaging analysis:
  - X-ray analysis using ResNet50
  - Cardiac MRI processing
  - ECG interpretation using deep learning
- Comprehensive risk scoring system with weighted analysis

### 2. Patient Care Classification
- InCare/OutCare patient classification
- Automated decision support system
- Confidence-based recommendations
- Batch processing capability for multiple patients

### 3. Interactive Dashboards
- Real-time risk assessment visualization
- Patient care statistics
- Session-based analysis tracking
- Detailed patient records and history

### 4. AI Health Assistant
- Built-in chatbot powered by Google's Gemini AI
- Evidence-based health information
- Real-time medical guidance
- Professional medical disclaimers

## Technical Stack

### Backend
- Flask (Python web framework)
- SQLite database
- TensorFlow (Deep Learning)
- scikit-learn (Machine Learning)

### Frontend
- HTML/CSS/JavaScript
- Interactive data visualization
- Responsive design
- Real-time updates

### AI/ML Models
- ResNet50 for image processing
- Custom deep learning models for ECG
- Random Forest for patient classification
- Gemini AI for chatbot interactions

## Installation

1. Clone the repository
2. Create a Python virtual environment:
```python
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
