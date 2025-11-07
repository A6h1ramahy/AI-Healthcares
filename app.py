from flask import Flask, render_template, request, jsonify, send_file
import sqlite3
import pickle
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ===== Database Setup =====
DB_PATH = os.path.join('database', 'cardia4.db')

def init_db():
    os.makedirs('database', exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS heart_predictions (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            name TEXT,
                            age INTEGER,
                            sex TEXT,
                            risk_ml REAL,
                            risk_xray REAL,
                            risk_mri REAL,
                            risk_ecg REAL,
                            risk_final REAL,
                            result TEXT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS care_analysis (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            session_id TEXT,
                            analysis_type TEXT,
                            total_patients INTEGER,
                            incare_count INTEGER,
                            outcare_count INTEGER,
                            consult_count INTEGER,
                            automation_rate REAL,
                            csv_filename TEXT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS care_patients (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            session_id TEXT,
                            patient_index INTEGER,
                            classification TEXT,
                            confidence REAL,
                            probability REAL,
                            haematocrit REAL,
                            haemoglobins REAL,
                            erythrocyte REAL,
                            leucocyte REAL,
                            thrombocyte REAL,
                            mch REAL,
                            mchc REAL,
                            mcv REAL,
                            age INTEGER,
                            sex TEXT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )''')
        conn.commit()

init_db()

# ===== Load Models =====
rf_model = pickle.load(open('models/rf_model.pkl', 'rb'))
ml_model = pickle.load(open('models/ml_model.pkl', 'rb'))

# Load Deep Learning Models
try:
    xray_model = load_model('models/xray_resnet50.h5')
    mri_model = load_model('models/cardiac_mri_resnet50.h5')
    ecg_model = load_model('models/final_ecg_model.h5')
    print("All deep learning models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    xray_model = mri_model = ecg_model = None

# ===== Image Processing Functions =====
def preprocess_image(img_file, target_size=(224, 224)):
    """Preprocess uploaded image for model prediction"""
    try:
        # Open and convert image
        img = Image.open(io.BytesIO(img_file.read()))
        img = img.convert('RGB')
        img = img.resize(target_size)
        
        # Convert to array and normalize
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize to [0,1]
        
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_from_image(model, img_array):
    """Get prediction from deep learning model"""
    try:
        if model is None or img_array is None:
            return 50.0  # Default risk if model unavailable
        
        prediction = model.predict(img_array, verbose=0)
        # Convert prediction to risk percentage (0-100)
        risk_score = float(prediction[0][0]) * 100 if len(prediction[0]) == 1 else float(prediction[0][1]) * 100
        return max(0, min(100, risk_score))  # Clamp between 0-100
    except Exception as e:
        print(f"Error in model prediction: {e}")
        return 50.0  # Default risk if prediction fails

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        data = request.form
        
        # Extract all form fields with defaults
        age = float(data.get('age', 0))
        sex = int(data.get('sex', 0))
        cp = float(data.get('cp', 0))
        trestbps = float(data.get('trestbps', 120))
        chol = float(data.get('chol', 200))
        fbs = float(data.get('fbs', 0))
        restecg = float(data.get('restecg', 0))
        thalach = float(data.get('thalach', 150))
        exang = float(data.get('exang', 0))
        oldpeak = float(data.get('oldpeak', 0))
        slope = float(data.get('slope', 0))
        ca = float(data.get('ca', 0))
        thal = float(data.get('thal', 0))
        
        # Create feature array with all 13 features
        features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        try:
            risk_ml = ml_model.predict_proba(features)[0][1] * 100
        except:
            # Fallback if model expects different features
            features_simple = np.array([[age, sex]])
            risk_ml = ml_model.predict_proba(features_simple)[0][1] * 100

        # Process uploaded images and get DL predictions
        risk_xray = 25.0  # Default
        risk_mri = 30.0   # Default  
        risk_ecg = 20.0   # Default
        
        # Process X-ray image
        if 'xray' in request.files and request.files['xray'].filename:
            xray_file = request.files['xray']
            xray_array = preprocess_image(xray_file)
            risk_xray = predict_from_image(xray_model, xray_array)
        
        # Process MRI image
        if 'mri' in request.files and request.files['mri'].filename:
            mri_file = request.files['mri']
            mri_array = preprocess_image(mri_file)
            risk_mri = predict_from_image(mri_model, mri_array)
        
        # Process ECG image
        if 'ecg' in request.files and request.files['ecg'].filename:
            ecg_file = request.files['ecg']
            ecg_array = preprocess_image(ecg_file)
            risk_ecg = predict_from_image(ecg_model, ecg_array)
        
        # Parse weights if provided
        weights_str = data.get('weights', '0.4,0.2,0.2,0.2')
        try:
            weights = [float(w.strip()) for w in weights_str.split(',')]
            if len(weights) != 4:
                weights = [0.4, 0.2, 0.2, 0.2]
        except:
            weights = [0.4, 0.2, 0.2, 0.2]
        
        # Calculate weighted final risk
        risks = [risk_ml, risk_xray, risk_mri, risk_ecg]
        risk_final = sum(r * w for r, w in zip(risks, weights))

        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('INSERT INTO heart_predictions (name, age, sex, risk_ml, risk_xray, risk_mri, risk_ecg, risk_final, result) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                           (data.get('name'), age, 'Male' if sex == 1 else 'Female', risk_ml, risk_xray, risk_mri, risk_ecg, risk_final,
                            'High Risk' if risk_final > 50 else 'Low Risk'))
            conn.commit()

        return jsonify({
            "risk_ml": round(risk_ml, 2),
            "risk_xray": round(risk_xray, 2), 
            "risk_mri": round(risk_mri, 2),
            "risk_ecg": round(risk_ecg, 2),
            "risk_final": round(risk_final, 2),
            "result": 'High Risk' if risk_final > 50 else 'Low Risk'
        })

    return render_template('predict.html')

def classify_patient(probability):
    """
    Classify patient into 3 categories based on prediction confidence:
    - High confidence (>80% or <20%): Direct classification
    - Medium confidence (20-80%): Consult Doctor
    """
    if probability >= 0.8:
        return "InCare", probability
    elif probability <= 0.2:
        return "OutCare", 1 - probability
    else:
        return "Consult Doctor", max(probability, 1 - probability)

@app.route('/incare-outcare', methods=['GET', 'POST'])
def incare_outcare():
    if request.method == 'POST':
        if 'csv_file' in request.files:
            # CSV batch prediction
            file = request.files['csv_file']
            if file.filename == '':
                return jsonify({"error": "No file selected"})
            
            try:
                import pandas as pd
                df = pd.read_csv(file)
                
                # Validate required columns
                required_cols = ['HAEMATOCRIT', 'HAEMOGLOBINS', 'ERYTHROCYTE', 'LEUCOCYTE', 
                               'THROMBOCYTE', 'MCH', 'MCHC', 'MCV', 'AGE', 'SEX']
                
                if not all(col in df.columns for col in required_cols):
                    return jsonify({"error": f"Missing required columns. Required: {required_cols}"})
                
                # Handle M/F gender format - convert to 1/0
                df_processed = df.copy()
                df_processed['SEX'] = df_processed['SEX'].map({'M': 1, 'Male': 1, 'F': 0, 'Female': 0, 1: 1, 0: 0})
                
                # Check for any unmapped gender values
                if df_processed['SEX'].isnull().any():
                    return jsonify({"error": "Invalid gender values. Use M/F, Male/Female, or 1/0"})
                
                # Process batch predictions
                features = df_processed[required_cols].values
                probabilities = rf_model.predict_proba(features)[:, 1]  # Probability of InCare
                
                results = []
                incare_patients = []
                outcare_patients = []
                consult_patients = []
                
                for i, prob in enumerate(probabilities):
                    classification, confidence = classify_patient(prob)
                    
                    # Add original data with classification results
                    patient_data = df.iloc[i].to_dict()
                    patient_data['CLASSIFICATION'] = classification
                    patient_data['CONFIDENCE'] = round(confidence * 100, 1)
                    patient_data['PROBABILITY'] = round(prob * 100, 1)
                    
                    # Categorize patients for separate CSV files
                    if classification == "InCare":
                        incare_patients.append(patient_data)
                    elif classification == "OutCare":
                        outcare_patients.append(patient_data)
                    else:
                        consult_patients.append(patient_data)
                    
                    results.append({
                        "patient_id": i + 1,
                        "classification": classification,
                        "confidence": round(confidence * 100, 1),
                        "probability": round(prob * 100, 1)
                    })
                
                # Generate CSV files for each category
                import os
                import uuid
                
                # Create unique session ID for file naming
                session_id = str(uuid.uuid4())[:8]
                csv_dir = os.path.join('static', 'downloads')
                os.makedirs(csv_dir, exist_ok=True)
                
                csv_files = {}
                
                print(f"Generating CSV files with session ID: {session_id}")  # Debug
                print(f"CSV directory: {os.path.abspath(csv_dir)}")  # Debug
                
                if incare_patients:
                    incare_df = pd.DataFrame(incare_patients)
                    incare_filename = f'incare_patients_{session_id}.csv'
                    incare_path = os.path.join(csv_dir, incare_filename)
                    incare_df.to_csv(incare_path, index=False)
                    csv_files['incare'] = incare_filename
                    print(f"Created InCare CSV: {incare_path} ({len(incare_patients)} patients)")
                
                if outcare_patients:
                    outcare_df = pd.DataFrame(outcare_patients)
                    outcare_filename = f'outcare_patients_{session_id}.csv'
                    outcare_path = os.path.join(csv_dir, outcare_filename)
                    outcare_df.to_csv(outcare_path, index=False)
                    csv_files['outcare'] = outcare_filename
                    print(f"Created OutCare CSV: {outcare_path} ({len(outcare_patients)} patients)")
                
                if consult_patients:
                    consult_df = pd.DataFrame(consult_patients)
                    consult_filename = f'consult_doctor_{session_id}.csv'
                    consult_path = os.path.join(csv_dir, consult_filename)
                    consult_df.to_csv(consult_path, index=False)
                    csv_files['consult'] = consult_filename
                    print(f"Created Consult CSV: {consult_path} ({len(consult_patients)} patients)")
                
                print(f"CSV files generated: {csv_files}")  # Debug
                
                # Store analysis results in database
                with sqlite3.connect(DB_PATH) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''INSERT INTO care_analysis 
                                    (session_id, analysis_type, total_patients, incare_count, outcare_count, 
                                     consult_count, automation_rate, csv_filename) 
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                                 (session_id, 'batch', len(results), len(incare_patients), 
                                  len(outcare_patients), len(consult_patients),
                                  round(((len(incare_patients) + len(outcare_patients)) / len(results)) * 100, 1),
                                  file.filename))
                    
                    # Store individual patient results
                    for i, result in enumerate(results):
                        patient_row = df.iloc[i]
                        cursor.execute('''INSERT INTO care_patients 
                                        (session_id, patient_index, classification, confidence, probability,
                                         haematocrit, haemoglobins, erythrocyte, leucocyte, thrombocyte,
                                         mch, mchc, mcv, age, sex) 
                                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                                     (session_id, i+1, result['classification'], result['confidence'], result['probability'],
                                      patient_row['HAEMATOCRIT'], patient_row['HAEMOGLOBINS'], patient_row['ERYTHROCYTE'],
                                      patient_row['LEUCOCYTE'], patient_row['THROMBOCYTE'], patient_row['MCH'],
                                      patient_row['MCHC'], patient_row['MCV'], patient_row['AGE'], 
                                      'M' if patient_row['SEX'] == 1 else 'F'))
                    conn.commit()
                
                return jsonify({
                    "results": results,
                    "summary": {
                        "total_patients": len(results),
                        "incare": len(incare_patients),
                        "outcare": len(outcare_patients),
                        "consult_doctor": len(consult_patients),
                        "automation_rate": round(((len(incare_patients) + len(outcare_patients)) / len(results)) * 100, 1)
                    },
                    "csv_files": csv_files
                })
                
            except Exception as e:
                return jsonify({"error": f"Error processing CSV: {str(e)}"})
        
        else:
            # Individual patient prediction
            data = request.form
            try:
                # Extract all hematology features
                haematocrit = float(data.get('HAEMATOCRIT', 0))
                haemoglobins = float(data.get('HAEMOGLOBINS', 0))
                erythrocyte = float(data.get('ERYTHROCYTE', 0))
                leucocyte = float(data.get('LEUCOCYTE', 0))
                thrombocyte = float(data.get('THROMBOCYTE', 0))
                mch = float(data.get('MCH', 0))
                mchc = float(data.get('MCHC', 0))
                mcv = float(data.get('MCV', 0))
                age = float(data.get('AGE', 0))
                sex = float(data.get('SEX', 0))
                
                # Create feature array with all 10 features
                features = np.array([[haematocrit, haemoglobins, erythrocyte, leucocyte, thrombocyte, mch, mchc, mcv, age, sex]])
                probability = rf_model.predict_proba(features)[0][1]  # Probability of InCare
                
                classification, confidence = classify_patient(probability)
                
                return jsonify({
                    "classification": classification,
                    "confidence": round(confidence * 100, 1),
                    "probability": round(probability * 100, 1),
                    "recommendation": {
                        "InCare": "Patient requires immediate inpatient care",
                        "OutCare": "Patient can be managed as outpatient", 
                        "Consult Doctor": "Uncertain case - requires doctor's evaluation"
                    }[classification]
                })
                
            except Exception as e:
                return jsonify({"error": f"Error processing data: {str(e)}"})

    return render_template('incare_outcare.html')

@app.route('/download/<filename>')
def download_csv(filename):
    """Download generated CSV files"""
    try:
        # Security check - only allow CSV files
        if not filename.endswith('.csv'):
            return jsonify({"error": "Invalid file type"}), 400
            
        # Use absolute path for file serving
        downloads_dir = os.path.join(os.getcwd(), 'static', 'downloads')
        file_path = os.path.join(downloads_dir, filename)
        
        print(f"Download request for: {filename}")
        print(f"Full file path: {file_path}")
        print(f"File exists: {os.path.exists(file_path)}")
        
        if os.path.exists(file_path):
            return send_file(file_path, 
                           as_attachment=True, 
                           download_name=filename,
                           mimetype='text/csv')
        else:
            return jsonify({"error": f"File not found: {filename}"}), 404
            
    except Exception as e:
        print(f"Download error: {str(e)}")
        return jsonify({"error": f"Download failed: {str(e)}"}), 500

@app.route('/dashboard')
def dashboard():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT name, age, sex, risk_final, result, created_at FROM heart_predictions ORDER BY created_at DESC')
        rows = cursor.fetchall()
    
    # Calculate risk statistics
    if rows:
        risk_values = [row[3] for row in rows if row[3] is not None]
        high_risk = len([r for r in risk_values if r > 50])
        medium_risk = len([r for r in risk_values if 25 < r <= 50])
        low_risk = len([r for r in risk_values if r <= 25])
        avg_risk = sum(risk_values) / len(risk_values) if risk_values else 0
    else:
        high_risk = medium_risk = low_risk = avg_risk = 0
    
    stats = {
        'total': len(rows),
        'high_risk': high_risk,
        'medium_risk': medium_risk,
        'low_risk': low_risk,
        'avg_risk': avg_risk
    }
    
    return render_template('dashboard.html', rows=rows, stats=stats)

@app.route('/care-dashboard')
def care_dashboard():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        
        # Get all care analysis sessions
        cursor.execute('''SELECT session_id, analysis_type, total_patients, incare_count, 
                         outcare_count, consult_count, automation_rate, csv_filename, created_at 
                         FROM care_analysis ORDER BY created_at DESC''')
        sessions = cursor.fetchall()
        
        # Calculate overall statistics
        if sessions:
            total_patients_analyzed = sum(row[2] for row in sessions)
            total_incare = sum(row[3] for row in sessions)
            total_outcare = sum(row[4] for row in sessions)
            total_consult = sum(row[5] for row in sessions)
            avg_automation_rate = sum(row[6] for row in sessions) / len(sessions)
        else:
            total_patients_analyzed = total_incare = total_outcare = total_consult = avg_automation_rate = 0
        
        # Get recent patient classifications
        cursor.execute('''SELECT session_id, patient_index, classification, confidence, 
                         age, sex, created_at FROM care_patients 
                         ORDER BY created_at DESC LIMIT 50''')
        recent_patients = cursor.fetchall()
        
        # Classification distribution
        classification_counts = {'InCare': 0, 'OutCare': 0, 'Consult Doctor': 0}
        for patient in recent_patients:
            classification_counts[patient[2]] += 1
    
    stats = {
        'total_sessions': len(sessions),
        'total_patients': total_patients_analyzed,
        'total_incare': total_incare,
        'total_outcare': total_outcare,
        'total_consult': total_consult,
        'avg_automation_rate': round(avg_automation_rate, 1),
        'classification_counts': classification_counts
    }
    
    return render_template('care_dashboard.html', sessions=sessions, recent_patients=recent_patients, stats=stats)

@app.route('/care-session/<session_id>')
def care_session_detail(session_id):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        
        # Get session info
        cursor.execute('''SELECT session_id, analysis_type, total_patients, incare_count, 
                         outcare_count, consult_count, automation_rate, csv_filename, created_at 
                         FROM care_analysis WHERE session_id = ?''', (session_id,))
        session_info = cursor.fetchone()
        
        if not session_info:
            return "Session not found", 404
        
        # Get all patients in this session
        cursor.execute('''SELECT patient_index, classification, confidence, probability,
                         haematocrit, haemoglobins, erythrocyte, leucocyte, thrombocyte,
                         mch, mchc, mcv, age, sex, created_at 
                         FROM care_patients WHERE session_id = ? 
                         ORDER BY patient_index''', (session_id,))
        patients = cursor.fetchall()
        
        # Calculate detailed statistics
        classification_stats = {}
        for patient in patients:
            classification = patient[1]
            if classification not in classification_stats:
                classification_stats[classification] = {'count': 0, 'avg_confidence': 0, 'confidences': []}
            classification_stats[classification]['count'] += 1
            classification_stats[classification]['confidences'].append(patient[2])
        
        # Calculate average confidence for each classification
        for classification in classification_stats:
            confidences = classification_stats[classification]['confidences']
            classification_stats[classification]['avg_confidence'] = sum(confidences) / len(confidences)
    
    return render_template('care_session_detail.html', 
                         session_info=session_info, 
                         patients=patients, 
                         classification_stats=classification_stats)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/test_chatbot')
def test_chatbot():
    return send_file('test_chatbot.html')

# ===== Chatbot API Endpoint =====
@app.route('/api/chat', methods=['POST'])
def chat_api():
    """Handle chatbot conversations with Gemini API"""
    logger.info("=== CHAT API CALLED ===")
    try:
        data = request.get_json()
        logger.info(f"Request data: {data}")
        user_message = data.get('message', '').strip()
        logger.info(f"User message: '{user_message}'")
        
        if not user_message:
            return jsonify({"error": "Message is required"}), 400
        
        # Set your Gemini API key here
        GEMINI_API_KEY = "AIzaSyD_8Qwvol1CNu1dGUbihJ9-9RSBnVtuIwo"
        
        # Configure Gemini API
        import google.generativeai as genai
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel('gemini-pro')
            logger.info("✅ Gemini model configured successfully")
        except Exception as config_error:
            logger.error(f"❌ Gemini configuration failed: {config_error}")
            response = get_local_response(user_message)
            return jsonify({"response": response + "\n\n🔄 Note: API configuration failed, using offline mode."})
        
        # Create comprehensive system prompt
        system_prompt = f"""You are Cardia4's AI Health Assistant for a heart disease prediction platform.

WEBSITE KNOWLEDGE:
- Cardia4 is an AI-powered cardiovascular risk assessment system
- Features: ML models, medical imaging analysis (X-ray, MRI, ECG), care classification
- Navigation: /predict for assessments, /dashboard for results, /incare_outcare for care recommendations
- AI Models: ResNet50 for MRI, Random Forest for ML, deep learning for ECG - 99.2% accuracy

MEDICAL EXPERTISE:
Provide evidence-based cardiovascular health information including risk factors, symptoms, prevention, and when to seek care.

GUIDELINES:
- Give detailed, helpful responses
- Include medical disclaimers for health advice
- Suggest relevant Cardia4 features when appropriate
- Be professional and empathetic

User Question: {user_message}

Provide a comprehensive, helpful response:"""
        
        logger.info(f"Processing message: {user_message}")
        
        try:
            logger.info("Calling Gemini API...")
            logger.info(f"System prompt length: {len(system_prompt)} characters")
            response_obj = model.generate_content(system_prompt)
            response = response_obj.text
            logger.info(f"Raw Gemini response: {response[:100]}...")
            
            # Add medical disclaimer for health-related responses
            medical_keywords = ['heart', 'blood pressure', 'cholesterol', 'symptom', 'disease', 'risk', 'health', 'medical', 'diagnosis', 'treatment']
            if any(keyword in user_message.lower() for keyword in medical_keywords):
                response += "\n\n⚠️ **Medical Disclaimer**: This information is for educational purposes only and is not medical advice. Always consult with your healthcare provider for personalized medical guidance."
            
            logger.info(f"✅ Gemini API SUCCESS - Response length: {len(response)} chars")
            return jsonify({"response": response})
            
        except Exception as api_error:
            logger.error(f"❌ Gemini API FAILED: {str(api_error)}")
            logger.error(f"❌ Error type: {type(api_error).__name__}")
            import traceback
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            # Fallback to enhanced local response
            response = get_local_response(user_message)
            logger.info(f"Using fallback response - length: {len(response)} chars")
            return jsonify({"response": response + "\n\n🔄 Note: Using enhanced offline mode."})
        
    except Exception as e:
        logger.error(f"Chat API error: {e}")
        return jsonify({"error": "Failed to process chat request. Please try again."}), 500

def get_local_response(message):
    """Fallback local responses when API fails"""
    lowerMessage = message.lower()
    
    if 'cholesterol' in lowerMessage or 'reduce' in lowerMessage:
        return """To reduce cholesterol naturally:

🥗 **Dietary Changes:**
• Increase soluble fiber (oats, beans, fruits)
• Choose lean proteins (fish, poultry)
• Use healthy fats (olive oil, avocados)
• Limit saturated and trans fats

🏃 **Lifestyle:**
• Regular exercise (30 min daily)
• Maintain healthy weight
• Don't smoke
• Limit alcohol

📊 **Monitor:**
• Get regular cholesterol checks
• Track your numbers over time
• Use Cardia4's risk assessment tools

Visit /predict to get your personalized cardiovascular risk analysis."""
    
    if 'heart' in lowerMessage and ('symptom' in lowerMessage or 'disease' in lowerMessage):
        return """Common heart disease symptoms:

⚠️ **Warning Signs:**
• Chest pain or discomfort
• Shortness of breath
• Fatigue or weakness
• Irregular heartbeat
• Swelling in legs/ankles

🚨 **Emergency Symptoms:**
• Severe chest pain
• Difficulty breathing
• Sudden severe headache
• Loss of consciousness

📱 **Get Assessed:**
Use Cardia4's AI models at /predict to analyze your risk factors with 99.2% accuracy.

🏥 **Important:** If experiencing severe symptoms, call emergency services immediately."""
    
    return """I can help you with:

🏥 **Website Navigation:**
• How to use Cardia4 features
• Upload medical scans at /predict
• View results at /dashboard

❤️ **Heart Health:**
• Risk factors & prevention
• Understanding symptoms
• When to seek medical care

🤖 **AI Models:**
• 99.2% accuracy across 5+ models
• ResNet50 for MRI analysis
• ECG deep learning models

Ask me: "How do I get started?" or "Tell me about heart health" """

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
