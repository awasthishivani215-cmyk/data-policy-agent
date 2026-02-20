import requests
from flask import Flask, request, jsonify, render_template
import os
import numpy as np

from src.data_loader import DataLoader
from src.policy_checker import PolicyChecker
from src.complaint_analyzer import ComplaintAnalyzer
from src.update_tracker import UpdateTracker

# Initialize Flask app with template folder
app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)

# Initialize components
policy_checker = PolicyChecker()
complaint_analyzer = ComplaintAnalyzer()
update_tracker = UpdateTracker()
data_loader = DataLoader()

@app.route('/')
def home():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/api/health')
def health():
    """API health check"""
    return jsonify({
        'status': 'online',
        'service': 'Data Policy Compliance System',
        'version': '1.0.0'
    })

@app.route('/api/check_policy', methods=['POST'])
def check_policy():
    data = request.json
    
    if 'policy_text' not in data:
        return jsonify({'error': 'No policy text provided'}), 400
    
    try:
        result = policy_checker.check_policy_compliance(data['policy_text'])
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze_complaint', methods=['POST'])
def analyze_complaint():
    data = request.json
    
    if 'complaint_text' not in data:
        return jsonify({'error': 'No complaint text provided'}), 400
    
    try:
        result = complaint_analyzer.analyze_complaint(data['complaint_text'])
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/track_update', methods=['POST'])
def track_update():
    data = request.json
    
    required_fields = ['old_policy', 'new_policy']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing field: {field}'}), 400
    
    try:
        policy_id = data.get('policy_id')
        version = data.get('version')
        
        result = update_tracker.track_policy_update(
            data['old_policy'],
            data['new_policy'],
            policy_id,
            version
        )
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload_policy_file', methods=['POST'])
def upload_policy_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        upload_dir = 'uploads'
        os.makedirs(upload_dir, exist_ok=True)
        
        filepath = os.path.join(upload_dir, file.filename)
        file.save(filepath)
        
        policy_text = data_loader.load_policy_document(filepath)
        
        result = policy_checker.check_policy_compliance(policy_text)
        
        result['filename'] = file.filename
        result['file_size'] = os.path.getsize(filepath)
        
        os.remove(filepath)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch_check', methods=['POST'])
def batch_check():
    data = request.json
    
    if 'policies' not in data or not isinstance(data['policies'], list):
        return jsonify({'error': 'Policies list required'}), 400
    
    results = []
    for policy_data in data['policies']:
        if 'text' in policy_data:
            try:
                result = policy_checker.check_policy_compliance(policy_data['text'])
                result['policy_id'] = policy_data.get('id', 'unknown')
                results.append(result)
            except:
                continue
    
    if not results:
        return jsonify({'error': 'No valid policies processed'}), 400
    
    compliance_scores = [r.get('compliance_percentage', 0) for r in results if isinstance(r.get('compliance_percentage'), (int, float))]
    summary = {
        'total_policies': len(results),
        'average_compliance': np.mean(compliance_scores) if compliance_scores else 0,
        'total_violations': sum(len(r.get('rule_violations', [])) for r in results),
        'total_warnings': sum(len(r.get('warnings', [])) for r in results)
    }
    
    return jsonify({
        'results': results,
        'summary': summary
    })

@app.route('/api/complaint_report', methods=['POST'])
def complaint_report():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    try:
        if file.filename.endswith('.csv'):
            import pandas as pd
            complaints_df = pd.read_csv(file)
        else:
            return jsonify({'error': 'Unsupported file format. Use CSV.'}), 400
        
        complaint_analyzer.load_complaints(complaints_df)
        report = complaint_analyzer.generate_complaint_report(complaints_df)
        
        return jsonify(report)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/update_history', methods=['GET'])
def get_update_history():
    policy_id = request.args.get('policy_id')
    try:
        days_back = int(request.args.get('days_back', 30))
    except:
        days_back = 30
    
    history = update_tracker.get_update_history(policy_id, days_back)
    
    return jsonify({
        'history': history,
        'count': len(history)
    })

# Template routes
@app.route('/policy_checker')
def policy_checker_page():
    return render_template('policy_checker.html')

@app.route('/complaint_analyzer')
def complaint_analyzer_page():
    return render_template('complaint_analyzer.html')

@app.route('/update_tracker')
def update_tracker_page():
    return render_template('update_tracker.html')

@app.route('/upload')
def upload_page():
    return render_template('upload.html')


# Add this route to your app.py
@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat with AI assistant (connects to Gemini API)"""
    data = request.json
    
    if 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400
    
    try:
        # Get Gemini API key from environment or config
        GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'YOUR_API_KEY_HERE')
        
        # Prepare request to Gemini API
        gemini_url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}'
        
        # Prepare conversation history
        messages = []
        if 'history' in data:
            for msg in data['history']:
                role = "user" if msg['role'] == 'user' else "model"
                messages.append({
                    "role": role,
                    "parts": [{"text": msg['content']}]
                })
        
        # Add current message
        messages.append({
            "role": "user",
            "parts": [{"text": data['message']}]
        })
        
        # Prepare the request payload
        payload = {
            "contents": messages,
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 1024,
            },
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
        }
        
        # Call Gemini API
        response = requests.post(gemini_url, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                ai_response = result['candidates'][0]['content']['parts'][0]['text']
                return jsonify({'response': ai_response})
            else:
                # Fallback response if no candidates
                return jsonify({'response': 'I understand you\'re asking about data policies. For data compliance, ensure you have clear privacy policies, proper consent mechanisms, and robust security measures in place.'})
        else:
            # Fallback response if API call fails
            fallback_responses = [
                "I'm here to help with data policy compliance questions. Remember to review GDPR requirements for EU data protection.",
                "For data compliance, ensure your policies cover data collection, storage, sharing, and user rights clearly.",
                "Data retention policies should specify exact timeframes and deletion procedures for different data types."
            ]
            import random
            return jsonify({'response': random.choice(fallback_responses)})
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ... (rest of your existing routes)

if __name__ == "__main__":
    app.run(debug=True)
