from flask import Flask, request, jsonify, render_template_string
import joblib
import re

app = Flask(__name__)

# Load model
try:
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
except:
    model = vectorizer = None

def clean_text(text):
    text = str(text).lower()
    return re.sub(r'[^a-zA-Z\s]', '', text)

HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>Fake News Detector</title>
    <style>
        body { font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px; background: #f5f5f5; }
        .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; text-align: center; }
        textarea { width: 100%; height: 200px; padding: 15px; border: 2px solid #ddd; border-radius: 5px; font-size: 16px; }
        button { width: 100%; padding: 15px; background: #007bff; color: white; border: none; border-radius: 5px; font-size: 18px; cursor: pointer; margin-top: 10px; }
        button:hover { background: #0056b3; }
        .result { margin-top: 20px; padding: 20px; border-radius: 5px; display: none; }
        .real { background: #d4edda; border: 2px solid #28a745; }
        .fake { background: #f8d7da; border: 2px solid #dc3545; }
        .loading { display: none; text-align: center; margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Fake News Detector</h1>
        <textarea id="text" placeholder="Paste news article here..."></textarea>
        <button onclick="analyze()">Analyze News</button>
        <div class="loading" id="loading">Analyzing...</div>
        <div class="result" id="result"></div>
    </div>
    
    <script>
        async function analyze() {
            const text = document.getElementById('text').value;
            if (!text.trim()) { alert('Please enter text'); return; }
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').style.display = 'none';
            
            const res = await fetch('/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({text})
            });
            
            const data = await res.json();
            document.getElementById('loading').style.display = 'none';
            
            const resultDiv = document.getElementById('result');
            resultDiv.className = 'result ' + (data.prediction === 'FAKE' ? 'fake' : 'real');
            resultDiv.innerHTML = `
                <h2>${data.prediction} NEWS</h2>
                <p>Confidence: ${data.confidence.toFixed(1)}%</p>
            `;
            resultDiv.style.display = 'block';
        }
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML)

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
    
    text = request.json.get('text', '')
    clean = clean_text(text)
    vec = vectorizer.transform([clean])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    
    return jsonify({
        'prediction': 'FAKE' if pred == 1 else 'REAL',
        'confidence': float(max(prob) * 100)
    })

if __name__ == '__main__':
    print("\n🚀 Starting server at http://127.0.0.1:5000\n")
    app.run(debug=True)