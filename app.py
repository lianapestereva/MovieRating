from flask import Flask, render_template, request, jsonify
import json
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/questionnaire')
def questionnaire():
    return render_template('questionnaire.html')

@app.route('/process_answers', methods=['POST'])
def process_answers():
    data = request.json
    
    result = [
        [
            int(data['criteria']['humor']),
            int(data['criteria']['plot']),
            int(data['criteria']['visuals']),
            int(data['criteria']['acting']),
            int(data['criteria']['sound'])
        ],
        data['filters']['age'],
        data['filters']['type'],
        data['filters']['genres']
    ]
    
    with open('user_answers.json', 'w') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    return jsonify({"status": "success"})

@app.route('/results')
def results():
    return render_template('results.html')

if __name__ == '__main__':
    if not os.path.exists('user_answers.json'):
        with open('user_answers.json', 'w') as f:
            json.dump({}, f)
    app.run(debug=True)
