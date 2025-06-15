from flask import render_template, request, jsonify
from app.models import get_recommendations, save_user_answers
from app import app
import json
import os

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/questionnaire')
def questionnaire():
    return render_template('questionnaire.html')

@app.route('/process_answers', methods=['POST'])
def process_answers():
    try:
        # Добавьте логгирование для отладки
        print("Получены данные:", request.json)
        
        if not request.json:
            return jsonify({"status": "error", "message": "No data received"}), 400

        data = request.json
        result = [
            [
                int(data.get('criteria', {}).get('plot', 0)),
                int(data.get('criteria', {}).get('acting', 0)),
                int(data.get('criteria', {}).get('humor', 0)),
                int(data.get('criteria', {}).get('picture', 0)),
                int(data.get('criteria', {}).get('sound', 0))
            ],
            data.get('filters', {}).get('age', []),
            data.get('filters', {}).get('type', []),
            data.get('filters', {}).get('genres', [])
        ]

        # Проверка перед сохранением
        print("Сформированный результат:", result)
        
        with open('data/user_answers.json', 'w') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return jsonify({"status": "success"})
    except Exception as e:
        print("Ошибка обработки:", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/results')
def results():
    try:
        with open('C:/Users/pingu/Desktop/hse/MovieRating/data/recommendations.json') as f:
            films = json.load(f)
        return render_template('results.html', films=films)
    except Exception as e:
        return render_template('error.html', message=str(e))