from flask import Flask, render_template, request, jsonify

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
    
    # Сохраняем ответы в файл (для демонстрации)
    with open('user_answers.txt', 'w') as f:
        f.write(f"Критерии:\n")
        f.write(f"Оценка: {data['rating']}\n")
        f.write(f"Сюжет: {data['plot']}\n")
        f.write(f"Картинка: {data['visuals']}\n")
        f.write(f"Актерская игра: {data['acting']}\n")
        f.write(f"Звуковой дизайн: {data['sound']}\n\n")
        
        f.write(f"Фильтры:\n")
        f.write(f"Возраст: {data['age']}\n")
        f.write(f"Тип: {data['type']}\n")
        f.write(f"Жанры: {data['genres']}\n")
    
    # В реальном приложении здесь была бы логика рекомендации
    return jsonify({"status": "success"})

@app.route('/results')
def results():
    return render_template('results.html')

if __name__ == '__main__':
    app.run(debug=True)
