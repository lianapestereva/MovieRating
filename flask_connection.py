from flask import Flask, render_template, request, jsonify
import sqlite3
import json
import os
import numpy as np
from src.absa import analyze_film  # Импорт функции анализа отзывов

app = Flask(__name__)

# Конфигурация БД
DATABASE = 'films.db'
ANALYSIS_DB = 'film_analysis.db'


def get_db_connection(db_file=DATABASE):
    conn = sqlite3.connect(db_file)
    conn.row_factory = sqlite3.Row
    return conn


def analyze_movies():
    """Анализирует все фильмы и сохраняет результаты в отдельную БД"""
    input_conn = get_db_connection(DATABASE)
    output_conn = get_db_connection(ANALYSIS_DB)

    try:
        input_cur = input_conn.cursor()
        output_cur = output_conn.cursor()

        # Создаем таблицу для анализа, если ее нет
        output_cur.execute('''CREATE TABLE IF NOT EXISTS film_analysis
                              (
                                  id
                                  INTEGER
                                  PRIMARY
                                  KEY,
                                  humor
                                  REAL,
                                  plot
                                  REAL,
                                  visuals
                                  REAL,
                                  acting
                                  REAL,
                                  sound
                                  REAL,
                                  review_count
                                  INTEGER
                              )''')

        # Получаем все фильмы с отзывами
        input_cur.execute('''SELECT f.id,
                                    f.name,
                                    f.rating,
                                    f.is_series,
                                    f.genres,
                                    f.age_ratings,
                                    f.duration,
                                    GROUP_CONCAT(r.text) AS reviews
                             FROM films f
                                      LEFT JOIN reviews r ON f.id = r.film_id
                             GROUP BY f.id''')

        for film in input_cur.fetchall():
            reviews = film['reviews'].split('|') if film['reviews'] else []

            # Анализируем отзывы
            if reviews:
                analyses = [analyze_film(review) for review in reviews]
                avg_analysis = {
                    'humor': np.mean([a['humor'] for a in analyses]),
                    'plot': np.mean([a['plot'] for a in analyses]),
                    'visuals': np.mean([a['visuals'] for a in analyses]),
                    'acting': np.mean([a['acting'] for a in analyses]),
                    'sound': np.mean([a['sound'] for a in analyses])
                }
            else:
                avg_analysis = {'humor': 0, 'plot': 0, 'visuals': 0, 'acting': 0, 'sound': 0}

            # Сохраняем анализ в БД
            output_cur.execute('''INSERT OR REPLACE INTO film_analysis 
                                VALUES (?, ?, ?, ?, ?, ?, ?)''',
                               (film['id'],
                                avg_analysis['humor'],
                                avg_analysis['plot'],
                                avg_analysis['visuals'],
                                avg_analysis['acting'],
                                avg_analysis['sound'],
                                len(reviews)))

        output_conn.commit()

    finally:
        input_conn.close()
        output_conn.close()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/questionnaire')
def questionnaire():
    return render_template('questionnaire.html')


@app.route('/process_answers', methods=['POST'])
def process_answers():
    try:
        data = request.json

        # Получаем критерии оценки
        user_weights = [
            float(data['criteria']['humor']),
            float(data['criteria']['plot']),
            float(data['criteria']['visuals']),
            float(data['criteria']['acting']),
            float(data['criteria']['sound'])
        ]

        # Получаем фильтры
        filters = {
            'age': data['filters']['age'],
            'type': data['filters']['type'],
            'genres': data['filters']['genres']
        }

        # Получаем рекомендации
        recommended_films = get_recommendations(user_weights, filters)

        # Сохраняем результаты
        with open('recommendations.json', 'w') as f:
            json.dump(recommended_films, f, ensure_ascii=False, indent=2)

        return jsonify({"status": "success"})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


def get_recommendations(user_weights: list, filters: dict) -> list:
    """Возвращает отсортированный список фильмов по релевантности"""
    conn = get_db_connection()
    analysis_conn = get_db_connection(ANALYSIS_DB)

    try:
        # Получаем фильмы с анализом
        cur = conn.cursor()
        analysis_cur = analysis_conn.cursor()

        query = '''SELECT f.*, a.humor, a.plot, a.visuals, a.acting, a.sound
                   FROM films f
                            LEFT JOIN film_analysis a ON f.id = a.id
                   WHERE 1 = 1'''

        # Применяем фильтры
        params = []

        if filters['age']:
            query += ' AND f.age_ratings IN ({})'.format(','.join(['?'] * len(filters['age'])))
            params.extend(filters['age'])

        if filters['type']:
            type_conditions = []
            if 'film' in filters['type']:
                type_conditions.append('f.is_series = 0')
            if 'series' in filters['type']:
                type_conditions.append('f.is_series = 1')
            if type_conditions:
                query += ' AND (' + ' OR '.join(type_conditions) + ')'

        if filters['genres']:
            genre_conditions = []
            for genre in filters['genres']:
                genre_conditions.append(f"f.genres LIKE '%{genre}%'")
            query += ' AND (' + ' OR '.join(genre_conditions) + ')'

        cur.execute(query, params)
        films = cur.fetchall()

        # Рассчитываем релевантность
        recommended_films = []
        for film in films:
            # Получаем оценки из анализа
            film_scores = [
                film['humor'] or 0,
                film['plot'] or 0,
                film['visuals'] or 0,
                film['acting'] or 0,
                film['sound'] or 0
            ]

            # Рассчитываем релевантность
            relevance = np.dot(user_weights, film_scores)

            # Нормализуем до 100%
            max_possible = sum(5 * w for w in user_weights)  # 5 - максимальная оценка
            relevance_percent = (relevance / max_possible) * 100 if max_possible > 0 else 0

            recommended_films.append({
                'id': film['id'],
                'title': film['name'],
                'rating': film['rating'],
                'year': film.get('year'),
                'genres': film['genres'].split(',') if film['genres'] else [],
                'age_rating': film['age_ratings'],
                'type': 'Сериал' if film['is_series'] else 'Фильм',
                'duration': film['duration'],
                'relevance': round(relevance_percent, 1)
            })

        # Сортировка по релевантности
        recommended_films.sort(key=lambda x: x['relevance'], reverse=True)

        return recommended_films[:10]  # Топ-10 рекомендаций

    finally:
        conn.close()
        analysis_conn.close()


@app.route('/results')
def results():
    try:
        with open('recommendations.json', 'r') as f:
            films = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        films = []

    return render_template('results.html', films=films)


if __name__ == '__main__':
    # Инициализация БД анализа при первом запуске
    if not os.path.exists(ANALYSIS_DB):
        analyze_movies()

    app.run(debug=True)