import sqlite3
import json
import numpy as np
from app import app

def get_db():
    conn = sqlite3.connect(app.config['DATABASE'])
    conn.row_factory = sqlite3.Row
    return conn

def save_user_answers(data):
    result = [
        [
            int(data['criteria'].get('plot', 0)),
            int(data['criteria'].get('acting', 0)),
            int(data['criteria'].get('humor', 0)),
            int(data['criteria'].get('picture', 0)),
            int(data['criteria'].get('sound', 0))
        ],
        data['filters'].get('age', []),
        data['filters'].get('type', []),
        data['filters'].get('genres', [])
    ]
    with open('C:/Users/pingu/Desktop/hse/MovieRating//user_answers.json', 'w') as f:
        json.dump(result, f)

def get_score_range(rating):
    return {
        5: (9, 10), 4: (7, 10), 3: (5, 10),
        2: (3, 10), 1: (1, 10), 0: (0, 10)
    }.get(rating, (0, 10))

def get_recommendations():
    try:
        with open('C:/Users/pingu/Desktop/hse/MovieRating/datauser_answers.json') as f:
            data = json.load(f)
    except:
        return []

    conn = get_db()
    try:
        query = '''SELECT f.id, f.name, f.rating, f.is_series, 
                          f.genres, f.age_rating, f.duration, f.year
                   FROM films f
                   LEFT JOIN movie_analysis ma ON f.id = ma.id
                   WHERE 1=1'''
        params = []
        
        # Добавляем фильтры
        if data[1]:  # age
            query += f" AND f.age_rating IN ({','.join(['?']*len(data[1]))})"
            params.extend(data[1])
        
        if data[2]:  # type
            types = []
            if 'film' in data[2]: types.append('f.is_series = 0')
            if 'series' in data[2]: types.append('f.is_series = 1')
            if types: query += " AND (" + " OR ".join(types) + ")"
        
        if data[3]:  # genres
            genres = [f"f.genres LIKE '%{g}%'" for g in data[3]]
            query += " AND (" + " OR ".join(genres) + ")"
        
        films = conn.execute(query, params).fetchall()
        
        recommendations = []
        for film in films:
            analysis = json.loads(film['analysis']) if film.get('analysis') else [0]*5
            relevance = calculate_relevance(analysis, data[0])
            
            recommendations.append({
                'title': film['name'],
                'rating': film['rating'],
                'year': film.get('year'),
                'genres': film['genres'].split(','),
                'age_rating': film['age_rating'],
                'type': 'Сериал' if film['is_series'] else 'Фильм',
                'duration': film['duration'],
                'relevance': relevance
            })
        
        recommendations.sort(key=lambda x: -x['relevance'])
        with open('C:/Users/pingu/Desktop/hse/MovieRating/recommendations.json', 'w') as f:
            json.dump(recommendations, f)
        
        return recommendations
    finally:
        conn.close()

def calculate_relevance(analysis, user_weights):
    match_score = 0
    max_score = 0
    for i, weight in enumerate(user_weights):
        if weight > 0:
            min_val, max_val = get_score_range(weight)
            film_score = analysis[i] if i < len(analysis) else 0
            if min_val <= film_score <= max_val:
                match_score += weight * film_score
            max_score += weight * 10
    return round((match_score / max_score * 100) if max_score > 0 else 0, 1)