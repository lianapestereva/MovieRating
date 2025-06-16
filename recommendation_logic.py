import json
import re
from app_flask import db
from app_flask.models import Film, MovieAnalysis

def get_recommendations():
    print("Проверка данных в БД:")
    print("Всего фильмов:", Film.query.count())
    print("Всего анализов:", MovieAnalysis.query.count())

def parse_analysis(analysis_str):
    try:
        cleaned = analysis_str.strip('[]')
        return [int(x) for x in cleaned.split(',')]
    except:
        return [0, 0, 0, 0, 0]

def save_user_answers(data):
    with open('collect-data/user_answers.json', 'w', encoding='utf-8') as f:
        json.dump(data, f)

def get_recommendations():
    with open('collect-data/user_answers.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    criteria = data['criteria']
    filters = data['filters']
    
    all_analysis = MovieAnalysis.query.all()
    
    matching_ids = []
    param_order = ['plot', 'acting', 'humor', 'picture', 'sound']
    
    for analysis in all_analysis:
        scores = parse_analysis(analysis.ANALYSIS)
        match = all(
            scores[i] >= (2 * criteria[i] - 1)
            for i in range(5)
            if criteria[i] > 0 
        )
        
        if match:
            matching_ids.append(analysis.ID)

    film_query = Film.query.filter(Film.ID.in_(matching_ids))
    
    if filters['age']:
        max_age = max(filters['age'])
        film_query = film_query.filter(Film.AGE_RATING <= max_age)
    
    film_query = film_query.filter(Film.IS_SERIES == bool(filters['type']))
    
    if filters['genres']:
        genre_conditions = [Film.GENRES.contains(genre) for genre in filters['genres']]
        film_query = film_query.filter(db.or_(*genre_conditions))
    
    film_query = film_query.order_by(Film.RATING.desc())
    
    return film_query.all()
