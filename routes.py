from flask import Blueprint, render_template, request, redirect, url_for
from app.recommendation_logic import save_user_answers, get_recommendations
import json

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/questionnaire')
def questionnaire():
    return render_template('questionnaire.html')

@main_bp.route('/process_answers', methods=['POST'])
def process_answers():
    data = request.get_json()
    
    genre_translation = {
        'anime': 'аниме',
         'biography': 'биография',
 	     'action': 'боевик',
 	     'war': 'военный',
 	     'detective': 'детектив',
 	     'drama': 'драма',
 	     'history': 'история', 
 	     'comedy': 'комедия',
 	     'crime': 'криминал',
 	     'romance': 'мелодрама', 
	     'cartoon': 'мультфильм', 
	     'musical': 'мюзикл', 
	     'adventure': 'приключения', 
	     'family': 'семейный',
	     'thriller': 'триллер', 
	     'horror': 'ужасы',
	     'sci-fi': 'фантастика', 
	     'film noir': 'фильм-нуар', 
 	     'fantasy': 'фэнтези'
    }
    
    russian_genres = [genre_translation[genre.lower()] 
                     for genre in data['filters']['genres']]
    
    simplified = {
        'criteria': [
            int(data['criteria']['plot']),
            int(data['criteria']['acting']),
            int(data['criteria']['humor']),
            int(data['criteria']['picture']),
            int(data['criteria']['sound'])
        ],
        'filters': {
            'age': [int(age.replace('+', '')) for age in data['filters']['age']],
            'type': 0 if 'film' in data['filters']['type'] else 1,
            'genres': russian_genres
        }
    }
    
    with open('data/user_answers.json', 'w', encoding='utf-8') as f:
        json.dump(simplified, f, ensure_ascii=False)
    
    return {'status': 'success'}

@main_bp.route('/results')
def results():
    try:
        films = get_recommendations()
        print(f"Передано фильмов в шаблон: {len(films)}")
                
        return render_template('results.html', films=films)
    except Exception as e:
        print(f"Ошибка в results: {str(e)}")
        return render_template('results.html', films=[])
