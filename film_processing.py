import sqlite3
from absa import analyze_film

#Обрабатывает все отзывы для каждого фильма и сохраняет усредненные результаты
def process_movie_reviews_advanced(input_db: str, output_db: str):

    input_conn = sqlite3.connect(input_db)
    input_cur = input_conn.cursor()
    
    output_conn = sqlite3.connect(output_db)
    output_cur = output_conn.cursor()
    output_cur.execute('''CREATE TABLE IF NOT EXISTS movie_analysis
                         (id INTEGER PRIMARY KEY, analysis BLOB, review_count INTEGER)''')
    

    input_cur.execute('SELECT DISTINCT movie_id FROM movie_reviews')
    movie_ids = [row[0] for row in input_cur.fetchall()]
    
    for movie_id in movie_ids:
        input_cur.execute('SELECT review_text FROM movie_reviews WHERE movie_id = ?', (movie_id,))
        reviews = [row[0] for row in input_cur.fetchall()]
        

        all_analyses = [analyze_film(review) for review in reviews]
        avg_analysis = [sum(x)/len(x) for x in zip(*all_analyses)] if all_analyses else []
        

        output_cur.execute('INSERT INTO movie_analysis VALUES (?, ?, ?)',
                          (movie_id, str(avg_analysis), len(reviews)))
    
    output_conn.commit()
    output_conn.close()
    input_conn.close()
