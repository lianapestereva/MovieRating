def process_movie_reviews_advanced(input_db: str, output_db: str):
    """Обрабатывает все отзывы для каждого фильма и сохраняет усредненные результаты"""
    input_conn = sqlite3.connect(input_db)
    input_cur = input_conn.cursor()
    
    output_conn = sqlite3.connect(output_db)
    output_cur = output_conn.cursor()
    output_cur.execute('''CREATE TABLE IF NOT EXISTS movie_analysis
                         (id INTEGER PRIMARY KEY, analysis BLOB, review_count INTEGER)''')
    
    # Получаем все фильмы
    input_cur.execute('SELECT DISTINCT movie_id FROM movie_reviews')
    movie_ids = [row[0] for row in input_cur.fetchall()]
    
    for movie_id in movie_ids:
        # Получаем все отзывы для этого фильма
        input_cur.execute('SELECT review_text FROM movie_reviews WHERE movie_id = ?', (movie_id,))
        reviews = [row[0] for row in input_cur.fetchall()]
        
        # Анализируем каждый отзыв и усредняем результаты
        all_analyses = [analyse(review) for review in reviews]
        avg_analysis = [sum(x)/len(x) for x in zip(*all_analyses)] if all_analyses else []
        
        # Сохраняем результат
        output_cur.execute('INSERT INTO movie_analysis VALUES (?, ?, ?)',
                          (movie_id, str(avg_analysis), len(reviews)))
    
    output_conn.commit()
    output_conn.close()
    input_conn.close()
