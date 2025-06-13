import sqlite3
import json
from src.absa.absa import analyze_review
import numpy as np


# проходится по фильмам в датабазе и обрабатывает все отзывы для каждого фильма и сохраняет усредненные результаты
def process_movies(input_db: str, output_db: str):
    input_conn = sqlite3.connect(input_db)
    input_cur = input_conn.cursor()
    print("db#1 init...")

    output_conn = sqlite3.connect(output_db)
    output_cur = output_conn.cursor()
    print("db#2 init...")

    output_cur.execute('''CREATE TABLE IF NOT EXISTS movie_analysis
                         (ID INTEGER PRIMARY KEY, ANALYSIS TEXT)''')

    input_cur.execute('SELECT DISTINCT ID FROM films')
    movie_ids = [row[0] for row in input_cur.fetchall()]

    cnt = 0
    for movie_id in movie_ids:

        cnt += 1
        if cnt % 50 == 0:
            print(f"processing movie #{cnt}...")
            output_conn.commit()

        input_cur.execute('SELECT REVIEWS FROM films WHERE ID = ?', (movie_id,))
        reviews = [row[0] for row in input_cur.fetchall()]

        all_analyses = [analyze_review(review) for review in reviews]
        avg_analysis = np.mean(all_analyses, axis=0).tolist() if all_analyses else [0, 0, 0, 0, 0]

        final_analysis = json.dumps(avg_analysis, ensure_ascii=False)

        output_cur.execute('INSERT INTO movie_analysis VALUES (?, json(?))',
                           (movie_id, final_analysis))

    output_conn.commit()
    output_conn.close()
    input_conn.close()
    print("all done!")


if __name__ == "__main__":
    process_movies("film.db", "movie_analysis.db")
