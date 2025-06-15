import sqlite3
import json
from srsly import json_dumps

from src.absa.absa import analyze_review


# проходится по фильмам в датабазе и обрабатывает все отзывы для каждого фильма и сохраняет усредненные результаты
def process_movies(input_db: str):
    conn = sqlite3.connect(input_db)
    cur = conn.cursor()
    print("db#1 init...")

    cur.execute('''CREATE TABLE IF NOT EXISTS movie_analysis
                         (ID INTEGER PRIMARY KEY, ANALYSIS TEXT)''')

    cur.execute('SELECT DISTINCT ID FROM films')
    movie_ids = [row[0] for row in cur.fetchall()]

    cnt = 0
    for movie_id in movie_ids:

        cur.execute("SELECT * FROM movie_analysis WHERE ID = ? ", (movie_id,))

        if cur.fetchone():
            print("already exists")
            continue

        movie_analysis = [0, 0, 0, 0, 0]
        cnt += 1
        print(f"processing movie #{cnt} with id {movie_id}")

        if cnt % 10 == 0:
            conn.commit()

        cur.execute('SELECT REVIEWS FROM films WHERE ID = ?', (movie_id,))

        reviews = []
        for row in cur:
            reviews.extend(json.loads(row[0]))

        for review in reviews:
            analysis = analyze_review(review['review'])
            for i in range(len(analysis)): movie_analysis[i] += analysis[i]

        print("result: ", movie_analysis)

        try:
            cur.execute('INSERT INTO movie_analysis VALUES (?, json(?))',
                        (movie_id, json_dumps(movie_analysis)))
        except sqlite3.IntegrityError:
            cur.execute("UPDATE movie_analysis SET ANALYSIS = json(?) WHERE ID = ?",
                        (json_dumps(movie_analysis), movie_id))

    conn.commit()
    conn.close()
    conn.close()
    print("all done!")


if __name__ == "__main__":
    process_movies("film.db")
