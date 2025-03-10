
import requests
import json
import sqlite3
import os
import dotenv
from dotenv import load_dotenv
#from src.preprocess_text import PreprocessText

connection = sqlite3.connect("film.db")
cursor = sqlite = connection.cursor()

load_dotenv()


TOKEN = os.dotenv("TOKEN")

if not TOKEN: raise ValueError("Token not found")

headers = {
    "X-API-KEY": f"{TOKEN}",
    "accept": "application/json"
}


def save_json(name, data) :
    with open(name, 'w') as json_file:
        json.dump(data, json_file, indent=4)


#gets reviews of a movie with known id through api and saves to json file
def get_reviews(movie_id):
    review_url = f"https://api.kinopoisk.dev/v1.4/review?page=1&limit=10&selectFields=id&selectFields=review&selectFields=type&movieId={movie_id}"
    response = requests.get(review_url, headers = headers)
    return response.json()


#gets movie info from json and puts it into database
def get_movie_info(movie):
    id_ = movie['id']

    cursor.execute("SELECT * FROM films WHERE ID = ?", (id_,))
    if cursor.fetchone():
        print(f"id {id_} already exists")
        return

    rating = movie['rating']['kp']
    name = movie['name']+ ' (' + str(movie['year']) + ')'
    is_series = movie['isSeries']
    if is_series:
        duration = movie['totalSeriesLength']
    else:
        duration = movie['movieLength']
    genres = json.dumps([g['name'] for g in movie.get('genres', [])], ensure_ascii=False)
    age_rating = movie["ageRating"]

    print(name)
    reviews_data = get_reviews(id_)
    reviews = json.dumps(reviews_data.get('docs', []), ensure_ascii=False)


    cursor.execute('''
                INSERT INTO films 
                (ID, NAME, YEAR, RATING, IS_SERIES, GENRES, REVIEWS, AGE_RATING, DURATION)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
        id_,
        name,
        rating,
        is_series,
        json.dumps(genres, ensure_ascii=False),
        reviews,
        age_rating,
        duration
    ))




def go_through_movie_list(file_name):
    with open(file_name, "r", encoding='utf-8') as file:
        movie_list = json.load(file)

    for i in range(100, 186):
        get_movie_info(movie_list['docs'][i])


if __name__=="__main__":

    try:
        print('db init...')

        connection.execute('''
            CREATE TABLE IF NOT EXISTS films
            (ID INTEGER PRIMARY KEY,
            NAME TEXT,
            RATING REAL,
            IS_SERIES INTEGER,
            GENRES TEXT,
            REVIEWS TEXT,
            AGE_RATING INTEGER,
            DURATION INTEGER
            );
        ''')

        go_through_movie_list("movie_list.json")

        connection.commit()
        cursor.close()

    except sqlite3.Error as e:
        print('Error occured - ', e)

    finally:
        if connection:
            connection.close()
            print("sqlite connection closed")
