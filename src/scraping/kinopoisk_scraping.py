
import requests
import json
import sqlite3
import os
from dotenv import load_dotenv


load_dotenv()

TOKEN = os.getenv("TOKEN")

if not TOKEN: raise ValueError("Token not found")

headers = {
    "X-API-KEY": f"{TOKEN}",
    "accept": "application/json"
}

connection = sqlite3.connect("../../film.db")
cursor = connection.cursor()
print('db init...')


def save_json(name, data) :
    with open(name, 'w') as json_file:
        json.dump(data, json_file, indent=4)


#gets reviews of a movie with known id through api and saves to json file
def get_reviews(movie_id: int):
    try:
        review_url = f"https://api.kinopoisk.dev/v1.4/review?page=1&limit=10&selectFields=id&selectFields=review&selectFields=type&movieId={movie_id}"
        response = requests.get(review_url, headers = headers)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f'could not access review url: {e}')


#gets movie info from json and puts it into database
def get_movie_info(movie: json):
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
    year = movie['year']

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
        year,
        rating,
        is_series,
        genres,
        reviews,
        age_rating,
        duration
    ))


def get_json_movie_list_through_api(url):
    response = requests.get(url, headers=headers)
    return response.json()


def go_through_movie_list(movie_data: dict, length=50):
    if not movie_data:
        print("Invalid movie data format")
        return

    for i in range(0, length):
        get_movie_info(movie_data['docs'][i])


def save_movies_to_db(url: str):
    try:
        connection.execute('''
            CREATE TABLE IF NOT EXISTS films
            (ID INTEGER PRIMARY KEY,
            NAME TEXT,
            YEAR INTEGER,
            RATING REAL,
            IS_SERIES INTEGER,
            GENRES TEXT,
            REVIEWS TEXT,
            AGE_RATING INTEGER,
            DURATION INTEGER
            );
        ''')

        go_through_movie_list(get_json_movie_list_through_api(url))

        connection.commit()
        cursor.close()

    except sqlite3.Error as e:
        print('Error occured - ', e)

    finally:
        if connection:
            connection.close()
            print("sqlite connection closed")



if __name__=="__main__":
    print("Using this file, you can add a list of movies via url to the pre-existing database film.db")

    print("To start press enter: ")
    url = "https://api.kinopoisk.dev/v1.4/movie?page=1&limit=50&selectFields=id&selectFields=name&selectFields=year&selectFields=rating&selectFields=movieLength&selectFields=totalSeriesLength&selectFields=seriesLength&selectFields=ageRating&selectFields=genres&selectFields=isSeries&lists=top500"

    save_movies_to_db(url)

    print("all done!")


