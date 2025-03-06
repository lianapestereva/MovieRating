from bs4 import BeautifulSoup
import sqlite3
import requests
import json

TOKEN = "4FV4SHD-7MPMSTF-HZPVTKM-JBYWWVG"

headers = {
    "X-API-KEY": f"{TOKEN}",
    "accept": "application/json"
}

seen_urls = set()
links = []

def save_json(name, data) :
    with open(name, 'w') as json_file:
        json.dump(data, json_file, indent=4)


#gets reviews of a movie with known id through api and saves to json file
def get_reviews(movie_id):
    review_url = f"https://api.kinopoisk.dev/v1.4/review?page=1&limit=10&selectFields=id&selectFields=review&selectFields=type&movieId={movie_id}"
    response = requests.get(review_url, headers = headers)

    save_json("reviews"+str(movie_id)+".json", response)


def get_movie_info(movie):
    id_ = movie['id']
    rating = movie['rating']['kp']
    name = movie['name']+ ' (' + str(movie['year']) + ')'
    is_series = movie['isSeries']
    if is_series:
        duration = movie['totalSeriesLength']
    else:
        duration = movie['movieLength']
    genres=[]
    for i in movie['genres']:
        genres.append(i['name'])
    age_rating = movie["ageRating"]

    get_reviews(id_)



def go_through_movie_list():
    with open("movie_list.json", "r", encoding='utf-8') as file:
        movie_list = json.load(file)
    for i in range(0, 10):
        get_movie_info(movie_list['docs'][i])


if __name__=="__main__":
    #go_through_movie_list()