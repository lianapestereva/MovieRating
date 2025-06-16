from app import db

class Film(db.Model):
    __tablename__ = 'films'
    
    ID = db.Column('ID', db.Integer, primary_key=True)
    NAME = db.Column('NAME', db.String(100), nullable=False)
    RATING = db.Column('RATING', db.Float)
    IS_SERIES = db.Column('IS_SERIES', db.Boolean)
    GENRES = db.Column('GENRES', db.String(50))
    REVIEWS = db.Column('REVIEWS', db.Integer)
    AGE_RATING = db.Column('AGE_RATING', db.String(10))
    DURATION = db.Column('DURATION', db.Integer)
    YEAR = db.Column('YEAR', db.Integer)

class MovieAnalysis(db.Model):
    __tablename__ = 'movie_analysis'
    
    ID = db.Column('ID', db.Integer, primary_key=True)
    ANALYSIS = db.Column('ANALYSIS', db.String(100)) 
