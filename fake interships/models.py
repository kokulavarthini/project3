from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    submissions = db.relationship('Submission', backref='user', lazy=True)

class Submission(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    job_text = db.Column(db.Text, nullable=False)
    prediction = db.Column(db.String(20), nullable=False) # 'Fake' or 'Real'
    confidence = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
