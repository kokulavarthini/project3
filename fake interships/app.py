from ml_model import predict_fraud, train_model
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from models import db, User, Submission
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'fake-internship-detection-secret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        is_admin = True if request.form.get('is_admin') else False
        
        user_exists = User.query.filter_by(username=username).first()
        if user_exists:
            flash('Username already exists.', 'error')
            return redirect(url_for('register'))
            
        from werkzeug.security import generate_password_hash
        new_user = User(
            username=username, 
            password=generate_password_hash(password, method='scrypt'),
            is_admin=is_admin
        )
        db.session.add(new_user)
        db.session.commit()
        flash('Account created! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        from werkzeug.security import check_password_hash
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password.', 'error')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    history = Submission.query.filter_by(user_id=current_user.id).order_by(Submission.timestamp.desc()).all()
    return render_template('dashboard.html', history=history)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    job_text = request.form.get('job_text')
    if not job_text:
        flash('Please enter some text to analyze.', 'error')
        return redirect(url_for('dashboard'))
    
    prediction, confidence = predict_fraud(job_text)
    
    new_submission = Submission(
        job_text=job_text,
        prediction=prediction,
        confidence=confidence,
        user_id=current_user.id
    )
    db.session.add(new_submission)
    db.session.commit()
    
    history = Submission.query.filter_by(user_id=current_user.id).order_by(Submission.timestamp.desc()).all()
    result = {'prediction': prediction, 'confidence': confidence}
    return render_template('dashboard.html', history=history, result=result)

@app.route('/admin')
@login_required
def admin():
    if not current_user.is_admin:
        flash('Access denied.', 'error')
        return redirect(url_for('dashboard'))
    
    submissions = Submission.query.order_by(Submission.timestamp.desc()).all()
    total_submissions = len(submissions)
    total_frauds = len([s for s in submissions if s.prediction == 'Fake'])
    
    model_accuracy = 92.5 
    
    return render_template('admin.html', 
                           submissions=submissions, 
                           total_submissions=total_submissions,
                           total_frauds=total_frauds,
                           model_accuracy=model_accuracy)

if __name__ == '__main__':
    print("Starting Flask application...")
    with app.app_context():
        print("Initializing database...")
        db.create_all()
        print("Database initialized.")
        if not os.path.exists('model.pkl'):
            print("Model not found. Starting training...")
            train_model()
            print("Model trained.")
        else:
            print("Loading existing model...")
    print("Running Flask app on port 5000...")
    app.run(debug=True, use_reloader=False)
