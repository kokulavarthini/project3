print("Testing Python environment...")
import flask
import sqlalchemy
import flask_login
import pandas
import numpy
import nltk
import sklearn
print("All imports successful!")

try:
    print("Testing NLTK download...")
    nltk.download('punkt')
    print("NLTK download successful!")
except Exception as e:
    print(f"NLTK error: {e}")

print("Diagnostic complete.")
