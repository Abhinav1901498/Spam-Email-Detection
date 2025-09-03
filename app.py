import os
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pandas as pd

 
data = {
    'text': [
        'Win a free iPhone now',
        'Meeting at 10 am',
        'Congratulations, you won a lottery',
        'Please submit the report',
        'Get your free coupon today',
        'Lunch at 1 pm'
    ],
    'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham']
}
df = pd.DataFrame(data)

 
clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])
clf.fit(df['text'], df['label'])
print("Pipeline trained successfully!")

 
base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, "templates")
app = Flask(__name__, template_folder=template_dir)

 
print("Current working directory:", os.getcwd())
print("Files in current directory:", os.listdir())
print("Flask template folder:", app.template_folder)

 
if os.path.exists(template_dir):
    print("Template folder exists:", True)
    print("Files in template folder:", os.listdir(template_dir))
else:
    print("Template folder exists:", False)
    print("Please make sure 'templates' folder exists with index.html inside it.")

 
@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    email_text = ""
    if request.method == "POST":
        email_text = request.form["email_text"]
        pred = clf.predict([email_text])[0]
        result = "Spam" if pred == "spam" else "Ham"
    return render_template("index.html", result=result, email_text=email_text)

@app.route("/login", methods=["GET"])
def login():
    return render_template("login.html")

if __name__ == "__main__":
    app.run(debug=True)
