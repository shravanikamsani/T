import nltk
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app)

FAQ = {
    "What is the college name?": "The college name is XYZ College.",
    "What courses are available?": "We offer courses in Computer Science, Engineering, Mathematics, etc.",
    "What is the admission process?": "You can apply online through our website. Visit the admissions page for more details.",
    "What are the college timings?": "The college operates from 9 AM to 5 PM, Monday to Friday.",
    "Tell me about the college.": "XYZ College is a great place to learn.",
    "How do I get admitted?": "Please visit our admission page.",
    "When does the college open?": "We open at 9 AM."

}

def get_answer(query):
    questions = list(FAQ.keys())
    responses = list(FAQ.values())

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(questions + [query])
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    max_similarity_index = similarity_scores.argmax()
    max_similarity_score = similarity_scores[0, max_similarity_index]

    if max_similarity_score > 0.4:  # Adjust the threshold as needed
        return responses[max_similarity_index]
    else:
        return "Sorry, I don't understand that question. Please try again."

@app.route('/')
def home():
    return render_template('index.html')

@socketio.on('message')
def handle_message(message):
    print('Message received:', message)
    response = get_answer(message)
    emit('response', response)

if __name__ == '__main__':
    socketio.run(app, debug=True)