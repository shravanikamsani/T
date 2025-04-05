import nltk
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app)

FAQ = {
    "What is the college name?": "The college name is DRK Institute of Science and Technology",
    "What courses are available?": "We offer courses in Computer Science Engineering, CSE with Specialization in AIML,DataScience,Cyber Security,Mechanical Engineering,Civil Engineering",
    "What is the admission process?": "You can apply online through TG EAPCET website.For Management Quota Contact Admission Department",
    "What are the college timings?": "The college operates from 9:20 AM to 4:20 PM, Monday to Friday.",
    "Tell me about the college.": "DRK Institute of Science and Technology College is a great place to learn.",
    "How do I get admit or seat?": "You Can get seat, Through TG EAPCET or Management ",
    "When does the college open?": "We open at 9:20 AM.",
    "Where College is Located or Address of college": "Near Pragathi Nagar, Bowrampet (V), Hyderabad - 500043, Telangana, India",
    "Contact Details of College":"Mobile: +91-90007 11899 or +91-87907 11899, Mail: principal@drkist.edu.in",
    "Is there a library available for students?":"Yes, There many Books in Library",
    "Are there sports or Sports and extracurricular activities?":"Yes, We have Cricket,Basket Ball,Volley Ball,Kho-Kho,Throw Ball and Batminton",
    "What are the transportation facilities Bus?":"We have Buses Facilities all over Hyderabad",
    "Where can I find my exam results?":"In JNTUH website",
    "Contact Details of College": "Mobile: +91-90007 11899 or +91-87907 11899, Mail: principal@drkist.edu.in",
    "Courses Available": "We offer courses in Computer Science Engineering, CSE with Specialization in AIML, Data Science, Cyber Security, Mechanical Engineering, Civil Engineering.",
    "College Timings": "The college operates from 9:20 AM to 4:20 PM, Monday to Friday.",
    "Address": "Near Pragathi Nagar, Bowrampet (V), Hyderabad - 500043, Telangana, India",
    "Placements": "Our college has a dedicated placement cell with a strong track record of placing students in top companies."
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