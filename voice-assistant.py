import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

class PersonalAssistant:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.classifier = LinearSVC()

    def load_data(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        X = []
        y = []
        for line in lines:
            data = line.strip().split(',')
            if len(data) == 2:
                X.append(data[0])
                y.append(data[1])
        X = self.vectorizer.fit_transform(X)
        return X, y

    def train(self, X, y):
        self.classifier.fit(X, y)

    def predict(self, user_input):
        user_input_vector = self.vectorizer.transform([user_input])
        prediction = self.classifier.predict(user_input_vector)
        return prediction[0]

assistant = PersonalAssistant()

data_file = 'data/data.csv'
X, y = assistant.load_data(data_file)
assistant.train(X, y)

while True:
    user_input = input('Enter your query (or "quit" to exit): ')
    if user_input.lower() == 'quit':
        break
    prediction = assistant.predict(user_input)
    print('Assistant:', prediction)
