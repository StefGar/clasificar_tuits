import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
import string

# Descargar stopwords
nltk.download('stopwords')

# Cargar datos
data = pd.read_csv('tweets.csv')  # Asegúrate de tener un archivo tweets.csv con columnas 'tweet' y 'label'

# Preprocesamiento de texto
def preprocess_text(text):
    # Convertir a minúsculas
    text = text.lower()
    # Eliminar puntuación
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Eliminar stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

data['tweet'] = data['tweet'].apply(preprocess_text)

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(data['tweet'], data['label'], test_size=0.2, random_state=42)

# Vectorización
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Entrenamiento del modelo
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predicción y evaluación
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Función para clasificar un nuevo tweet
def classify_tweet(tweet):
    tweet = preprocess_text(tweet)
    tweet_vec = vectorizer.transform([tweet])
    prediction = model.predict(tweet_vec)
    return prediction[0]

# Ejemplo de uso
new_tweet = "This is an example tweet about machine learning."
print(f'Topic: {classify_tweet(new_tweet)}')