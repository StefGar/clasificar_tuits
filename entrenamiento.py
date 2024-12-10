import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
import string
from sklearn.metrics import classification_report

print("Start of the training script")

# Descargar stopwords si no están disponibles
try:
    stop_words = set(stopwords.words('spanish'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('spanish'))

# Asegurarse de que el archivo CSV exista
if not os.path.exists('tweets.csv'):
    from main import crear_csv
    crear_csv()

# Verificar nuevamente si el archivo fue creado exitosamente
if not os.path.exists('tweets.csv'):
    raise FileNotFoundError("El archivo 'tweets.csv' no existe y no pudo ser creado.")

# Cargar datos
datos = pd.read_csv('tweets.csv')  # Asegúrate de tener un archivo tweets.csv con columnas 'tweet' y 'label'

# Asegurarse de que el archivo CSV tenga las columnas correctas
if 'label' not in datos.columns:
    raise KeyError("La columna 'label' no existe en el archivo 'tweets.csv'.")

# Preprocesamiento de texto
def preprocess_text(texto):
    # Convertir a minúsculas
    texto = texto.lower()
    # Eliminar puntuación
    texto = texto.translate(str.maketrans('', '', string.punctuation))
    # Eliminar stopwords
    texto = ' '.join([palabra for palabra in texto.split() if palabra not in stop_words])
    return texto

datos['tweet'] = datos['tweet'].apply(preprocess_text)

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(datos['tweet'], datos['label'], test_size=0.2, random_state=42)

# Vectorización
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Entrenamiento del modelo
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predicción y evaluación
y_pred = model.predict(X_test_vec)

# Función para obtener la precisión del modelo
def get_model_accuracy():
    return accuracy_score(y_test, y_pred)

# Verificar predicciones
accuracy = get_model_accuracy()
print(f'Accuracy: {accuracy:.2f}')  # Display accuracy with two decimals

print("End of the training script")

# Función para clasificar un nuevo tweet
def classify_tweet(tweet):
    tweet = preprocess_text(tweet)
    tweet_vec = vectorizer.transform([tweet])
    prediction = model.predict(tweet_vec)
    return prediction[0]

# Ejemplo de uso
nuevo_tweet = "Este es un tweet de ejemplo sobre aprendizaje automático."
print(f'Tema: {classify_tweet(nuevo_tweet)}')