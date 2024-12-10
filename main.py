import os
import pandas as pd

# Set display options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

def crear_csv():
    datos = {
        'tweet': [
            "I love machine learning and AI",
            "I just watched a great football match",
            "Python is an amazing programming language",
            "I had a wonderful dinner with family",
            "Bitcoin prices are rising",
            "Today I learned something new about programming",
            "The weather is perfect for a hike",
            "The global economy is changing rapidly",
            "I really enjoyed the movie I watched last night",
            "Artificial intelligence is revolutionizing the world",
            "I'm reading a fascinating book about history",
            "The music at the concert was spectacular",
            "The new phone has impressive features",
            "Current politics are very complicated",
            "I made a delicious recipe for dinner",
            "Tourism is growing in many cities",
            "Mental health is very important",
            "Modern art is very interesting",
            "Online education is gaining popularity",
            "This year's fashion is very colorful",
            "5G technology is advancing rapidly",
            "The basketball team won the championship",
            "Programming in JavaScript is very versatile",
            "I had an amazing day at the beach",
            "Tesla stocks are rising",
            "I learned about neural networks today",
            "Hiking in the mountains was refreshing",
            "The economy is in recession",
            "I watched a very entertaining TV series",
            "Robotics is transforming the industry",
            "Climate change is a global threat",
            "Artificial intelligence is improving medicine",
            "The new music album is incredible",
            "The economy is recovering",
            "Sports are essential for health",
            "Programming in Python is very popular",
            "Distance education is the future",
            "Sustainable fashion is booming",
            "Blockchain technology is revolutionizing finance",
            "Digital art is gaining popularity"
        ],
        'label': [
            "technology",
            "sports",
            "technology",
            "lifestyle",
            "finance",
            "technology",
            "lifestyle",
            "finance",
            "entertainment",
            "technology",
            "culture",
            "entertainment",
            "technology",
            "politics",
            "lifestyle",
            "travel",
            "health",
            "art",
            "education",
            "fashion",
            "technology",
            "sports",
            "technology",
            "lifestyle",
            "finance",
            "technology",
            "lifestyle",
            "finance",
            "entertainment",
            "technology",
            "environment",
            "technology",
            "entertainment",
            "finance",
            "health",
            "technology",
            "education",
            "fashion",
            "technology",
            "art"
        ]
    }
    df = pd.DataFrame(datos)
    df.to_csv('tweets.csv', index=False)
    print("CSV file created successfully.")

def actualizar_csv(nuevos_datos):
    if os.path.exists('tweets.csv'):
        df = pd.read_csv('tweets.csv')
        df = pd.concat([df, pd.DataFrame(nuevos_datos)], ignore_index=True)
        df.to_csv('tweets.csv', index=False)
        print("CSV file updated successfully.")
    else:
        crear_csv()

if not os.path.exists('tweets.csv'):
    crear_csv()

from entrenamiento import preprocess_text, vectorizer, model, classify_tweet, get_model_accuracy
from sklearn.metrics import classification_report, accuracy_score

def display_all_tweets(datos):
    tweets_list = datos['tweet'].tolist()
    for tweet in tweets_list:
        print(tweet)

def display_all_tweets_from_csv():
    if os.path.exists('tweets.csv'):
        datos = pd.read_csv('tweets.csv')
        pd.set_option('display.max_rows', None)  # Ensure all rows are displayed
        print(datos[['tweet', 'label']].to_string(index=False))
    else:
        print("The 'tweets.csv' file does not exist.")

def display_classification_report(datos):
    print(classification_report(datos['label'], datos['predicted_label'], zero_division=0))

def display_all_tweets_with_labels(datos):
    print(datos[['tweet', 'label', 'predicted_label']].to_string(index=False))

def display_misclassified_tweets(datos):
    misclassified = datos[datos['label'] != datos['predicted_label']]
    if not misclassified.empty:
        print("Misclassified tweets:")
        print(misclassified[['tweet', 'label', 'predicted_label']].to_string(index=False))
    else:
        print("No misclassified tweets found.")

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    clear_console()
    # Crear CSV si no existe
    if not os.path.exists('tweets.csv'):
        crear_csv()
    
    # Cargar datos
    datos = pd.read_csv('tweets.csv')
    
    # Verificar si la columna 'label' existe
    if 'label' not in datos.columns:
        raise KeyError("The 'label' column does not exist in the 'tweets.csv' file.")
    
    # Preprocesar y vectorizar datos
    datos['tweet'] = datos['tweet'].apply(preprocess_text)
    X = vectorizer.transform(datos['tweet'])
    
    # Clasificar tweets
    datos['predicted_label'] = model.predict(X)
    
    # Calcular y mostrar la precisión
    accuracy = accuracy_score(datos['label'], datos['predicted_label'])
    print(f'Accuracy: {accuracy:.2f}')
    
    # Mostrar todos los resultados en formato tabular
    print(datos[['tweet', 'label', 'predicted_label']].to_string(index=False))
    
    # Mostrar el reporte de clasificación
    display_classification_report(datos)
    
    # Mostrar más tweets con etiquetas y etiquetas predichas
    display_all_tweets_with_labels(datos)
    
    # Mostrar tweets mal clasificados
    display_misclassified_tweets(datos)

if __name__ == "__main__":
    main()