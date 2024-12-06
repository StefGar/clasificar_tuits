import os
import pandas as pd

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

def actualizar_csv(nuevos_datos):
    if os.path.exists('tweets.csv'):
        df = pd.read_csv('tweets.csv')
        df = pd.concat([df, pd.DataFrame(nuevos_datos)], ignore_index=True)
        df.to_csv('tweets.csv', index=False)
    else:
        crear_csv()

if not os.path.exists('tweets.csv'):
    crear_csv()

from entrenamiento import preprocess_text, vectorizer, model, classify_tweet

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

def main():
    print("Start of the main function")
    # Cargar datos
    datos = pd.read_csv('tweets.csv')
    
    # Verificar si la columna 'label' existe
    if 'label' not in datos.columns:
        raise KeyError("The 'label' column does not exist in the 'tweets.csv' file.")
    
    # Preprocesar y vectorizar datos
    datos['tweet'] = datos['tweet'].apply(preprocess_text)
    X = vectorizer.transform(datos['tweet'])
    
    # Clasificar tweets
    datos['predicted_label'] = datos['tweet'].apply(lambda tweet: classify_tweet(tweet))
    
    # Mostrar todos los resultados en formato tabular
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(datos[['tweet', 'label', 'predicted_label']].to_string(index=False))
    
    print("End of the main function")

if __name__ == "__main__":
    main()