import os
import pandas as pd

def create_csv():
    data = {
        'tweet': [
            "I love machine learning and AI",
            "Just watched a great football match",
            "Python is an amazing programming language",
            "Had a wonderful dinner with family",
            "Bitcoin prices are soaring"
        ],
        'label': [
            "technology",
            "sports",
            "technology",
            "lifestyle",
            "finance"
        ]
    }
    df = pd.DataFrame(data)
    df.to_csv('tweets.csv', index=False)

if not os.path.exists('tweets.csv'):
    create_csv()

from entrenamiento import preprocess_text, vectorizer, model, classify_tweet

def main():
    # Cargar datos
    data = pd.read_csv('tweets.csv')
    
    # Preprocesar y vectorizar datos
    data['tweet'] = data['tweet'].apply(preprocess_text)
    X = vectorizer.transform(data['tweet'])
    
    # Clasificar tweets
    data['predicted_label'] = data['tweet'].apply(classify_tweet)
    
    # Mostrar resultados
    print(data[['tweet', 'label', 'predicted_label']])

if __name__ == "__main__":
    main()