import pandas as pd
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