import os
import pandas as pd

def crear_csv():
    datos = {
        'tweet': [
            "Me encanta el aprendizaje automático y la IA",
            "Acabo de ver un gran partido de fútbol",
            "Python es un lenguaje de programación increíble",
            "Tuve una cena maravillosa con la familia",
            "Los precios de Bitcoin están subiendo"
        ],
        'etiqueta': [
            "tecnología",
            "deportes",
            "tecnología",
            "estilo de vida",
            "finanzas"
        ]
    }
    df = pd.DataFrame(datos)
    df.to_csv('tweets.csv', index=False)

if not os.path.exists('tweets.csv'):
    crear_csv()

from entrenamiento import preprocess_text, vectorizer, model, classify_tweet

def main():
    # Cargar datos
    datos = pd.read_csv('tweets.csv')
    
    # Verificar si la columna 'etiqueta' existe
    if 'etiqueta' not in datos.columns:
        raise KeyError("La columna 'etiqueta' no existe en el archivo 'tweets.csv'.")
    
    # Preprocesar y vectorizar datos
    datos['tweet'] = datos['tweet'].apply(preprocess_text)
    X = vectorizer.transform(datos['tweet'])
    
    # Clasificar tweets
    datos['etiqueta_predicha'] = datos['tweet'].apply(classify_tweet)
    
    # Mostrar resultados
    print(datos[['tweet', 'etiqueta', 'etiqueta_predicha']])

if __name__ == "__main__":
    main()