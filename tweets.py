import pandas as pd

# Crear un DataFrame con datos de ejemplo
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

# Guardar el DataFrame como un archivo CSV
df.to_csv('tweets.csv', index=False)