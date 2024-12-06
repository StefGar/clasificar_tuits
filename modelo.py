from entrenamiento import preprocess_text, vectorizer, model, classify_tweet

# Ejemplo de uso
new_tweet = "This is an example tweet about machine learning."
print(f'Topic: {classify_tweet(new_tweet)}')