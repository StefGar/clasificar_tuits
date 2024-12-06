from entrenamiento import preprocess_text, vectorizer, model

def classify_tweet(tweet):
    tweet = preprocess_text(tweet)
    tweet_vec = vectorizer.transform([tweet])
    prediction = model.predict(tweet_vec)
    return prediction[0]

# Ejemplo de uso
new_tweet = "This is an example tweet about machine learning."
print(f'Topic: {classify_tweet(new_tweet)}')