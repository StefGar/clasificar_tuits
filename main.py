import os
import pandas as pd

def crear_csv():
    datos = {
        'tweet': [
            "Me encanta el aprendizaje automático y la IA",
            "Acabo de ver un gran partido de fútbol",
            "Python es un lenguaje de programación increíble",
            "Tuve una cena maravillosa con la familia",
            "Los precios de Bitcoin están subiendo",
            "Hoy aprendí algo nuevo sobre la programación",
            "El clima está perfecto para una caminata",
            "La economía global está cambiando rápidamente",
            "Disfruté mucho la película que vi anoche",
            "La inteligencia artificial está revolucionando el mundo",
            "Estoy leyendo un libro fascinante sobre historia",
            "La música en el concierto fue espectacular",
            "El nuevo teléfono tiene características impresionantes",
            "La política actual es muy complicada",
            "Hice una receta deliciosa para la cena",
            "El turismo está creciendo en muchas ciudades",
            "La salud mental es muy importante",
            "El arte moderno es muy interesante",
            "La educación en línea está ganando popularidad",
            "La moda de este año es muy colorida",
            "La tecnología 5G está avanzando rápidamente",
            "El equipo de baloncesto ganó el campeonato",
            "La programación en JavaScript es muy versátil",
            "Pasé un día increíble en la playa",
            "Las acciones de Tesla están subiendo",
            "Aprendí sobre redes neuronales hoy",
            "El senderismo en las montañas fue refrescante",
            "La economía está en recesión",
            "Vi una serie de televisión muy entretenida",
            "La robótica está transformando la industria"
        ],
        'label': [
            "tecnología",
            "deportes",
            "tecnología",
            "estilo de vida",
            "finanzas",
            "tecnología",
            "estilo de vida",
            "finanzas",
            "entretenimiento",
            "tecnología",
            "cultura",
            "entretenimiento",
            "tecnología",
            "política",
            "estilo de vida",
            "viajes",
            "salud",
            "arte",
            "educación",
            "moda",
            "tecnología",
            "deportes",
            "tecnología",
            "estilo de vida",
            "finanzas",
            "tecnología",
            "estilo de vida",
            "finanzas",
            "entretenimiento",
            "tecnología"
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
    
    # Verificar si la columna 'label' existe
    if 'label' not in datos.columns:
        raise KeyError("La columna 'label' no existe en el archivo 'tweets.csv'.")
    
    # Preprocesar y vectorizar datos
    datos['tweet'] = datos['tweet'].apply(preprocess_text)
    X = vectorizer.transform(datos['tweet'])
    
    # Clasificar tweets
    datos['predicted_label'] = datos['tweet'].apply(classify_tweet)
    
    # Mostrar resultados
    print(datos[['tweet', 'label', 'predicted_label']])

if __name__ == "__main__":
    main()