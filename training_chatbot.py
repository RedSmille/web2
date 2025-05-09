import nltk
nltk.download('punkt')

from nltk.stem import WordNetLemmatizer 
import json
import pickle
import numpy as np 
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.optimizers.schedules import ExponentialDecay
import random

# Descarga de recursos necesarios
nltk.download('punkt')
nltk.download('wordnet')

# Carga del archivo JSON
with open('Informacion.json', 'r', encoding='utf-8') as Archivo:
    Intentos = json.load(Archivo)

Lematizador = WordNetLemmatizer()

Palabras = []
Clases = []
Documentos = []
IgnorarPalabras = ['.', ',', ';', ':', '"', "'", '(', ')', '[', ']', '{', '}', '¿', '?', '¡', '!']

# Recorre cada intento y sus patrones en el archivo
for Intento in Intentos['intents']:
    for Patron in Intento['preguntas']:       
        # Tokeniza las palabras en cada patrón
        PalabrasTokenizadas = nltk.word_tokenize(Patron)
        Palabras.extend(PalabrasTokenizadas)
        
        # Agrega el par (patrón, etiqueta) a la lista de documentos
        Documentos.append((PalabrasTokenizadas, Intento['tag']))
        
        # Si la etiqueta no está en la lista de clases, la agrega
        if Intento["tag"] not in Clases:
            Clases.append(Intento['tag'])

# Lematización y ordenamiento de palabras y clases
Palabras = [Lematizador.lemmatize(Palabra.lower()) for Palabra in Palabras if Palabra not in IgnorarPalabras]
Palabras = sorted(list(set(Palabras)))
Clases = sorted(list(set(Clases)))

# Guarda las listas de palabras y clases en archivos pickle
pickle.dump(Palabras, open('words.pkl', 'wb'))
pickle.dump(Clases, open('classes.pkl', 'wb'))

Entrenamiento = []
SalidaVacia = [0] * len(Clases)

for Documento in Documentos:
    Bolsa = []
    PatronPalabras = Documento[0]
    PatronPalabras = [Lematizador.lemmatize(Palabra.lower()) for Palabra in PatronPalabras]

    for Palabra in Palabras:
        Bolsa.append(1 if Palabra in PatronPalabras else 0)
        
    FilaSalida = list(SalidaVacia)
    FilaSalida[Clases.index(Documento[1])] = 1

    Entrenamiento.append([Bolsa, FilaSalida])

random.shuffle(Entrenamiento)

EntrenamientoX = np.array([Fila[0] for Fila in Entrenamiento])
EntrenamientoY = np.array([Fila[1] for Fila in Entrenamiento])

# Construcción del modelo
Modelo = Sequential()
Modelo.add(Dense(128, input_shape=(len(EntrenamientoX[0]),), activation='relu'))
Modelo.add(Dropout(0.5))
Modelo.add(Dense(64, activation='relu'))
Modelo.add(Dropout(0.5))
Modelo.add(Dense(len(EntrenamientoY[0]), activation='softmax'))

# Configura el optimizador con tasa de aprendizaje decreciente
TasaAprendizaje = ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=10000,
    decay_rate=0.9)

OptimizadorSGD = SGD(learning_rate=TasaAprendizaje, momentum=0.9, nesterov=True)
Modelo.compile(loss='categorical_crossentropy', optimizer=OptimizadorSGD, metrics=['accuracy'])

# Entrenamiento del modelo
Historial = Modelo.fit(EntrenamientoX, EntrenamientoY, epochs=200, batch_size=5, verbose=1)

# Guarda el modelo entrenado
Modelo.save('chatbot_model.keras')

print("Modelo creado")
