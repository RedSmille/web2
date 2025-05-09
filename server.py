import http.server
import socketserver
import json
import os
import pickle
import numpy as np
import nltk
import unicodedata
from urllib.parse import urlparse, parse_qs
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from datetime import datetime
from respuestas_chatbot import ObtenerRespuesta
import locale

# Intentar establecer localización en español
try:
    locale.setlocale(locale.LC_TIME, 'es_ES.UTF-8')
except locale.Error:
    print("La configuración regional 'es_ES.UTF-8' no está disponible. Usando la predeterminada.")

# Inicialización del chatbot
Lematizador = WordNetLemmatizer()

try:
    with open('Informacion.json', 'r', encoding='utf-8') as archivo:
        Intentos = json.load(archivo)
except Exception as e:
    print(f"❌ Error cargando 'Informacion.json': {e}")
    Intentos = {}

try:
    Palabras = pickle.load(open('words.pkl', 'rb'))
    Clases = pickle.load(open('classes.pkl', 'rb'))
    Modelo = load_model('chatbot_model.keras')
except Exception as e:
    print(f"❌ Error cargando archivos del modelo: {e}")
    exit(1)

# Función para eliminar acentos y convertir a minúsculas
def NormalizarTexto(Texto):
    Texto = Texto.lower()
    Texto = ''.join(c for c in unicodedata.normalize('NFKD', Texto) if unicodedata.category(c) != 'Mn')
    return Texto

# Función para limpiar el texto del usuario
def LimpiarOracion(Oracion):
    Oracion = NormalizarTexto(Oracion)
    PalabrasOracion = nltk.word_tokenize(Oracion)
    PalabrasOracion = [Lematizador.lemmatize(Palabra) for Palabra in PalabrasOracion]
    return PalabrasOracion

# Convierte la oración en una "bolsa de palabras"
def BolsaDePalabras(Oracion):
    PalabrasOracion = LimpiarOracion(Oracion)
    Bolsa = [0] * len(Palabras)
    for Palabra in PalabrasOracion:
        for i, PalabraLista in enumerate(Palabras):
            if PalabraLista == Palabra:
                Bolsa[i] = 1
    return np.array(Bolsa)

# Predice la intención del usuario
def PredecirIntencion(Oracion):
    Bolsa = BolsaDePalabras(Oracion)
    ResultadoModelo = Modelo.predict(np.array([Bolsa]))[0]
    UMBRAL_ERROR = 0.25
    Resultados = [[i, Probabilidad] for i, Probabilidad in enumerate(ResultadoModelo) if Probabilidad > UMBRAL_ERROR]
    Resultados.sort(key=lambda x: x[1], reverse=True)
    return [{'Intencion': Clases[Resultado[0]], 'Probabilidad': str(Resultado[1])} for Resultado in Resultados]

# Configuración del servidor HTTP
PUERTO = 8000

class ManejadorChatbot(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.path = '/index.html'

        if os.path.exists(self.path[1:]):
            return super().do_GET()
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Archivo no encontrado')

    def do_POST(self):
        try:
            LongitudContenido = int(self.headers.get('Content-Length', 0))
            DatosPost = self.rfile.read(LongitudContenido)

            print("📥 Datos recibidos (raw):", DatosPost)

            Datos = json.loads(DatosPost.decode('utf-8'))
            print("📥 Datos decodificados (JSON):", Datos)

            Pregunta = Datos.get('prompt', '').strip()
            print("❓ Pregunta recibida:", Pregunta)

            if not Pregunta:
                Respuesta = {"response": ["Por favor, ingresa un mensaje o pregunta."]}
            else:
                IntentosDetectados = PredecirIntencion(Pregunta)
                print("🔎 Intenciones detectadas:", IntentosDetectados)

                TextoRespuesta = ObtenerRespuesta(IntentosDetectados, Intentos)
                print("💬 Respuesta generada:", TextoRespuesta)

                Respuesta = {"response": TextoRespuesta}

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(Respuesta, ensure_ascii=False).encode('utf-8'))

        except Exception as e:
            print("❌ Error en do_POST:", str(e))
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            ErrorMensaje = json.dumps({"error": str(e)}, ensure_ascii=False).encode('utf-8')
            self.wfile.write(ErrorMensaje)

# Inicia el servidor
with socketserver.ThreadingTCPServer(('0.0.0.0', PUERTO), ManejadorChatbot) as httpd:
    print(f'🚀 Servidor ejecutándose en: http://localhost:{PUERTO}/')
    httpd.serve_forever()
