import os
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import keyboard
from openai import OpenAI
import pyttsx3

# Asegúrate de actualizar estas configuraciones según tu entorno y necesidades
ffmpeg_path = 'C:\\ProgramData\\chocolatey\\bin'
os.environ['PATH'] += os.pathsep + ffmpeg_path
fs = 44100  # Frecuencia de muestreo
duration = 10  # Duración máxima de la grabación en segundos
filename = 'output.wav'  # Nombre del archivo de salida

# Carga el modelo de Whisper
model = whisper.load_model("small")

# Configura el cliente de OpenAI (ajusta según tu configuración de LM Studio)
client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

def record_audio():
    print("Iniciando grabación... presiona 's' para detener.")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=2, dtype='float64')
    sd.wait()  # Espera hasta que la grabación termine o se detenga
    write(filename, fs, recording)  # Guarda el archivo
    print(f"Grabación guardada como {filename}")

def transcribe_audio(file_path):
    result = model.transcribe(file_path)
    return result["text"]



def ask_lm_studio(question):
    completion = client.chat.completions.create(
        model="local-model",
        messages=[
            {"role": "system", "content": "respond only in spanish."},
            {"role": "user", "content": question}
        ],
        temperature=0.7,
    )
    # Accede correctamente a la propiedad 'content' del objeto 'ChatCompletionMessage'
    return completion.choices[0].message.content



def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def main_loop():
    while True:
        print("\nPresiona 'r' para comenzar a grabar.")
        keyboard.wait('r')
        record_audio()
        
        print("\nTranscribiendo audio...")
        transcribed_text = transcribe_audio(filename)
        print("Texto transcrito:", transcribed_text)
        
        print("\nObteniendo respuesta de LM Studio...")
        response = ask_lm_studio(transcribed_text)
        print("Respuesta de LM Studio:", response)
        
        # Leer la respuesta con voz
        print("\nLeyendo respuesta...")
        speak(response)
        
        print("\nPresiona 'c' para hacer otra pregunta o 'q' para salir.")
        key = keyboard.read_key()
        if key == 'q':
            print("Saliendo...")
            break

if __name__ == "__main__":
    main_loop()
