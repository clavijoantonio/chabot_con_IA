import kagglehub
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, pipeline
from datasets import Dataset
import torch
import os
from flask import Flask, request
from flask_socketio import SocketIO, emit

# Configuración de Flask y SocketIO
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Variables globales para el modelo (se cargarán al iniciar)
model = None
tokenizer = None
translator = None
translator2 = None

# 2. Descargar el dataset desde Kaggle
def descargar_dataset():
    print("Descargando dataset...")
    path = kagglehub.dataset_download("kreeshrajani/3k-conversations-dataset-for-chatbot")
    print(f"Dataset descargado en: {path}")
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".csv"):
                csv_path = os.path.join(root, file)
                print(f"Archivo CSV encontrado: {csv_path}")
                return csv_path
    raise FileNotFoundError("No se encontró ningún archivo CSV en el dataset descargado.")

# 3. Cargar y preprocesar el dataset
def cargar_y_preprocesar_dataset(csv_path):
    print("Cargando y preprocesando dataset...")
    df = pd.read_csv(csv_path)
    print("Primeras filas del dataset:")
    print(df.head())

    if 'question' not in df.columns or 'answer' not in df.columns:
        raise ValueError("El dataset debe contener las columnas 'question' y 'answer'.")

    df['text'] = df['question'] + " [SEP] " + df['answer']
    print("Dataset cargado y preprocesado.")
    return df

# 4. Cargar el modelo y tokenizador
def cargar_modelo_y_tokenizador():
    print("Cargando modelo y tokenizador...")
    model_name = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(model_name)
    translator = pipeline("translation_es_to_en", model="Helsinki-NLP/opus-mt-es-en")
    translator2 = pipeline("translation_en_to_es", model="Helsinki-NLP/opus-mt-en-es")
    print("Modelo y tokenizador cargados.")
    return model, tokenizer, translator, translator2

# 5. Fine-tuning del modelo
def fine_tuning_modelo(model, tokenizer, df):
    print("Preparando fine-tuning...")
    dataset = Dataset.from_pandas(df)
    dataset = dataset.train_test_split(test_size=0.2)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# Función para generar respuestas del chatbot
def generate_response(user_input):
    global model, tokenizer, translator, translator2
    
    try:
        # Traducir la entrada del usuario a inglés
        examples = """
               Usuario: Hola, ¿cómo estás?
               Chatbot: ¡Hola! Estoy bien, ¿y tú?

               Usuario: ¿Qué haces?
               Chatbot: Estoy aquí para ayudarte. ¿En qué puedo colaborarte?
               """
        prompt = f"{examples}\nUsuario: {user_input}\nChatbot:"
        translated_input = translator(user_input, max_length=400)[0]["translation_text"]
        print("Input traducido:", translated_input)

        # Tokenizar la entrada traducida
        inputs = tokenizer(
            translated_input + tokenizer.eos_token,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        # Generar una respuesta en inglés
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=512,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            temperature=0.5,
            top_k=50,
        )

        # Decodificar la respuesta en inglés
        english_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Traducir la respuesta al español
        translated_response = translator2(english_response, max_length=400)[0]["translation_text"]
        print("Respuesta generada:", translated_response)
        return translated_response
    
    except Exception as e:
        print(f"Error al generar respuesta: {e}")
        return "Lo siento, ocurrió un error al procesar tu mensaje."

# Manejo de conexiones SocketIO
@socketio.on('connect')
def handle_connect():
    print('Cliente conectado:', request.sid)

@socketio.on('disconnect')
def handle_disconnect():
    print('Cliente desconectado:', request.sid)

@socketio.on('message')
def handle_message(data):
    print('Mensaje recibido desde el cliente:', data)
    if isinstance(data, dict) and 'text' in data:
        user_input = data['text']
        response = generate_response(user_input)
        emit('response', {'text': response})
    else:
        emit('error', {'text': 'Formato de mensaje inválido'})

# 8. Función principal
def main():
    global model, tokenizer, translator, translator2
    
    try:
        # Descargar el dataset
        path = descargar_dataset()

        # Cargar y preprocesar el dataset
        df = cargar_y_preprocesar_dataset(path)

        # Cargar el modelo y tokenizador
        model, tokenizer, translator, translator2 = cargar_modelo_y_tokenizador()

        # Fine-tuning (opcional)
        fine_tuning_modelo(model, tokenizer, df)

        # Iniciar el servidor SocketIO
        print("Servidor SocketIO iniciado. Esperando conexiones...")
        socketio.run(app, host='0.0.0.0', port=5000, debug=True)
        
    except Exception as e:
        print(f"Error en la función principal: {e}")

if __name__ == "__main__":
    main()