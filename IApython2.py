import kagglehub
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, pipeline
from datasets import Dataset
import torch
import os
from flask import Flask, request
from flask_socketio import SocketIO, emit
from sklearn.model_selection import train_test_split

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

    # Preprocesamiento: combinar preguntas y respuestas con un token especial
    df['text'] = df['question'] + " [SEP] " + df['answer']
    
    # Dividir en train y validation
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    print(f"Dataset dividido: {len(train_df)} entrenamiento, {len(val_df)} validación")
    
    return train_df, val_df

# 4. Cargar el modelo y tokenizador
def cargar_modelo_y_tokenizador():
    print("Cargando modelo y tokenizador...")
    model_name = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Configurar token especial para separación
    if tokenizer.sep_token is None:
        tokenizer.add_special_tokens({'sep_token': '[SEP]'})
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Ajustar el modelo para el nuevo token
    model.resize_token_embeddings(len(tokenizer))
    
    translator = pipeline("translation_es_to_en", model="Helsinki-NLP/opus-mt-es-en")
    translator2 = pipeline("translation_en_to_es", model="Helsinki-NLP/opus-mt-en-es")
    print("Modelo y tokenizador cargados.")
    return model, tokenizer, translator, translator2

# 5. Fine-tuning del modelo optimizado para AWS CPU
def fine_tuning_modelo(model, tokenizer, train_df, val_df):
    print("Preparando fine-tuning del modelo...")
    
    # Convertir DataFrames a datasets de HuggingFace
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    # Tokenizar los datos con procesamiento por lotes optimizado
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=128,
            return_overflowing_tokens=False
        )
    
    # Optimización: Usar multiprocesamiento y caché
    train_dataset = train_dataset.map(
        tokenize_function, 
        batched=True,
        batch_size=8,
        num_proc=os.cpu_count() - 1 if os.cpu_count() > 1 else 1,
        cache_file_name="./train_cache.arrow"
    )
    
    val_dataset = val_dataset.map(
        tokenize_function, 
        batched=True,
        batch_size=8,
        num_proc=os.cpu_count() - 1 if os.cpu_count() > 1 else 1,
        cache_file_name="./val_cache.arrow"
    )
    
    # Configurar argumentos de entrenamiento optimizados para CPU en AWS
    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=3,  # Balance entre tiempo y calidad
        per_device_train_batch_size=4,  # Ajustado para CPU AWS
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,  # Para simular batch más grande
        save_steps=10_000,
        save_total_limit=2,
        evaluation_strategy="steps",
        eval_steps=500,  # Evaluación más frecuente para monitoreo
        logging_dir="./logs",
        logging_steps=100,
        load_best_model_at_end=True,
        no_cuda=True,  # Forzar uso de CPU
        fp16=False,    # No usar mixed precision en CPU
        optim="adamw_torch",  # Optimizador eficiente para CPU
        lr_scheduler_type="linear",  # Planificador de tasa de aprendizaje
        warmup_steps=500,  # Calentamiento para mejor convergencia
        weight_decay=0.01,  # Regularización
        report_to="none",  # Deshabilitar reportes externos
        dataloader_num_workers=os.cpu_count() - 1 if os.cpu_count() > 1 else 1,  # Paralelización
        disable_tqdm=False  # Habilitar barra de progreso
    )
    
    # Crear Trainer con configuración optimizada
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    print("Iniciando fine-tuning optimizado para AWS CPU...")
    trainer.train()
    
    # Liberar memoria después del entrenamiento
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print("Fine-tuning completado!")
    return model

# Función para generar respuestas del chatbot optimizada
def generate_response(user_input):
    global model, tokenizer, translator, translator2
    
    try:
        # Ejemplos de contexto para mejor generación
        examples = """
               Usuario: Hola, ¿cómo estás?
               Chatbot: ¡Hola! Estoy bien, ¿y tú?

               Usuario: ¿Qué haces?
               Chatbot: Estoy aquí para ayudarte. ¿En qué puedo colaborarte?
               """
        prompt = f"{examples}\nUsuario: {user_input}\nChatbot:"
        
        # Traducir la entrada del usuario a inglés con manejo de errores
        translated_input = translator(
            user_input, 
            max_length=400,
            truncation=True,
            num_beams=1  # Más rápido que beam search
        )[0]["translation_text"]
        print("Input traducido:", translated_input)

        # Tokenizar con configuración optimizada
        inputs = tokenizer(
            prompt + translated_input + tokenizer.eos_token,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=256,  # Reducido para mejor rendimiento
            return_attention_mask=True
        )

        # Generar respuesta con parámetros optimizados para CPU
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=256,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            temperature=0.7,
            top_k=30,  # Reducido para mejor rendimiento
            top_p=0.9,
            num_beams=1,  # Beam search deshabilitado para CPU
            early_stopping=True,
            no_repeat_ngram_size=2
        )

        # Decodificar la respuesta en inglés
        english_response = tokenizer.decode(
            outputs[0], 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        # Traducir la respuesta al español con configuración optimizada
        translated_response = translator2(
            english_response, 
            max_length=400,
            truncation=True,
            num_beams=1
        )[0]["translation_text"]
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

# 8. Función principal optimizada
def main():
    global model, tokenizer, translator, translator2
    
    try:
        # Configuración inicial para optimizar recursos
        torch.set_num_threads(os.cpu_count())
        print(f"Configurando PyTorch para usar hasta {os.cpu_count()} hilos de CPU")
        
        # Descargar el dataset
        path = descargar_dataset()

        # Cargar y preprocesar el dataset
        train_df, val_df = cargar_y_preprocesar_dataset(path)

        # Cargar el modelo y tokenizador
        model, tokenizer, translator, translator2 = cargar_modelo_y_tokenizador()

        # Fine-tuning con el dataset
        model = fine_tuning_modelo(model, tokenizer, train_df, val_df)

        # Optimizar el modelo para inferencia
        model.eval()
        if hasattr(model, 'prune_heads'):
            model.prune_heads()  # Opcional: podar cabezas de atención no utilizadas

        # Iniciar el servidor SocketIO con configuración optimizada
        print("Servidor SocketIO iniciado. Esperando conexiones...")
        socketio.run(
            app, 
            host='0.0.0.0', 
            port=5000, 
            debug=False,  # Deshabilitar debug en producción
            use_reloader=False,  # Deshabilitar recarga para AWS
            threaded=True,  # Habilitar modo threaded
            allow_unsafe_werkzeug=True  # Para entornos de producción controlados
        )
        
    except Exception as e:
        print(f"Error en la función principal: {e}")

if __name__ == "__main__":
    main()