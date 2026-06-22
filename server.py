import os
import json
import io
from flask import Flask, request, Response
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment

app = Flask(__name__)

# Путь к модели Vosk
MODEL_PATH = "models/vosk-model-ru-0.42"
if not os.path.exists(MODEL_PATH):
    print(f"Ошибка: Папка '{MODEL_PATH}' не найдена! Скачайте модель и переименуйте в 'model'.")
    exit(1)

print("Загрузка модели Vosk в память...")
model = Model(MODEL_PATH)
print("Модель успешно загружена!")

@app.route('/inference', methods=['POST'])
def inference():
    # 1. Проверяем наличие файла в запросе
    if 'file' not in request.files:
        return Response("Ошибка: Поле 'file' отсутствует в запросе", status=400, mimetype='text/plain')
    
    file = request.files['file']
    if file.filename == '':
        return Response("Ошибка: Файл не выбран", status=400, mimetype='text/plain')

    # Чтение дополнительных параметров из вашей команды curl (опционально)
    temperature = request.form.get('temperature', '0.0')
    condition_on_prev = request.form.get('condition_on_previous_text', 'false')
    response_format = request.form.get('response_format', 'text')

    try:
        # 2. Читаем бинарные данные файла из памяти
        audio_bytes = file.read()
        
        # 3. Конвертируем аудио в формат, пригодный для Vosk (16кГц, Моно, PCM 16-bit)
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        
        # 4. Распознавание
        recognizer = KaldiRecognizer(model, 16000)
        recognizer.AcceptWaveform(audio.raw_data)
        
        # Получаем финальный текст
        result_json = json.loads(recognizer.FinalResult())
        recognized_text = result_json.get("text", "")
        
        # 5. Возвращаем чистый текст, как ожидает response_format="text"
        return Response(recognized_text, status=200, mimetype='text/plain; charset=utf-8')

    except Exception as e:
        return Response(f"Ошибка обработки аудио: {str(e)}", status=500, mimetype='text/plain')

if __name__ == '__main__':
    # Запуск на порту 8080, как в вашем шаблоне команды
    app.run(host='0.0.0.0', port=8080, debug=False)
