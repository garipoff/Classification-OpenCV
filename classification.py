# python classification.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
from threading import Thread
import pygame
# Указываем путь к своему Google API
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/home/grf/Project.json"
from google.cloud import texttospeech


last = 0
# Воспроизводим аудио файл
pygame.mixer.init()
# Генерируем речь из текста
client = texttospeech.TextToSpeechClient()
# Отоброжаем кирилицу в OpenCV
ft = cv2.freetype.createFreeType2()
ft.loadFontData(fontFileName='times.ttf', id=0) 

# Загружаем обученную модель Caffe
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="путь до разворачиваемого файла prototxt")
ap.add_argument("-m", "--model", required=True,
	help="путь до предварительно обученной Caffe модели")
ap.add_argument("-c", "--assurance", type=float, default=0.2,
	help="минимальная вероятность фильтрации слабо классифицированных объектов")
args = vars(ap.parse_args())
# Список классифицируемых объектов
CLASSES = ["фон", "самолет", "велосипед", "птица", 
"лодка", "бутылка", "автобус", "машина", "кошка", "стул", "корова", "обеденный стол",
"собака", "лошадь", "мотоцикл", "человек", "растение в горшке", "овца",
"диван", "поезд", "монитор"]

print("[INFO] Загрузка модели...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

print("[INFO] Запуск видеопотока...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

print("[INFO] Для выхода нажмите Q")

while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=600)
    # Преобразовываем полученный с камеры кадр
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)

	net.setInput(blob)
	classified = net.forward()
    # Производим классификация кадра
	for i in np.arange(0, classified.shape[2]):
		assurance = classified[0, 0, i, 2]
		id = int(classified[0, 0, i, 1])

		if assurance > 0.85 and not pygame.mixer.music.get_busy() and last>5:
			last = 0
			synthesis_input = texttospeech.types.SynthesisInput(text=CLASSES[id])
			voice = texttospeech.types.VoiceSelectionParams(
				language_code = 'ru-RU',
				ssml_gender = texttospeech.enums.SsmlVoiceGender.FEMALE)

			# Выбираем тип аудиофайла, который хотим вернуть от сервера
			audio_config = texttospeech.types.AudioConfig(
				audio_encoding = texttospeech.enums.AudioEncoding.MP3)

			# Выполняем запрос преобразования текста в речь при вводе текста с выбранными
            # параметрами голоса и типом аудио файла
			response = client.synthesize_speech(synthesis_input, voice, audio_config)
			with open('output.mp3', 'wb') as out:
				# Сохраняем полученный файл
				out.write(response.audio_content)			
			# Открываем полученный файл и воспроизводим
			name = 'output.mp3'
			pygame.mixer.music.load(name)
			pygame.mixer.music.play()
		if assurance > 0.85 and last < 5:
            # Выводим название классифицированного объекта и процент точности
			label = "{}: {:.2f}%".format(CLASSES[id], assurance * 100)
			frame = ft.putText(img=frame,
				text=label,
				org=(10, 440),
				fontHeight=60,
				color=(255, 255, 255),
				thickness=-1,
				line_type=cv2.LINE_AA,
				bottomLeftOrigin=True)
	cv2.imshow("Classification", frame)
	last += 1
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break

print("[INFO] Программа закрыта.")

cv2.destroyAllWindows()
vs.stop()