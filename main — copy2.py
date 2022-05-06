# Были импортированы библиотеки
# tensorflow==2.4.0   tensorflow-gpu==2.4.0     keras==2.4.3
# numpy==1.19.3     pillow==7.0.0   scipy==1.4.1    h5py==2.10.0
# matplotlib==3.3.2     opencv-python   keras-resnet==0.2.0

from imageai.Detection import VideoObjectDetection
import os
import cv2

# VideoFile1 = "/images/99 (2021-10-02 10'00'00 - 2021-10-02 10'30'00).avi"
VideoFile2 = "images/100 (22-05-06).mp4"
# Photo = "1.jpg"
cap = cv2.VideoCapture(0)   # звхват видео с веб-камеры (0 - выход вебки, VideoFile - переменная с расположением на файл
exec_path = os.getcwd()     # переменная с корневой папкой проекта

detector = VideoObjectDetection()   # метод трекинга
detector.setModelTypeAsRetinaNet()      # шаблон
detector.setModelPath(os.path.join(exec_path, 'resnet50_coco_best_v2.1.0.h5'))     # путь к файлу шаблона
detector.loadModel()

while cap.isOpened():
    _, frame = cap.read()

    if cv2.waitKey(1) & 0xFF == ord('q'):       # Кнопка выхода из цикла
        break
    video_path = detector.detectObjectsFromVideo(camera_input=cap,
                                                 output_file_path='/images/trainDetected',
                                                 frames_per_second=20,
                                                 log_progress=True,
                                                 detection_timeout=120)
                                              #  input_file_path='/images/100 (22-05-06).mp4'    *(обработка видофайла)*
    cv2.imshow('Result', video_path)   # не работает нифига
    print(video_path)


#    for eachObject in detections:
#        print(eachObject["name"], eachObject["percentage_probability"], eachObject["box_points"])
#        print("-------------------------------------")
