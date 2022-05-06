# Были импортированы библиотеки
# tensorflow==2.4.0   tensorflow-gpu==2.4.0     keras==2.4.3
# numpy==1.19.3     pillow==7.0.0   scipy==1.4.1    h5py==2.10.0
# matplotlib==3.3.2     opencv-python   keras-resnet==0.2.0
#
#

from imageai.Detection import ObjectDetection
import cv2

VideoFile1 = "/images/99 (2021-10-02 10'00'00 - 2021-10-02 10'30'00).avi"
VideoFile2 = "images/100 (22-05-06).mp4"
Photo = ""
cap = cv2.VideoCapture(VideoFile2)

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath('yolo-tiny.h5') # шаблон объектов для нейронки https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo.h5
detector.loadModel()

while cap.isOpened():
    ret, frame = cap.read()
    cv2.imshow('Result', frame)
    if cv2.waitKey(25) % 0xFF == ord('q'):
        break
    detector.detectObjectsFromImage(input_image=frame,
                                    input_type='array',
                                    output_type="array",
                                    output_image_path='images/newObj.jpg')

    print(detector)
