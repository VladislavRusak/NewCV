# Были импортированы библиотеки
# tensorflow==2.4.0   tensorflow-gpu==2.4.0     keras==2.4.3
# numpy==1.19.3     pillow==7.0.0   scipy==1.4.1    h5py==2.10.0
# matplotlib==3.3.2     opencv-python   keras-resnet==0.2.0

from imageai.Detection import ObjectDetection
import os

# VideoFile1 = "/images/99 (2021-10-02 10'00'00 - 2021-10-02 10'30'00).avi"
# VideoFile2 = "images/100 (22-05-06).mp4"
# Photo = "1.jpg"
# cap = cv2.(Photo)
exec_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(exec_path, 'resnet50_coco_best_v2.1.0.h5'))
detector.loadModel()

# while cap.isOpened():
#    _, frame = cap.read()
#    cv2.imshow('Result', frame)
#    if cv2.waitKey(25) & 0xFF == ord('q'):
#        break
detections = detector.detectObjectsFromImage(input_image=os.path.join(exec_path, 'image2.jpg'),
                                             output_image_path=os.path.join(exec_path, 'image2new.jpg'),
                                             minimum_percentage_probability=30)
for eachObject in detections:
    print(eachObject["name"], " : ", eachObject["percentage_probability=30"], " : ", eachObject["box_points"])
    print("-------------------------------------")
