import cv2

cap = cv2.VideoCapture('images/101rzd.mp4')

#   накладываем маску ЧБ
obj_detect = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=60)

while True:
    _, frame = cap.read()

    # проверяем размеры окна
    height, weight, _ = frame.shape
    print(height, weight)

    # выбираем область интереса
    roi = frame[300:720, 0:600]

    # трекинг
    mask = obj_detect.apply(roi)
    #mask = obj_detect.apply(frame)
    contours, _ =cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        #   считаем область и удалаем мелкие элементы
        area = cv2.contourArea(cnt)
        if area > 100:
            #cv2.drawContours(roi, [cnt], -1, (0, 255, 0))
            x, y, h, w = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)


    #cv2.imshow("Frame", frame)
    #cv2.imshow("Mask", mask)
    cv2.imshow("Roi", roi)

    cv2.waitKey(5)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyWindow()