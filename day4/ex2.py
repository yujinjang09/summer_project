import cv2


cap = cv2.VideoCapture("ex2.mp4")
if cap.isOpened():
  ret, img = cap.read()
  cv2.imwrite("ex2.jpg", img)

cap.release()