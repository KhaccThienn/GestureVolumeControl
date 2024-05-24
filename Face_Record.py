import cv2

haar = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")

picture = cv2.imread("./family.jpg")

picture_gray = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)


faces = haar.detectMultiScale(
    picture_gray,
    scaleFactor=1.1,
    minNeighbors=15,
    minSize=(30, 30),
    flags=cv2.CASCADE_DO_ROUGH_SEARCH
)

for(x, y, w, h) in faces:
    cv2.rectangle(picture, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
img_resized = cv2.resize(picture, None, fx=0.5, fy=0.5)

cv2.imshow("Images", img_resized)
cv2.waitKey(0)