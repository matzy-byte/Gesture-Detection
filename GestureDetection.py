import cv2

TITLE_CAPTURE = "Capture"
CAP = cv2.VideoCapture(0)

while True:
    ret, image = CAP.read()
    if ret:
        cv2.namedWindow(TITLE_CAPTURE, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(TITLE_CAPTURE, image)

    key = cv2.waitKey(10)
    if key == ord("q"):
        break

cv2.destroyAllWindows()