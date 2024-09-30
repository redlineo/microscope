import cv2
import pytesseract
import os
import re

pytesseract.pytesseract.tesseract_cmd = './Tesseract-OCR/tesseract.exe'


def show_img(img):
    cv2.imwrite('color_img.jpg', img)
    cv2.imshow("image", img)
    cv2.waitKey()


def prepair_img(img, crop_coef=0):
    img = img[crop_coef:, :]
    show_img(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    show_img(thresh1)
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours


def recognize_and_crop(contours, img, debug=0):
    im2 = img.copy()
    height = im2.shape[0]
    width = im2.shape[1]
    file = open("recognized.txt", "w+")
    file.write("")
    file.close()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h >= int(0.05 * height) and w >= int(0.8 * width):
        # if True:
            rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cropped = im2[y:y + h, x:x + w]
            if debug == 1:
                show_img(cropped)
            file = open("recognized.txt", "a")
            text = pytesseract.image_to_string(cropped)
            if text != "":
                file.write(text)
                file.write("\n")
            file.close()
    return 1


def main():
    directory = os.fsencode('./dataset/')
    for file in os.listdir(directory):
        filename = os.fsdecode(directory + file)
        img = cv2.imread(filename)
        contours = prepair_img(img, 0)
        recognize_and_crop(contours, img, 1)


if __name__ == "__main__":
    main()
