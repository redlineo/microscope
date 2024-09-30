import cv2
import pytesseract
import os
import re

pytesseract.pytesseract.tesseract_cmd = './Tesseract-OCR/tesseract.exe'


def show_img(img):
    cv2.imwrite('color_img.jpg', img)
    cv2.imshow("image", img)
    cv2.waitKey()


def prepair_img(img, crop_coef=0, debug=0):
    img = img[crop_coef:, :]
    if debug == 1:
        show_img(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV)
    # thresh1 = cv2.inRange(gray, 254, 255)
    if debug == 1:
        show_img(thresh1)
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours


def recognize_and_crop(contours, img, debug=0):
    im2 = img.copy()
    height = im2.shape[0]
    width = im2.shape[1]
    recognized_text = ""
    cropped = im2
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h >= int(0.05 * height) and w >= int(1.0 * width):
            # if True:
            rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cropped = im2[y:y + h, x:x + w]
            if debug == 1:
                show_img(cropped)
            text = pytesseract.image_to_string(cropped)
            if text != "":
                recognized_text += text
    if debug == 1:
        print(recognized_text)
    if recognized_text == "":
        exit(1)
    scale_x = re.findall(r'x\d+.\d{0,3}', recognized_text)
    if debug == 1:
        print(scale_x)
    if len(scale_x) == 0:
        exit(1)
    scale_x = scale_x[0]
    length_nm = re.findall(r'\d+.\d{0,3}nm', recognized_text)
    if debug == 1:
        print(length_nm)
    if len(length_nm) == 0:
        length_um = re.findall(r'\d+.\d{0,3}[up]m', recognized_text)
        if debug == 1:
            print(length_um)
        if len(length_um) == 0:
            exit(1)
        length_um = length_um[0][:-2]
        length_nm = float(length_um) * 1000
    else:
        length_nm = length_nm[0][:-2]
    return scale_x, int(length_nm), cropped


def get_nm_of_pixel(scale_x, length_nm, cropped_image):
    return 1


def main():
    # directory = os.fsencode('./dataset/')
    directory = os.fsencode('./dataset2/')
    for file in os.listdir(directory):
        filename = os.fsdecode(directory + file)
        img = cv2.imread(filename)
        crop_coef = 800
        debug = 1
        contours = prepair_img(img, crop_coef, debug)
        scale_x, length_nm, cropped_image = recognize_and_crop(contours, img[crop_coef:, :], debug)
        pixel_nm = get_nm_of_pixel(scale_x, length_nm, cropped_image)
        print(file,scale_x, length_nm, pixel_nm, end=' ')
        print(end='\n')


if __name__ == "__main__":
    main()
