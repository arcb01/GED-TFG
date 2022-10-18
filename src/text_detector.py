from unittest import result
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import pprint
import cv2
import shutil
import os
import pytesseract


def read_data(path_json):
    with open(path_json, encoding='UTF-8') as json_file:
        data = json.load(json_file)
        pprint.pprint(data[0])
        non_answble_imgs_labels = [d["image"] 
                                    for d in data if d["answerable"] == 0]
    return non_answble_imgs_labels


def find_text(path_json, destination):
    results = []
    non_answble_imgs_labels = read_data(path_json)

    for im in non_answble_imgs_labels[:5]:
        img = cv2.imread(destination + im)

        # Preprocessing the image starts
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Performing OTSU threshold
        ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

        # Specify structure shape and kernel size.
        # Kernel size increases or decreases the area
        # of the rectangle to be detected.
        # A smaller value like (10, 10) will detect
        # each word instead of a sentence.
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

        # Applying dilation on the threshold image
        dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)

        # Finding contours
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                                            cv2.CHAIN_APPROX_NONE)
        # Creating a copy of image
        im2 = img.copy()

        # A text file is created and flushed
        file = open("recognized.txt", "w+")
        file.write("")
        file.close()

        # Looping through the identified contours
        # Then rectangular part is cropped and passed on
        # to pytesseract for extracting text from it
        # Extracted text is then written into the text file
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            # Drawing a rectangle on copied image
            rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Cropping the text block for giving input to OCR
            cropped = im2[y:y + h, x:x + w]

            # Open the file in append mode
            file = open("recognized.txt", "a")

            # Apply OCR on the cropped image
            text = pytesseract.image_to_string(cropped)

            # Appending the text into file
            file.write(text)
            file.write("\n")

            # Close the file
            file.close

        if os.path.getsize("./recognized.txt") > 0:
            results.append(True)
        else:
            results.append(False)

    return np.array(results, dtype=bool)


if __name__ == "__main__":
    path_json = '../Annotations/train.json'
    destination = '../data/NAI_train/'

    results = find_text(path_json, destination)

    # Count how many True values (images containing text)
    count = np.count_nonzero(results)
    perct_of_text = round(count / results.shape[0] * 100, 2)

    print(f"Total number of imgs = {results.shape[0]}\nTotal percentage of imgs containing text = {perct_of_text}%")