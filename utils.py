import cv2 as cv

# colors
# value = (blue, green, red)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (255, 0, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
PURPLE = (128, 0, 128)
PINK = (147, 20, 255)
points_list = [(200, 300), (150, 150), (400, 200)]


def colorBackgroundText(img, text, font, fontScale, textPos, textThickness=1, textColor=(0, 255, 0), bgColor=(0, 0, 0),
                        pad_x=3, pad_y=3):
    (t_w, t_h), _ = cv.getTextSize(text, font, fontScale, textThickness)  # getting the text size
    x, y = textPos
    cv.rectangle(img, (x - pad_x, y + pad_y), (x + t_w + pad_x, y - t_h - pad_y), bgColor, -1)  # draw rectangle
    cv.putText(img, text, textPos, font, fontScale, textColor, textThickness)  # draw in text

    return img


def textWithBackground(img, text, font, fontScale, textPos, textThickness=1, textColor=(0, 255, 0), bgColor=(0, 0, 0),
                       pad_x=3, pad_y=3, bgOpacity=0.5):
    (t_w, t_h), _ = cv.getTextSize(text, font, fontScale, textThickness)  # getting the text size
    x, y = textPos
    overlay = img.copy()  # coping the image
    cv.rectangle(overlay, (x - pad_x, y + pad_y), (x + t_w + pad_x, y - t_h - pad_y), bgColor, -1)  # draw rectangle
    new_img = cv.addWeighted(overlay, bgOpacity, img, 1 - bgOpacity, 0)  # overlaying the rectangle on the image.
    cv.putText(new_img, text, textPos, font, fontScale, textColor, textThickness)  # draw in text
    img = new_img
    return img

