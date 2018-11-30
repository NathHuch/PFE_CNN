from PIL import Image


def display_PIL(nparray):
    image = Image.fromarray(nparray, 'L')
    image.show()

