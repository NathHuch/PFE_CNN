from PIL import Image
import matplotlib.pyplot as plt

def display_PIL(nparray):
    img = plt.imshow(nparray, origin='lower', cmap='jet', interpolation='nearest', aspect='auto')
    plt.show()

