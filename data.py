import numpy as np
from os import listdir
from os.path import isfile, join
from display import display_PIL
import matplotlib.pyplot as plt
from random import shuffle

class Data:

    def __init__(self, load=False):
        if load:
            print('Loading data ...')
            lysandre_data = np.loadtxt('Matrice_grandes_images/' + 'Lysandre' + '.txt')
            nathan_data = np.loadtxt('Matrice_grandes_images/' + 'Nathan' + '.txt')
            sam_data = np.loadtxt('Matrice_grandes_images/' + 'Sam' + '.txt')
            morgane_data = np.loadtxt('Matrice_grandes_images/' + 'Morgane' + '.txt')

            self.data = {
                "Lysandre": lysandre_data,
                "Nathan": nathan_data,
                "Sam": sam_data,
                "Morgane": morgane_data
            }
            print('Data loaded !')

    @staticmethod
    def convert():
        onlyfiles = [f for f in listdir("Matrice_images") if isfile(join("Matrice_images",f))]
        lysandre = list(filter(lambda file_name: "Lysandre" in file_name, listdir('Matrice_images/Lysandre')))
        nathan = list(filter(lambda file_name: "Nathan" in file_name, listdir('Matrice_images/Nathan')))
        sam = list(filter(lambda file_name: "Sam" in file_name, listdir('Matrice_images/Samuel')))
        morgane = list(filter(lambda file_name: "Morgane" in file_name, listdir('Matrice_images/Morgane')))

        nathan_data = np.loadtxt("Matrice_images/Nathan/" + nathan[0])

        for file_name in range(1, len(nathan)):
            nathan_data = np.c_[nathan_data, np.loadtxt("Matrice_images/Nathan/" + nathan[file_name])]
            print("Nathan:", file_name, '/', len(nathan))

        np.savetxt("Matrice_grandes_images/Nathan.txt", nathan_data)

        sam_data = np.loadtxt("Matrice_images/Samuel/" + sam[0])

        for file_name in range(1, len(sam)):
            sam_data = np.c_[sam_data, np.loadtxt("Matrice_images/Samuel/" + sam[file_name])]
            print("Sam:", file_name, '/', len(sam))

        np.savetxt("Matrice_grandes_images/Sam.txt", sam_data)

        morgane_data = np.loadtxt("Matrice_images/Morgane/" + morgane[0])

        for file_name in range(1, len(morgane)):
            morgane_data = np.c_[morgane_data, np.loadtxt("Matrice_images/Morgane/" + morgane[file_name])]
            print("Morgane:", file_name, '/', len(morgane))

        np.savetxt("Matrice_grandes_images/Morgane.txt", morgane_data)

        lysandre_data = np.loadtxt("Matrice_images/Lysandre/" + lysandre[0])

        for file_name in range(1, len(lysandre)):
            lysandre_data = np.c_[lysandre_data, np.loadtxt("Matrice_images/Lysandre/" + lysandre[file_name])]
            print("Lysandre:", file_name, '/', len(lysandre))

        np.savetxt("Matrice_grandes_images/Lysandre.txt", lysandre_data)

    def fetch(self, name, width=512):
        print('Fetching', name, 'data')
        try:
            data = self.data[name]
            print('Correctly fetched data from dictionary.')
        except:
            print('Data not found in dictionary. Loading ...')
            data = np.loadtxt('Matrice_grandes_images/' + name + '.txt')

            if not hasattr(self, 'data'):
                self.data = {}

            self.data[name] = data

            print('Data loaded !')

        nb_batches = int(np.floor(data.shape[1] / width))
        data = data[:, 0:int(nb_batches * width)]
        data_batches = [data[:, i:i + width] for i in range(nb_batches)]
        return data_batches

    def training(self, width=512):
        lysandre = [[x, [1, 0, 0, 0]] for x in self.fetch("Lysandre", width=width)]
        nathan = [[x, [0, 1, 0, 0]] for x in self.fetch("Nathan", width=width)]
        samuel = [[x, [0, 0, 1, 0]] for x in self.fetch("Sam", width=width)]
        morgane = [[x, [0, 0, 0, 1]] for x in self.fetch("Morgane", width=width)]

        dataset = lysandre + nathan + samuel + morgane
        shuffle(dataset)

        x = list(map(lambda elem: elem[0], dataset))
        y = list(map(lambda elem: elem[1], dataset))

        x_train = x[0:int(len(x) * 8/10)]
        x_test = x[int(len(x) * 8/10)+1:]

        y_train = y[0:int(len(x) * 8/10)]
        y_test = y[int(len(x) * 8/10)+1:]

        return (np.expand_dims(np.array(x_train), axis=3), np.array(y_train)), (np.expand_dims(np.array(x_test), axis=3), np.array(y_test))


data = Data(True)
morgane = data.training()
