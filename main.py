########################################################################################################################
#################################################   Librairies   #######################################################
########################################################################################################################
import os, os.path
import numpy as np
import matplotlib.pyplot as plt
import wave
import tkinter as Tk
import math
import json
import pickle
from scipy.io.wavfile import read, write
from tkinter import filedialog
from functions import *
from scipy.signal import stft, istft, correlate

dir = "Donnes_wav"
list = os.listdir(dir)  # dir is your directory path
number_files = len(list)
index = 0
max_dim = 0
min_dim = 431
while index!=number_files:
    os.chdir("C:\\Users\\Nathan\\Documents\\Nathan_Travail\\I3\\PFE\\Python\\Creation_Spectre")
    # On récupère les fichiers audio générés et les métadonnées stockées dans le fichier json


    [fe, data] = read(dir+'\\'+list[index])
    ########################################################################################################################
    ############################################   Sélection des sources ##################################################
    ########################################################################################################################
    # Ouverture du fichier json courant !! Par la suite il faudra fournir à cette fonction le nom du fichier json
    # with open(dir+'\\'+list[index].split('.')[0]+'.json') as json_data:
    #     d = json.load(json_data)
    #     print(d)

    # # On récupère théoriquement le nb de microphones présents dans la scène grâce au nombre de voies
    nb_sources = 1#data.shape[1]
    data_length = data.shape[0]

    # Visualisation des signaux d'entrées
    t = np.arange(0, data_length / fe, 1 / fe)

    # plt.figure()
    # plt.plot(t,data)
    # plt.title("Données d'entrées en fonction du temps")
    # plt.xlabel("Temps en (s)")
    # plt.ylabel("Amplitude in (V)")
    # plt.show()

    # Sous- échantillonnage pour arriver à une fréquence d'échantillonnage de fe/4 = 11025 Hz
    data = data[0:data_length:4]
    # mise à jour des nouveaux paramètres
    fse = fe / 4
    data_length = data.shape[0]

    # Nouvelle visuaisation possible des données
    # tse = np.arange(0,data_length/fse,1/fse)
    # plt.figure()
    # plt.plot(tse,data)
    # plt.title("Données sous-échantillonnées en fonction du temps")
    # plt.xlabel("Temps en (s)")
    # plt.ylabel("Amplitude in (V)")
    # plt.show()

    # Libération de la mémoire avant d'effectuer les calculs principaux
    # del t,tse

    # Paramètres des STFT
    NFFT = 2**6 # taille des fft

    # retard_estimé = estimation_delay(donéées,taille_des_fenêtres_étudiées,voie à comparer)
    # retard = estimation_delay(data, NFFT, 0)

    # On va découper nos signaux en plusieurs portions pour estimer nos retards
    #signal, evaluation = delay_correction2(data, retard, NFFT, 0)

    # Affichage des signaux corrigés
    # tse = np.arange(0, signal.shape[0] / fse, 1 / fse)
    # plt.figure()
    # plt.plot(tse, signal)
    # plt.title("Données sous-échantillonnées en fonction du temps")
    # plt.xlabel("Temps en (s)")
    # plt.ylabel("Amplitude in (V)")
    # plt.show()

    # On initialise les vecteurs recevant nos signaux en performant une STFT sur une voie
    # f   : Vecteur fréquentiel permettant d'afficher les STFT
    # t   : Vecteur temporel permettant d'afficher les STFT
    # Zxx : Matrice contenant les STFT
    f, t, Zxx = stft(data, fse, nperseg=NFFT-1, nfft=NFFT-1)
    result = np.empty([np.shape(Zxx)[0], np.shape(Zxx)[1]], dtype='complex_')
    result[:, :] = Zxx

    # On applique ce calcul sur les autres sources
    # for channel in np.arange(1, nb_sources):
    #     f, t, result[:, :, channel] = stft(signal[:, channel], fse, nperseg=NFFT-1, nfft=NFFT-1)

    ## Calcul des Densité spectrales de puissance
    PSD = np.zeros((int(NFFT/2), result.shape[1]))
    for compo in np.arange(0, result.shape[1]):
        # for channel in np.arange(0, nb_sources):
            PSD[:, compo] = np.real(result[:, compo] * np.conjugate(result[:, compo]))
    dir = "Donnes_wav"
    os.chdir("C:\\Users\\Nathan\\Documents\\Nathan_Travail\\I3\\PFE\\Python\\Creation_Spectre\\Matrice_images")
    np.savetxt(list[index][:-3]+'txt', PSD)
    #np.loadtxt(list[index][:-3]+'txt')
    index +=1
    c_max_dim = PSD.shape[1]
    c_min_dim = PSD.shape[1]
    if c_max_dim > max_dim:
        max_dim = c_max_dim
    if c_min_dim < min_dim:
        min_dim = c_min_dim
print('hI')
    # object_pi = math.pi
    # file_pi = open('filename_pi.obj', 'w')
    # pickle.dump(object_pi, file_pi)
    # # Récupération des données
    # x_lignes = int(np.ceil(np.shape(signal)[0] / fft_NFFT)) * fft_NFFT
    # x = np.empty([x_lignes, np.shape(data)[1] + 1])
    #
    # for channel in np.arange(0, data_nb_channel):
    #     t, x[:, channel] = istft(result[:, :, channel], data_fe, nperseg=fft_NFFT)
    #     t, x[:, channel + 1] = istft(result[:, :, channel + 1], data_fe, nperseg=fft_NFFT)
    #
    # retrieve_signals = x[0:np.shape(data)[0], :]
    # time = t[0:np.shape(data)[0]]
