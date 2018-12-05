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
while index!=number_files:
    os.chdir("C:\\Users\\Nathan\\Documents\\Nathan_Travail\\I3\\PFE\\Python\\Creation_Spectre")

    # On récupère les fichiers audio générés et les métadonnées stockées dans le fichier json
    [fe, data] = read(dir+'\\'+list[index])
    nb_sources = 1#data.shape[1]
    data_length = data.shape[0]
    data = data/np.maximum(np.abs(max(data)),np.abs(min(data)))

    # Sous- échantillonnage pour arriver à une fréquence d'échantillonnage de fe/4 = 11025 Hz
    data = data[0:data_length:4]
    fse = fe / 4
    data_length = data.shape[0]

    # Paramètres des STFT
    NFFT = 2**9 # taille des fft

    f, t, Zxx = stft(data, fse, nperseg=NFFT-1, nfft=NFFT-1)
    result = np.empty([np.shape(Zxx)[0], np.shape(Zxx)[1]], dtype='complex_')
    result[:, :] = Zxx

    # Calcul des Densité spectrales de puissance
    PSD      = np.zeros((int(NFFT/2), result.shape[1]))
    PSD_MFCC = np.zeros((int(NFFT/2), result.shape[1]))
    Energie  = np.zeros((result.shape[1]))
    Label    = np.zeros((result.shape[1]))
    for compo in np.arange(0, result.shape[1]):
        PSD[:, compo] = 20*np.log10(np.abs(result[:, compo] * np.conjugate(result[:, compo])))
        PSD_MFCC[:, compo] = 1125 * np.log((1 + (np.abs(result[:, compo] * np.conjugate(result[:, compo])))/700))

        ###classement de données poubelles
        # Calcul de l'énergie d'un spectrograme pour des fréquences corresspondantes à la voix humaine
        # ie intégrale sous la courbe du spectrogramme courrant.
        Energie[compo] = np.sum(PSD_MFCC[:, compo])
        #Classement en fonction de l'énergie (voix ou silence)
        if Energie[compo]>1e-5:
            Label[compo] = 1
        else:
            Label[compo] = 0

    # On va ensuite classer nos imagettes entre voix ou bruit
    # filtrage passe bas-de la taille de nos imagettes
    Y = np.correlate(Label,np.ones((12)))

    for rang in np.arange(0,len(Y)):
        # Au cas ou on a un mélange parfait de bruit et de voix
        if Y[rang]==0:
            vect  = np.arange(rang,rang+24)
            PSD_MFCC = np.delete(PSD_MFCC,vect,axis=1)
            PSD_MFCC = np.delete(PSD_NORM,vect,axis=1)

    # Normalisation des données pour le RNA une fois que les max sont détectés
    PSD_NORM = 2*((PSD-PSD.min())/(PSD.max()-PSD.min()))-1
    PSD_MFCC = 2 *((PSD_MFCC - PSD_MFCC.min()) / (PSD_MFCC.max() - PSD_MFCC.min())) - 1

    # Sauvegarde des spectres dans les dossiers suivants
    os.chdir("C:\\Users\\Nathan\\Documents\\Nathan_Travail\\I3\\PFE\\Python\\Creation_Spectre\\Matrice_images\\MFC")
    np.savetxt(list[index][:-3]+'txt', PSD_MFCC)
    os.chdir("C:\\Users\\Nathan\\Documents\\Nathan_Travail\\I3\\PFE\\Python\\Creation_Spectre\\Matrice_images\\20LOG10")
    np.savetxt(list[index][:-3]+'txt', PSD_NORM)
    index += 1












    #np.loadtxt(list[index][:-3]+'txt')

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
    ########################################################################################################################
    ############################################   Sélection des sources ##################################################
    ########################################################################################################################
    # Ouverture du fichier json courant !! Par la suite il faudra fournir à cette fonction le nom du fichier json
    # with open(dir+'\\'+list[index].split('.')[0]+'.json') as json_data:
    #     d = json.load(json_data)
    #     print(d)

    # Visualisation des signaux d'entrées
    #t = np.arange(0, data_length / fe, 1 / fe)

    # plt.figure()
    # plt.plot(t,data)
    # plt.title("Données d'entrées en fonction du temps")
    # plt.xlabel("Temps en (s)")
    # plt.ylabel("Amplitude in (V)")
    # plt.show()

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
    # On applique ce calcul sur les autres sources
    # for channel in np.arange(1, nb_sources):
    #     f, t, result[:, :, channel] = stft(signal[:, channel], fse, nperseg=NFFT-1, nfft=NFFT-1)
