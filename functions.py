import os
import numpy as np
import matplotlib.pyplot as plt
import wave
from scipy.io.wavfile import read
from scipy.signal import stft,istft,correlate

########################################################################################################################
########################################################################################################################
###########################################   Fonction utiles pour le main   ###########################################
########################################################################################################################
########################################################################################################################

def STFT(fft_overlap_fac,fft_NFFT,data,data_fe,data_nb_channel,data_length):

    fft_hop_size = np.int32(np.floor(fft_NFFT * (1 - fft_overlap_fac)))

    # Zero-padding pour la sélection des données
    fft_nb_segments = np.ceil(data_length / fft_NFFT)
    fft_nb_zeros = int(fft_nb_segments * fft_NFFT - data_length)
    data_proc = np.concatenate((data, np.zeros((fft_nb_zeros+fft_hop_size,data_nb_channel))))

    # Initialisation des params de la fft
    fft_total_segments = np.int32(np.ceil(len(data) / np.float32(fft_hop_size)))
    fft_window = np.hanning(fft_NFFT)  # our half cosine window
    fft_inner_pad = np.zeros(fft_NFFT)  # the zeros which will be used to double each segment size

    # données attendues en sortie
    result = np.empty((fft_total_segments, data_nb_channel, fft_NFFT), dtype=np.float32)  # space to hold the result

    for i in np.arange(fft_total_segments):  # for each segment
        current_hop = fft_hop_size * i  # figure out the current segment offset
        segment = np.transpose(data_proc[current_hop:current_hop + fft_NFFT])  # get the current segment
        for k in (0, data_nb_channel - 1):
            windowed = segment[k, :] * fft_window  # multiply by the half cosine function
            padded = np.append(windowed, fft_inner_pad)  # add 0s to double the length of the data
            spectrum = np.fft.fft(padded) / fft_NFFT  # take the Fourier Transform and scale by the number of samples
            #autopower = np.abs(spectrum * np.conj(spectrum))  # find the autopower spectrum
            result[i,k, :] = np.abs(spectrum[:fft_NFFT])  # append to the results array
    #result = 20 * np.log10(result)  # scale to db
    #result = np.clip(result, -40, 200)  # clip values
    return result


def estimation_delay_f(result,t,data_fe,fft_NFFT):
    moy_delay = 0
    for tranche in np.arange(0,result.shape[1]-1):
        ts = t[tranche+1]-t[tranche]
        time_vector = np.arange(0,ts,1/data_fe)
        X  = result[0:int(fft_NFFT/2),tranche,0]
        Y  = result[0:int(fft_NFFT/2),tranche,1]
        Z  = np.fft.ifft(abs(X*np.conjugate(Y)))
        idx = np.argmax(Z)
        # Si le delay est inchangé nous avons toujours la même voix
        delay = idx - int(fft_NFFT/4)
        plt.figure()
        plt.plot(np.real(Z))
        plt.xlabel("Nombre d'échantillons de la fft")
        plt.show()
        moy_delay += delay
    moy_delay = int(np.round(moy_delay/(result.shape[1]-1)))
    return moy_delay





def estimation_delay(data,window_size,canal):
    delay       = 0  # moyenne des retards estimés pour une coarse_delay_estimation
    data_length = data.shape[0]
    m           = int(np.ceil(data_length/window_size))

    #zero_padding pour un bon découpage
    nb_zeros = m*window_size - data_length
    signal   = np.concatenate((data,np.zeros((int(nb_zeros),data.shape[1]))))
    del nb_zeros

    #Création du vecteur acceuillant les différents retards
    delay = np.zeros((m,(data.shape[1])),dtype=int)
    if delay.shape[0]==0:
        return "Pas de retard à estimer, nous étudions un signal unidimensionel"

    for ii in range(0,m):
        for channel in range(0,delay.shape[1]):
            result = correlate(signal[ii * window_size:(ii + 1) * window_size,canal],
                       signal[ii *window_size:(ii + 1) *window_size,channel])
            if np.argmax(result)==0:
                delay[ii, channel] = 0
            else:
                delay[ii,channel] = np.argmax(result) - (window_size-1)
    return delay







def delay_correction2(data,delay,window_size,canal):

    m        = int(np.ceil(data.shape[0] / window_size))
    # zero_padding pour un bon découpage
    nb_zeros = m * window_size - data.shape[0] + np.argmax(delay)
    signal   = np.concatenate((data, np.zeros((int(nb_zeros), data.shape[1]))))

    #Estimate the maximum number of zeros for the delayed data vector
    signal_out = np.zeros((signal.shape[0],signal.shape[1]))
    evaluation = np.zeros((m,), dtype=int)
    resultat = np.zeros((window_size, data.shape[1]))
    for ii in range(0,m):
        for channel in range(0, delay.shape[1]):
            if delay[ii, channel] == 0:
                resultat[:,channel] = signal[ii * window_size:(ii + 1) * window_size, channel]
                # Pas de retard considéré on prend le signal courant

            if delay[ii,channel]> 0 :
                # la voie étudiée est en avance par rapport au canal principal
                # il faut donc prendre le signal décalé de delay
                resultat[:,channel] = signal[ii * window_size - np.abs(delay[ii,channel]):
                                          (ii + 1) * window_size - np.abs(delay[ii,channel]), channel]
            if delay[ii,channel]< 0 :
                # la voie étudiée est en retard par rapport au canal principal
                # il faut donc prendre le signal décalé de delay
                resultat[:,channel] = signal[ii * window_size + np.abs(delay[ii, channel]):
                                           (ii + 1) * window_size + np.abs(delay[ii, channel]), channel]
        pic = np.argmax(correlate(resultat[:, 0], resultat[:, 1]))
        if (pic!=0):
            evaluation[ii] = pic - (window_size-1)
        else:
            evaluation[ii] = 0
        if resultat.shape[0]==window_size:
            signal_out[ii * window_size:(ii + 1) * window_size,:] = resultat
        else:
            crop_data = resultat
            signal_out[ii * window_size:ii * window_size + crop_data.shape[0],:] = crop_data

    return signal_out,evaluation


def I_STFT(result,fft_overlap_fac,fft_NFFT,data_fe):
    # On récupère les paramètres généraux
    nb_segments = len(result[:,0,:])
    window      = np.hanning(fft_NFFT)
    fft_hop_size= np.int32(np.floor(fft_NFFT * (1 - fft_overlap_fac)))
    signal_out      = np.zeros([len(time),data_nb_channel])
    for k in (0, nb_segments - 1):
        for channel in np.arange(0,len(result[0,:,:])-1):
            spectrum = result[k,channel,:]
            signal = np.real(np.fft.ifft(spectrum,fft_NFFT)* fft_NFFT)  # take the Fourier Transform and scale by the number of samples
            # 1/ Ajouter les données aux données précédentes et continuer le vecteur temps
            signal_out = signal_out[k*fft_NFFT + fft_hop_size:(k+1)*fft_NFFT ,channel]
            # autopower = np.abs(spectrum * np.conj(spectrum))  # find the autopower spectrum
            time_signal = signal[:,fft_NFFT]

    return time_signal
########################################################################################################################
########################################################################################################################
###########################################   Fonction d'affichage   ###################################################
########################################################################################################################
########################################################################################################################


def delay_correction(data,delay,window_size,canal):

    m        = int(np.ceil(data.shape[0] / window_size))
    # zero_padding pour un bon découpage
    nb_zeros = m * window_size - data.shape[0]
    signal   = np.concatenate((data, np.zeros((int(nb_zeros), data.shape[1]))))

    #Estimate the maximum number of zeros for the delayed data vector
    nb_zeros = np.amax(np.abs(delay))
    signal_out = []
    evaluation = np.zeros((m,), dtype=int)
    for ii in range(0,m):
        result = signal[ii * window_size:(ii + 1) * window_size, :]
        data_corrected = np.zeros((nb_zeros + result.shape[0], data.shape[1]))

        for channel in range(0, delay.shape[1]):
            if delay[ii,channel]> 0 :
                nb_zeros_beg = np.abs(delay[ii, channel])
                nb_zeros_end = nb_zeros - np.abs(delay[ii, channel]);
                data_corrected[:, channel] = np.concatenate((np.zeros((nb_zeros_beg)), result[:, channel], np.zeros((nb_zeros_end))), axis=0)
            else:
                # On cherche le delay courant
                nb_zeros_beg = nb_zeros - np.abs(delay[ii, channel])
                nb_zeros_end = np.abs(delay[ii, channel])
                data_corrected[:,channel] = np.concatenate((np.zeros((nb_zeros_beg))  ,result[:,channel]  ,np.zeros((nb_zeros_end))),axis=0)
        evaluation[ii] = np.argmax(correlate(data_corrected[:,0],data_corrected[:,1]) ) - (window_size-1)
        signal_out.append(data_corrected)
    signal_out = np.reshape(signal_out,[signal_out[0].shape[0]*(ii+1),data.shape[1]])
    return signal_out,evaluation




def affiche_signal_in(signal,fe):
    Time = np.linspace(0, len(signal[0]) / fe, num=len(signal[0]))
    plt.plot(Time, signal[0,:], 'b',Time, signal[1,:], 'r')
    plt.xlabel("Time (in s)")
    plt.ylabel("Amplitude")
    plt.title("Signaux récupérés")
    plt.show()