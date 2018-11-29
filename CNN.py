# Plot ad hoc mnist instances
import os
import numpy as np

# load the data
dir = "Matrice_images"    # On se place dans le répertoire contenant nos matrices
list = os.listdir(dir)

for name in list:
    # Pour chaque classe composant notre Dataset
    dir_voix   = dir+'\\'+name        # On se place dans le répertoire contenant nos matrices pour la voix courante
    list_voix  = os.listdir(dir_voix)
    nb_file    = len(list)
    index_file = 0                    # On récupère le fichier à l'index
    while index_file!=nb_file:
        X = np.loadtxt(dir+'\\'+list[index_file])
        

