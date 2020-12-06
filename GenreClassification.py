import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, json, math, librosa

dataPath = os.getcwd()+'\\Data\\genres_original'
songLocations = []
genreLabels = []
for root, dirs, files in os.walk(dataPath):
    for name in files:
        filename = os.path.join(root, name)
        if filename != '/Data/genres_original/jazz/jazz.00054.wav':
            songLocations.append(filename)
            genreLabels.append(filename.split('\\')[8])
test
print(set(genreLabels))
