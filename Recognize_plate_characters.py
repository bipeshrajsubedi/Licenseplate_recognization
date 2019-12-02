import numpy as np
import pickle
import cv2
import License_setup
from skimage import io
from skimage.filters import threshold_otsu
from skimage.transform import resize
import imutils
from sklearn.metrics import confusion_matrix

print("LOADING MODEL...")

modelPath = "./dataModel.sav"
loadModel = pickle.load(open(modelPath,"rb"))

print("#################################################")
print("MODEL LOADED")
print(loadModel)

resultPredictTop =[]

########################################## TOP HALF OF PLATE LETTERS PREDICTION########################################

for each_top_Letter in reversed(License_setup.charactersTop):

    each_top_Letter = each_top_Letter.reshape(1,-1)
    result_top = loadModel.predict(each_top_Letter)
    resultPredictTop.append(result_top)

#PRINTING PLATE DATA

plate_characters_top = ""
print("..................................................")
print("...Predicting top half plate characters... ")
for each_top_character in resultPredictTop:
    plate_characters_top+= each_top_character[0]
print("TOP PLATE CHARACTERS ::::",plate_characters_top)

#ARRANGING PLATE CHARACTERS
orderedPlateTop = []
listCopyTop= License_setup.columnListTop[:]
print(listCopyTop)
for each_index_top in reversed(sorted(License_setup.columnListTop)):
    orderedPlateTop+= plate_characters_top[listCopyTop.index(each_index_top)]

print("FINAL PLATE TOP LETTERS::::",orderedPlateTop)
trueLetter = ["BA","9","2","P"]
print(len(trueLetter))
print(len(orderedPlateTop))
cm = confusion_matrix(trueLetter,orderedPlateTop)
print(cm)



###################################BOTTOM HALF OF PLATE LETTER TREDICTIONS############################
resultPredictBot = []
print(".....................................................")
print("...predicting bottom half plate characters...")
for each_bot_letter in License_setup.charactersBot:

    each_bot_letter = each_bot_letter.reshape(1,-1)
    result_bottom = loadModel.predict(each_bot_letter)
    resultPredictBot.append(result_bottom)

# PRINTING BOTTOM PLATE LETTERS

plate_characters_bot = ""

for each_bot_character in resultPredictBot:
    plate_characters_bot+=each_bot_character[0]

print("BOTTOM PLATE LETTERS::::",plate_characters_bot)

#ORDERING BOTTOM PLATE LETTERS
orderedPlateBot = ""
listCopyBot = License_setup.columnListBot[:]
print(listCopyBot)
for each_index_bot in sorted(License_setup.columnListBot):
    orderedPlateBot+=plate_characters_bot[listCopyBot.index(each_index_bot)]

print("FINAL BOTTOM CHARACTERS::::",orderedPlateBot)