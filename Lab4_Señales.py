# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 07:02:26 2021

@author: Mario valderrama
"""
##
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sciio
import scipy.signal as sig # Análisis de señal

v_ECGSig = [] #Se crea lista
s_FsHz = 0
s_FileNum = 0 #archivo
s_File_list = ["001N1_ECG.mat","002N7_ECG.mat"]


str_FileName = s_File_list[s_FileNum]
st_File = sciio.loadmat(str_FileName)
v_ECGSig = np.double(st_File['v_ECGSig'])
s_FsHz = np.double(st_File['s_FsHz'])
v_ECGSig = v_ECGSig[0]
s_ElectricNoiseHz = 50 #Ruido para elimnar (No se utiliza) #1
#print(st_File)
##
v_Time = np.arange(0.0, np.size(v_ECGSig)) / s_FsHz

plt.figure()
plt.plot(v_Time, v_ECGSig, linewidth=1)
plt.xlabel('Time (sec.)')
plt.title("Signal: {:s}".format(str_FileName))
plt.grid(1)
plt.show()



#%%
# Obtención primera derivada, filtro pasa altas - QRS
v_ECGFiltDiff = np.zeros(np.size(v_ECGSig)) #Se crea un arreglo en 0 del tamaño de la señal
v_ECGFiltDiff[1:] = np.diff(v_ECGSig) #Se extrae la primera derivada (Se amplifican señales grandes)
v_ECGFiltDiff[0] = v_ECGFiltDiff[1] #Se arregla valor inicial ya que se recorta un valor.

s_FigNum = 2 #Cantidad de figuras a graficar
s_FigCount = 0
axhdl = plt.subplots(s_FigNum, 1, sharex=True) #Se crea el subplor, se compraten los ejes
axhdl[1][s_FigCount].plot(v_Time, v_ECGSig, linewidth=1) #Se grafica la señal cruda
axhdl[1][s_FigCount].grid(1)
axhdl[1][s_FigCount].set_title("Signal: {:s}".format(str_FileName))

s_FigCount += 1 #Siguiente figura
axhdl[1][s_FigCount].plot(v_Time, v_ECGFiltDiff, linewidth=1) #Se grafica la señal derivada
axhdl[1][s_FigCount].grid(1)

#%%
# Atenuar lo pequeña y ampliar lo que es grande, acumula los dos picos anteriores en uno solo
v_ECGFiltDiffSqrt = v_ECGFiltDiff**2
s_AccSumWinSizeSec = 0.03 # Ventana de 30 ms
s_AccSumWinHalfSizeSec = s_AccSumWinSizeSec / 2.0 # Toma la mitad del intervalo
s_AccSumWinHalfSizeSam = int(np.round(s_AccSumWinHalfSizeSec * s_FsHz)) # Nos da el núnmero de puntos de la ventana

v_ECGFiltDiffSqrtSum = np.zeros(np.size(v_ECGFiltDiffSqrt))         #Se inicializa la ventana 
for s_Count in range(np.size(v_ECGFiltDiffSqrtSum)):                #Se recorre toda la lista vacia
    s_FirstInd = s_Count - s_AccSumWinHalfSizeSam                   #Se extrae el indice inicial
    s_LastInd = s_Count + s_AccSumWinHalfSizeSam                    #Se extrae el indice final
    if s_FirstInd < 0:                                              #Si es menor a 0
        s_FirstInd = 0                                                #El indice es 0
    if s_LastInd >= np.size(v_ECGFiltDiffSqrtSum):                  #Si es mayor a la longitud total
        s_LastInd = np.size(v_ECGFiltDiffSqrtSum)                     #El indice es la longitud total
        
    v_ECGFiltDiffSqrtSum[s_Count] = np.mean(v_ECGFiltDiffSqrt[s_FirstInd:s_LastInd + 1]) #Se extrae la media para la ventana
#%%
v_PeaksInd = sig.find_peaks(v_ECGFiltDiffSqrtSum)   #Se extraen los indices de los picos
v_Peaks = v_ECGFiltDiffSqrtSum[v_PeaksInd[0]]       #Se extraen los valores de los picos
s_PeaksMean = np.mean(v_Peaks)                      #Valor medio del valor de los picos
s_PeaksStd = np.std(v_Peaks)                        #Valor std del valor de los picos

s_MinTresh = s_PeaksMean + 1 * s_PeaksStd           #Umbral minimo de un pico a detectar
s_MaxTresh = s_PeaksMean + 9.5 * s_PeaksStd         #Umbral maximo de un pico a detectar
s_QRSInterDurSec = 0.2                              #Dsitancia minima entre picos (Timepo)
s_MinDurSam = np.round(s_QRSInterDurSec * s_FsHz)   #Dsitancia minima entre picos (puntos)

##Se extraen los indices de los picos con las condiciones
v_PeaksInd,_ = sig.find_peaks(v_ECGFiltDiffSqrtSum,
                            height=[s_MinTresh, s_MaxTresh],
                            distance=s_MinDurSam)

#Corregir esa identificación de picos corridos

s_QRSPeakAdjustHalfWinSec = 0.05 #
s_QRSPeakAdjustHalfWinSam = int(np.round(s_QRSPeakAdjustHalfWinSec * s_FsHz))

for s_Count in range(np.size(v_PeaksInd)):
    s_Ind = v_PeaksInd[s_Count]
    s_FirstInd = s_Ind - s_QRSPeakAdjustHalfWinSam
    s_LastInd = s_Ind + s_QRSPeakAdjustHalfWinSam
    if s_FirstInd < 0:
        s_FirstInd = 0
    if s_LastInd >= np.size(v_ECGSig):
        s_LastInd = np.size(v_ECGSig)
    v_Aux = v_ECGSig[s_FirstInd:s_LastInd + 1]
    v_Ind1 = sig.find_peaks(v_Aux)
    if np.size(v_Ind1[0]) == 0:
        continue
    s_Ind2 = np.argmax(v_Aux[v_Ind1[0]])
    s_Ind = int(v_Ind1[0][s_Ind2])
    v_PeaksInd[s_Count] = s_FirstInd + s_Ind


#%%
s_FigNum = 4
s_FigCount = 0
axhdl = plt.subplots(s_FigNum, 1, sharex=True)
axhdl[1][s_FigCount].plot(v_Time, v_ECGSig, linewidth=1)
axhdl[1][s_FigCount].grid(1)
axhdl[1][s_FigCount].set_title("Signal: {:s}".format(str_FileName))

s_FigCount += 1
axhdl[1][s_FigCount].plot(v_Time, v_ECGFiltDiff, linewidth=1)
axhdl[1][s_FigCount].grid(1)

s_FigCount += 1
axhdl[1][s_FigCount].plot(v_Time, v_ECGFiltDiffSqrt, linewidth=1)
axhdl[1][s_FigCount].grid(1)

s_FigCount += 1
axhdl[1][s_FigCount].plot(v_Time, v_ECGFiltDiffSqrtSum, linewidth=1)
#axhdl[1][s_FigCount].plot(v_Time[v_PeaksInd],
#                          v_ECGFiltDiffSqrtSum[v_PeaksInd], '.r')
axhdl[1][s_FigCount].grid(1)
axhdl[1][s_FigCount].set_xlabel('Time (sec.)')





##%%
v_Taco = np.diff(v_PeaksInd) / s_FsHz
v_TacoBPM = 60.0 / v_Taco

s_FigNum = 3
s_FigCount = 0
axhdl = plt.subplots(s_FigNum, 1, sharex=True)
axhdl[1][s_FigCount].plot(v_Time, v_ECGSig, linewidth=1)
axhdl[1][s_FigCount].plot(v_Time[v_PeaksInd],v_ECGSig[v_PeaksInd], '.r')
axhdl[1][s_FigCount].grid(1)
axhdl[1][s_FigCount].set_title("Signal: {:s}".format(str_FileName))

s_FigCount += 1
axhdl[1][s_FigCount].plot(v_Time[v_PeaksInd[1:]], v_Taco, linewidth=1)
axhdl[1][s_FigCount].grid(1)
#axhdl[1][s_FigCount].set_xlabel('Time (sec.)')
axhdl[1][s_FigCount].set_ylabel('R-R (sec.)')
axhdl[1][s_FigCount].set_title('Tacograma (sec.)')

s_FigCount += 1
axhdl[1][s_FigCount].plot(v_Time[v_PeaksInd[1:]], v_TacoBPM, linewidth=1)
axhdl[1][s_FigCount].grid(1)
axhdl[1][s_FigCount].set_xlabel('Time (sec.)')
axhdl[1][s_FigCount].set_ylabel('BPM')
axhdl[1][s_FigCount].set_title('Tacograma (BPM)')


#%% HYPNOGRAMA

str_FileName = "./001N1_ECG.mat"
st_File = sciio.loadmat(str_FileName)
v_HypTime = np.transpose(np.double(st_File['v_HypTime']))
v_HypTime = v_HypTime[0]
v_HypCode = np.transpose(np.double(st_File['v_HypCode']))
v_HypCode = v_HypCode[0]
s_FsHz = np.double(st_File['s_FsHz'])
v_ECGSig = np.double(st_File['v_ECGSig'])
v_ECGSig = v_ECGSig[0]
v_Time = np.arange(0.0, np.size(v_ECGSig)) / \
         s_FsHz
v_HypCodeLabels = st_File['v_HypCodeLabels']
v_HypTime = v_HypTime + 15
s_NumCodes = np.int(np.size(v_HypCodeLabels) / np.size(v_HypCodeLabels[0]))
v_HypCodeTicksLabels = np.array([])
v_HypCodeTicks = np.array([])

for s_Count in range(s_NumCodes):
    v_HypCodeTicks = np.append(v_HypCodeTicks,np.int(v_HypCodeLabels[s_Count][2][0]))
    v_HypCodeTicksLabels = np.append(v_HypCodeTicksLabels,(v_HypCodeLabels[s_Count][1][0]))

v_Ind = np.argsort(v_HypCodeTicks)
v_HypCodeTicks = v_HypCodeTicks[v_Ind]
v_HypCodeTicksLabels = list(v_HypCodeTicksLabels[v_Ind])

fig = plt.figure()
axhdl = fig.add_subplot(111)

axhdl.plot(v_HypTime,v_HypCode,linewidth = 1)
axhdl.grid(1)
axhdl.set_xlabel('Time (sec.)')
axhdl.set_ylabel('Estado de sueño')
axhdl.set_yticks(v_HypCodeTicks)
axhdl.set_yticklabels(v_HypCodeTicksLabels)
axhdl.set_title('Hipnograma')
