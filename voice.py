import sys
import numpy
from pylab import *
from scipy.io import wavfile
from matplotlib import pyplot as plt
from ipywidgets import *
import warnings
import math

warnings.filterwarnings('ignore')
samplerate, data = wavfile.read(sys.argv[1]) #czytamy plik podany jako parametr

#jesli tablica data jest dwuwymiarowa - jest dzwiek dwukanalowy - wybieramy pierwszy kanal
if data.ndim == 2:
    data = data[:,0]

n = len(data)
T = 1 / samplerate

#Dzielimy sygnal na czesci
part = n//9
m = 0
k = 0

for x in range(0,9):
    datapart = data[x*part:part*(x+1)]

    #Wyznaczamy okno
    hann = []
    for i in range(0,part):
        hann.append(0.5-0.5*math.cos((2*math.pi*i)/(part-1)))

    datapart = datapart * hann

    #Wykonujemy FFT
    fourier = np.fft.fft(datapart)
    fourier = abs(fourier) # modul
    fourier = fourier/part*2 #skalowanie osi

    freqs = np.fft.fftfreq(part,T) #wyznaczamy czestotliwości
    pfreqs = freqs[freqs >= 0]
    fourier = fourier[freqs >=0] #wybieramy tylko dodatnie czestotliwosci

    #algorytm HPS
    multipliedfourier = []
    lowerbound = 0

    for i in range(0,len(pfreqs)//4):
        multipliedfourier.append(fourier[i]*fourier[2*i]*fourier[3*i]*fourier[4*i])
        if pfreqs[i]>=70 and lowerbound==0:
            lowerbound = i

    for i in range(0,lowerbound):
        multipliedfourier[i] = 0

    frequency = pfreqs[numpy.argmax(multipliedfourier)] #dominujaca czestotliwosc

    #decyzja
    if frequency < 165:
        m+=1
    else:
        k+=1

if m>k:
    print("M",end="")
else:
    print("K",end="")