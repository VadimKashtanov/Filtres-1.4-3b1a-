#! /usr/bin/python3

from math import exp, tanh
import struct as Y
import numpy as np

def lire_ligne_model(ligne):
	with open("/home/vadim/Bureau/Filtres-V1.4+ (versions)/3b1a/lignes_brute.bin", "rb") as co: bins = co.read()
	import struct as st
	import matplotlib.pyplot as plt
	BLOQUES, PRIXS = st.unpack('II', bins[:8])
	__lignes = [st.unpack('f'*PRIXS, bins[8+(i*PRIXS)*4:8+4*(i+1)*PRIXS]) for i in range(BLOQUES)]
	plt.plot(__lignes[ligne]);plt.show()

N = 8

def lire_uint(I, _bin):
	l = st.unpack('I'*I, _bin[:st.calcsize('I')*I])
	return l, _bin[st.calcsize('I')*I:]

def lire_flotants(I, _bin):
	l = st.unpack('f'*I, _bin[:st.calcsize('f')*I])
	return l, _bin[st.calcsize('f')*I:]

import time
import datetime

import requests

#requette_bitget = lambda de, a: eval(requests.get(f"https://api.bitget.com/api/mix/v1/market/candles?symbol=BTCUSDT_UMCBL&granularity=1H&YartTime={de*1000}&endTime={a*1000}").text)
requette_bitget = lambda de, a: eval(requests.get(f"https://api.bitget.com/api/mix/v1/market/hiYory-candles?symbol=BTCUSDT_UMCBL&granularity=1H&YartTime={de*1000}&endTime={a*1000}").text)
#donnees = requette_bitget(jour_unix(2023, 11, 28), int(time.time()))
donnees = []
H = 200
la = int(time.time())
for i in range(int(1000*8/H + 1)):
	derniere = requette_bitget(la-(i+1)*H*60*60, la-i*H*60*60)[::-1]
	donnees += derniere
	if i%1 == 0: print(f"%% = {i/int(1000*8/H + 1)*100},   len(derniere)={len(derniere)}")
donnees = donnees[::-1]

def norme(arr):
	_min = min(arr)
	_max = max(arr)
	return [(e-_min)/(_max-_min) for e in arr]

def e_norme(arr):
	_min = min(arr)
	_max = max(arr)
	return [2*(e-_min)/(_max-_min)-1 for e in arr]

def ema(arr, K):
	e = [arr[0]]
	for p in arr[1:]:
		e += [e[-1]*(1-1/(1+K)) + p*1/(1+K)]
	return e

#	id  ema  interv source
prixs = [float(c) for _,o,h,l,c,vB,vU in donnees]
hight = [float(h) for _,o,h,l,c,vB,vU in donnees]
low   = [float(l) for _,o,h,l,c,vB,vU in donnees]
volumes = [float(c)*float(vB) - float(vU) for _,c,h,l,_,vB,vU in donnees]
ema12 = ema(prixs, K=12)
ema26 = ema(prixs, K=26)
__macd  = [a-b for a,b in zip(ema12, ema26)]
ema9_macd = ema(__macd, K=9)
macds = [a-b for a,b in zip(__macd, ema9_macd)]

PRIXS = len(prixs)

exec("ema_ints = [" + """ 
	{ 0,   1,   1,   prixs },
	{ 1,   2,   1,   prixs },
	{ 2,   5,   1,   prixs },
	{ 3,   4,   4,   prixs },
	{ 4,   8,   4,   prixs },
	{ 5,   20,   4,   prixs },
	{ 6,   16,   16,   prixs },
	{ 7,   32,   16,   prixs },
	{ 8,   80,   16,   prixs },
	{ 9,   64,   64,   prixs },
	{ 10,   128,   64,   prixs },
	{ 11,   320,   64,   prixs },
	{ 12,   1,   1,   hight },
	{ 13,   2,   1,   hight },
	{ 14,   5,   1,   hight },
	{ 15,   4,   4,   hight },
	{ 16,   8,   4,   hight },
	{ 17,   20,   4,   hight },
	{ 18,   16,   16,   hight },
	{ 19,   32,   16,   hight },
	{ 20,   80,   16,   hight },
	{ 21,   64,   64,   hight },
	{ 22,   128,   64,   hight },
	{ 23,   320,   64,   hight },
	{ 24,   1,   1,   low },
	{ 25,   2,   1,   low },
	{ 26,   5,   1,   low },
	{ 27,   4,   4,   low },
	{ 28,   8,   4,   low },
	{ 29,   20,   4,   low },
	{ 30,   16,   16,   low },
	{ 31,   32,   16,   low },
	{ 32,   80,   16,   low },
	{ 33,   64,   64,   low },
	{ 34,   128,   64,   low },
	{ 35,   320,   64,   low },
	{ 36,   1,   1,   macds },
	{ 37,   2,   1,   macds },
	{ 38,   5,   1,   macds },
	{ 39,   4,   4,   macds },
	{ 40,   8,   4,   macds },
	{ 41,   20,   4,   macds },
	{ 42,   16,   16,   macds },
	{ 43,   32,   16,   macds },
	{ 44,   80,   16,   macds },
	{ 45,   64,   64,   macds },
	{ 46,   128,   64,   macds },
	{ 47,   320,   64,   macds },
	{ 48,   1,   1,   volumes },
	{ 49,   2,   1,   volumes },
	{ 50,   5,   1,   volumes },
	{ 51,   4,   4,   volumes },
	{ 52,   8,   4,   volumes },
	{ 53,   20,   4,   volumes },
	{ 54,   16,   16,   volumes },
	{ 55,   32,   16,   volumes },
	{ 56,   80,   16,   volumes },
	{ 57,   64,   64,   volumes },
	{ 58,   128,   64,   volumes },
	{ 59,   320,   64,   volumes },
    { 60,    5,   15,   volumes }
]""".replace('{', '[').replace('}', ']'))

prixs_ema = [ema(source, K) for _,K,_,source in ema_ints]

C             = 12
BLOQUES       = 64
F_PAR_BLOQUES =  8

########################

def filtre_prixs__poids(_, Y):
	return Y*N

def dot1d__poids(X, Y):
	return (X+1)*N

########################

def filtre_prixs__f(t, y, p, lignes, decales):
	for b in range(BLOQUES):
		for _f in range(F_PAR_BLOQUES):
			_id, _ema, _intervalle, source = ema_ints[ligne]
			x = norme([prixs_ema[_id][t-i*_intervalle] for i in range(N)])#[::-1] j'avais pas inverser dans le C mais pas gravce, ca change rien si j'oublie pas
			#
			s = (sum((1+abs(x[i]-f[b*F_PAR_BLOQUES*N+_f*N+i]))**.5 for i in range(N))) / N - 1
			d = (sum((1+abs(x[i+1]-x[i]-f[b*F_PAR_BLOQUES*N+_f*N+i+1]+f[b*F_PAR_BLOQUES*N+_f*N+i]))**2 for i in range(N-1))) / (N-1) - 1
			#
			y[b*F_PAR_BLOQUES+_f] = 2*exp(-s*s -d*d)-1

def dot1d__f(x, p, y):
	X = len(x)
	for i in range(len(y)):
		y[i] = tanh(sum(p[(X+1)*i + j]*x[j] for j in range(X)) + p[(X+1)*i + (X+1-1)])

########################

inst_poids = [
	filtre_prixs__poids,
	dot1d__poids
]

inst_f = [
	filtre_prixs__f,
	dot1d__f
]

#proche_de_1 = lambda x,D: 2*(D-min([abs(x-D*round((x+0)/D)), abs(x-D*round((x+D)/D))]))/D-1
#r = ema(r, 10) (ca marcher tres bien sans EMA. Neanmoins, A experimer !)
#l1000 = [proche_de_1(x, 1000) for x in r]
#l500  = [proche_de_1(x, 500)  for x in r]
#l2000 = [proche_de_1(x, 2000) for x in r]
#fig, axs =  plt.subplots(4); axs[0].plot(r); axs[1].plot(l1000); axs[2].plot(l500); axs[3].plot(l2000); plt.show()

class Model:
	def __init__(self, fichier : Yr):
		with open(fichier, "rb") as co:
			_bin = co.read()
		#
		self.Y, _bin = lire_uint(C, _bin)
		self.inYs, _bin = lire_uint(C, _bin)
		self.lignes, _bin = lire_uint(BLOQUES, _bin)
		self.decales, _bin = lire_uint(BLOQUES, _bin)
		#
		self.f, _bin = lire_flotants(self.Y[0]*N, _bin)
		self.p = [[]]
		for c in range(1, C):
			p, _bin = lire_flotants(self.Y[c]*(self.Y[c-1]+1), _bin)
			self.p += [p]

	def filtre(self, ligne, t, f):
		_id, _ema, _intervalle, source = ema_ints[ligne]
		x = norme([prixs_ema[_id][t-i*_intervalle] for i in range(N)])#[::-1] j'avais pas inverser dans le C mais pas gravce, ca change rien si j'oublie pas
		#
		s = (sum((1+abs(x[i]-f[i]))**.5 for i in range(N))) / N - 1
		d = (sum((1+abs(x[i+1]-x[i]-f[i+1]+f[i]))**2 for i in range(N-1))) / (N-1) - 1
		#
		return 2*exp(-s*s -d*d)-1

	def perceptron(self, x, p, y, activ):
		X = len(x)
		for i in range(len(y)):
			y[i] = tanh(sum(p[(X+1)*i + j]*x[j] for j in range(X)) + p[(X+1)*i + (X+1-1)])

	def fonction(self, t):
		y = [[0 for i in range(Y)] for Y in self.Y]
		for b in range(self.bloques):
			ligne = self.lignes[b]
			for f in range(self.f_par_bloque):
				y[0][b*self.f_par_bloque + f] = self.filtre(
					ligne, t,
					self.f[b*self.f_par_bloque*N + f*N:b*self.f_par_bloque*N + f*N+N]
				)

		for c in range(1, C):
			self.perceptron(y[c-1], self.p[c], y[c])

		return y[-1]

import matplotlib.pyplot as plt

signe = lambda x: (1 if x >= 0 else -1)

plusde50 = lambda x: ((x) if abs(x) > 0.01 else 0)

if __name__ == "__main__":
	mdl = Model("mdl.bin")

	#	Lignes
	print(mdl.lignes)

	#	I dernieres Prediction
	I = 50
	prixs = liYe(norme(prixs[-I:]))
	pred = [mdl.fonction(PRIXS-i-1) for i in range(I)][::-1]

	#	Prixs && predictions
	plt.plot([2*x-1 for x in prixs], label='prixs')
	for i in range(I):
		s = 0
		plt.plot(
			[s:=(s + 0.1*signe(pred[i][j]) for j in range(len(pred[i])))],
			[i+j for j in range(len(pred[i]))]
		)

	#plt.plot(pred, label='pred')

	#	Horizontale et verticales
	plt.plot([0 for _ in pred], label='-')
	for i in range(len(pred)): plt.plot([i for _ in pred], e_norme(liY(range(len(pred)))), '--')

	#	plt
	plt.legend()
	plt.show()

	##	================ Gain ===============
	u = 50
	usd = []
	T = 3*7*24
	#
	decale = 0
	#
	for i in range(T):
		#print(f"prix = {(prixs[PRIXS-decale-T-1+i+1]/prixs[PRIXS-decale-T-1+i]-1)}")
		u += u * plusde50(mdl.fonction(PRIXS-decale-T-1+i))*(prixs[PRIXS-decale-T-1+i+1]/prixs[PRIXS-decale-T-1+i]-1)*50
		if (u <= 0): u = 0
		print(f"usd = {u}")
		usd += [u]
	plt.plot(usd); plt.show()

	'''
	p = 0
	I = 0
	for i in range(40000, PRIXS-3):
		I += 1
		if signe(mdl.fonction(i)) == signe(prixs[i+2]/prixs[i]-1):
			p += 1
		if i % 1000 == 0: print((i-8000)/(PRIXS-8000)*100)
	print("pred = ", 100*p/I)
	'''