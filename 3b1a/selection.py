#! /usr/bin/python3

from os import system
import struct as st
import matplotlib.pyplot as plt
from random import randint, random

from mdl import *

def lire_ligne_model(ligne):
	with open("/home/vadim/Bureau/Filtres-V1.4+ (versions)/3b1a/lignes_brute.bin", "rb") as co: bins = co.read()
	import struct as st
	import matplotlib.pyplot as plt
	BLOQUES, PRIXS = st.unpack('II', bins[:8])
	__lignes = [st.unpack('f'*PRIXS, bins[8+(i*PRIXS)*4:8+4*(i+1)*PRIXS]) for i in range(BLOQUES)]
	plt.plot(__lignes[ligne]);plt.show()

##################################

def changer_K_ema(choix, depart, fin):
	mdl = Mdl(depart)
	#	Choix Ema int
	choix = randint(0, BLOQUES-1)
	#	Changement de valeur
	ema = mdl.ema_int[choix]['K_ema']
	alea = random()
	if alea > 0.80:
		foix = 0.10
	elif alea > 0.90:
		foix = 0.50
	else:
		foix = 1.00
	ema += int(randint(MIN_EMA-ema, MAX_EMA-ema) * foix)
	ema = max([ MIN_EMA, min([ema, MAX_EMA]) ])
	#	Ecriture du changement
	mdl.ema_int[choix]['K_ema'] = ema
	mdl.ecrire(fin)

def changer_intervalle(choix, depart, fin):
	mdl = Mdl(depart)
	#	Changement de valeur
	intervalle = mdl.ema_int[choix]['intervalle']
	alea = random()
	if alea > 0.80:
		foix = 0.10
	elif alea > 0.90:
		foix = 0.50
	else:
		foix = 1.00
	intervalle += int(randint(MIN_INTERVALLE-intervalle, MAX_INTERVALLE-intervalle) * foix)
	intervalle = max([ MIN_INTERVALLE, min([intervalle, MAX_INTERVALLE]) ])
	#	Ecriture du changement
	mdl.ema_int[choix]['intervalle'] = intervalle
	mdl.ecrire(fin)

def changer_decale(choix, depart, fin):
	mdl = Mdl(depart)
	#	Changement de valeur
	decale = mdl.ema_int[choix]['decale']
	alea = random()
	if alea > 0.80:
		foix = 0.10
	elif alea > 0.90:
		foix = 0.50
	else:
		foix = 1.00
	decale += int(randint(MIN_DECALE-decale, MAX_DECALE-decale) * foix)
	decale = max([ MIN_DECALE, min([decale, MAX_DECALE]) ])
	#	Ecriture du changement
	mdl.ema_int[choix]['decale'] = decale
	mdl.ecrire(fin)

def changer_nature(choix, depart, fin):
	mdl = Mdl(depart)
	#	Changement de valeur
	nature = mdl.ema_int[choix]['nature']
	alea = random()
	if alea > 0.80:
		foix = 0.10
	elif alea > 0.90:
		foix = 0.50
	else:
		foix = 1.00
	nature += int(randint(MIN_NATURES-nature, MAX_NATURES-nature) * foix)
	nature = max([ MIN_NATURE, min([nature, MAX_NATURES]) ])
	#	Re-initialisation parametres
	mdl.ema_int[choix]['params'] = [randint(min_params[nature][i], max_params[nature][i]) for i in range(MAX_PARAMS)]
	#	Ecriture du changement
	mdl.ema_int[choix]['nature'] = nature
	mdl.ecrire(fin)

def changer_params(choix, depart, fin):
	mdl = Mdl(depart)
	#	Changement de valeur
	nature      = mdl.ema_int[choix]['nature']
	if NATURE_PARAMS[nature] != 0:
		choix_param = randint(0, NATURE_PARAMS[nature]-1)
		param       = mdl.ema_int[choix]['params'][choix_param]
		alea = random()
		if alea > 0.80:
			foix = 0.10
		elif alea > 0.90:
			foix = 0.50
		else:
			foix = 1.00
		param += int(randint(min_params[nature][choix_param]-param, max_params[nature][choix_param]-param) * foix)
		param = max([ min_params[nature][choix_param], min([param, max_params[nature][choix_param]]) ])
		#	Ecriture du changement
		mdl.ema_int[choix]['params'][choix_param] = param
	mdl.ecrire(fin)

##################################

def pred(fichier, resultat):
	system(f"./prog4__sel_entrainnement {fichier}")
	with open(resultat, "rb") as co:
		return st.unpack('f', co.read())[0]

##################################

if __name__ == "__main__":
	fichier    = "mdl.bin"
	changement = "alternatif.bin"
	resultat   = "resultat.bin"

	parmis = [
		changer_K_ema,
		changer_intervalle,
		changer_decale,
		changer_nature,
		changer_params
	]

	ch_gagne = 0

	i = 0
	while True:
		print(f"python3 >>> =================== ######## Itération {i} ######### ======================");
		#
		choix_ema_int = randint(0, BLOQUES-1)
		#for i in range(randint(1, 3)):
		parmis[randint(0, len(parmis)-1)](choix_ema_int, fichier, changement)
		#
		pred_fichier    = pred(fichier,    resultat)
		pred_changement = pred(changement, resultat)
		#
		i += 1
		if pred_changement > pred_fichier:
			system(f"rm {fichier}             ")
			system(f"cp {changement} {fichier}")
			ch_gagne += 1
			print(f"python3 >>> Changements gagants : {int(ch_gagne/i*100)}% sur {i} changements")