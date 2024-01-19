import struct as st

def lire_uint(I, _bin):
	l = st.unpack('I'*I, _bin[:st.calcsize('I')*I])
	return l, _bin[st.calcsize('I')*I:]

def lire_flotants(I, _bin):
	l = st.unpack('f'*I, _bin[:st.calcsize('f')*I])
	return l, _bin[st.calcsize('f')*I:]

with open("structure_generale.bin", "rb") as co:
	bins = co.read()

	constantes, bins = lire_uint(18, bins)

	exec("""P,
		P_INTERV,
		N,
		MAX_INTERVALLES,
		MAX_DECALES,
		SOURCES,
		MAX_PARAMS, NATURES,
		MAX_EMA, MAX_PLUS, MAX_COEF_MACD,
		NORMER_LES_FILTRES, BORNER_LES_FILTRES,
		C, MAX_Y, BLOQUES, F_PAR_BLOQUES,
		INSTS
		""".replace('\n', '') + " = constantes")

	min_param, bins = lire_uint(NATURES*MAX_PARAMS, bins)
	max_param, bins = lire_uint(NATURES*MAX_PARAMS, bins)

	NATURE_PARAMS, bins = lire_uint(NATURES, bins)

MIN_EMA = 1
MIN_NATURES = 0
MIN_NATURES = NATURES-1
MIN_INTERVALLES = 1

class Mdl:
	inst_poids = [
		lambda Y,X: X*N,
		lambda Y,X: (Y+1)*X,
		0,
		0
	]
	def __init__(self, fichier):
		with open(fichier, "rb") as co:
			bins = co.read()

		self.Y,     bins = lire_uint(C, bins)
		self.insts, bins = lire_uint(C, bins)

		self.ema_int = []
		for _ in range(BLOQUES):
			(source, nature, K_ema, intervalle, decale), bins = lire_uint(5, bins)
			params, bins = lire_uint(MAX_PARAMS, bins)

			self.ema_int += [{
				'source' : source,
				'nature' : nature,
				'K_ema' : K_ema,
				'intervalle' : intervalle,
				'decale' : decale,
				'params' : params
			}]

		self.p = []
		self.poids = []
		for i in range(C):
			Y, X = (self.Y[i-1] if i!=0 else 0), self.Y[i]
			self.poids += [self.inst_poids[self.insts[i]](Y,X)]
			poids, bins = lire_flotants(self.poids[i], bins)
			self.p += [poids]

	def ecrire(self, fichier):
		with open(fichier, "wb") as co:
			co.write(st.pack('I'*C, *self.Y))
			co.write(st.pack('I'*C, *self.insts))
			#
			for ema_int in self.ema_int:
				co.write(st.pack(
					'I'*5,
					ema_int['source'],
					ema_int['nature'],
					ema_int['K_ema'],
					ema_int['intervalle'],
					ema_int['decale']
				))
				co.write(st.pack('I'*MAX_PARAMS, *ema_int['params']))
			#
			for i in range(len(self.p)):
				co.write(st.pack('f'*self.poids[i], *self.p[i]))