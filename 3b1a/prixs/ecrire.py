#! /usr/bin/python3

ema = lambda l,K,ema=0: [ema:=(ema*(1-1/K) + e/K) for e in l]

from sys import argv

prixs         = argv[1]
sortie_prixs  = "prixs.bin"		#close
sortie_volume = "volumes.bin"	#vBTC*close - vUSDT
sortie_macd   = "macds.bin"		#macd(close)
sortie_hight  = "hight.bin"		#hight
sortie_low    = "low.bin"		#low

import struct as st

#	======== Lecture .csv ======= 
with open(prixs, "r") as co:
	text = co.read().split('\n')
	del text[0]
	del text[0]
	del text[-1]
	lignes = [l.split(',') for l in text][::-1] # <-- Important le [::-1] (car les prixs sont du plus recent au plus ancien)
	infos = [(float(Close), float(Volume_BTC), float(Volume_USDT), float(Low), float(High)) for Unix,Date,Symbol,Open,High,Low,Close,Volume_BTC,Volume_USDT,tradecount in lignes]

#	========= Ecriture ==========
prixs   = [p       for p,_,_,_,_   in infos]
_low    = [l       for _,_,_,l,_   in infos]
_hight  = [h       for _,_,_,_,h   in infos]
s=0; volumes = [s:=(s + (vb*p-vu)) for p,vb,vu,_,_ in infos]
ema12 = ema(prixs, K=12)
ema26 = ema(prixs, K=26)
macd  = [a-b for a,b in zip(ema12, ema26)]
ema9_macd = ema(macd, K=9)
histo_macd = [a-b for a,b in zip(macd, ema9_macd)]

with open(sortie_prixs, "wb") as co:
	print(f"LEN prixs   = {len(prixs)}")
	co.write(st.pack('I', len(prixs)))
	co.write(st.pack('f'*len(prixs), *prixs))

with open(sortie_volume, "wb") as co:
	print(f"LEN volumes = {len(volumes)}")
	co.write(st.pack('I', len(volumes)))
	co.write(st.pack('f'*len(volumes), *volumes))

with open(sortie_macd, "wb") as co:
	print(f"LEN macd    = {len(histo_macd)}")
	co.write(st.pack('I', len(histo_macd)))
	co.write(st.pack('f'*len(histo_macd), *histo_macd))

with open(sortie_hight, "wb") as co:
	print(f"LEN hight   = {len(_hight)}")
	co.write(st.pack('I', len(_hight)))
	co.write(st.pack('f'*len(_hight), *_hight))

with open(sortie_low,   "wb") as co:
	print(f"LEN low     = {len(_low)}")
	co.write(st.pack('I', len(_low)))
	co.write(st.pack('f'*len(_low), *_low))