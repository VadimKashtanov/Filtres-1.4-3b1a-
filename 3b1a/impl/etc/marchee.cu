#include "mdl.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

/*
def lire_ligne_model(ligne):
	with open("/home/vadim/Bureau/Filtres-V1.4+ (versions)/3b1a/lignes_brute.bin", "rb") as co: bins = co.read()
	import struct as st
	import matplotlib.pyplot as plt
	BLOQUES, PRIXS = st.unpack('II', bins[:8])
	__lignes = [st.unpack('f'*PRIXS, bins[8+(i*PRIXS)*4:8+4*(i+1)*PRIXS]) for i in range(BLOQUES)]
	plt.plot(__lignes[ligne]);plt.show()
*/

char * nom_sources[SOURCES] = {
	" prixs ",
	" macds ",
	"volumes",
	"  haut ",
	"  bas  "
};

//	Sources
float   prixs[PRIXS] = {};
float   macds[PRIXS] = {};
float volumes[PRIXS] = {};
float   hight[PRIXS] = {};
float     low[PRIXS] = {};

float *          prixs__d = 0x0;
float *          macds__d = 0x0;
float *        volumes__d = 0x0;
float *          hight__d = 0x0;
float *            low__d = 0x0;

float * sources[SOURCES] = {
	prixs, macds, volumes, hight, low
};

float * sources__d[SOURCES] = {
	prixs__d, macds__d, volumes__d, hight__d, low__d
};

void charger_les_prixs() {
	uint __PRIXS;
	FILE * fp;
	//
	fp = fopen("prixs/prixs.bin", "rb");
	ASSERT(fp != 0);
	(void)!fread(&__PRIXS, sizeof(uint), 1, fp);
	ASSERT(__PRIXS == PRIXS);
	(void)!fread(prixs, sizeof(float), PRIXS, fp);
	fclose(fp);
	//
	fp = fopen("prixs/volumes.bin", "rb");
	ASSERT(fp != 0);
	(void)!fread(&__PRIXS, sizeof(uint), 1, fp);
	ASSERT(__PRIXS == PRIXS);
	(void)!fread(volumes, sizeof(float), PRIXS, fp);
	fclose(fp);
	//
	fp = fopen("prixs/macds.bin", "rb");
	ASSERT(fp != 0);
	(void)!fread(&__PRIXS, sizeof(uint), 1, fp);
	ASSERT(__PRIXS == PRIXS);
	(void)!fread(macds, sizeof(float), PRIXS, fp);
	fclose(fp);
	//
	fp = fopen("prixs/hight.bin", "rb");
	ASSERT(fp != 0);
	(void)!fread(&__PRIXS, sizeof(uint), 1, fp);
	ASSERT(__PRIXS == PRIXS);
	(void)!fread(hight, sizeof(float), PRIXS, fp);
	fclose(fp);
	//
	fp = fopen("prixs/low.bin", "rb");
	ASSERT(fp != 0);
	(void)!fread(&__PRIXS, sizeof(uint), 1, fp);
	ASSERT(__PRIXS == PRIXS);
	(void)!fread(low, sizeof(float), PRIXS, fp);
	fclose(fp);
};

//	===========================================================

void ema_int_calc_ema(ema_int_t * ema_int) {
	//			-- Parametres --
	uint K = ema_int->K_ema;
	float _K = 1.0 / ((float)K);
	//	EMA
	ema_int->ema[0] = sources[ema_int->source][0];
	FOR(1, i, FIN) {
		ema_int->ema[i] = ema_int->ema[i-1] * (1.0 - _K) + sources[ema_int->source][i]*_K;
	}
};

//	===========================================================

nature_f fonctions_nature[NATURES] = {
	nature0__direct,
	nature1__macd,
	nature2__chiffre,
	nature3__dx,
	nature4__dxdx
};

uint NATURE_PARAMS[NATURES] = {
	0,
	1,
	1,
	2,
	4
};

uint min_param[NATURES][MAX_PARAMS] = {
	{0,0,0,0},
	{1,0,0,0},
	{1,0,0,0},
	{0,1,0,0},
	{0,1,0,1}
};

uint max_param[NATURES][MAX_PARAMS] = {
	{0,             0,       0,        0      }, 
	{MAX_COEF_MACD, 0,       0,        0      },
	{10000,         0,       0,        0      },
	{MAX_PLUS,      MAX_EMA, 0,        0      },
	{MAX_PLUS,      MAX_EMA, MAX_PLUS, MAX_EMA}
};

char * nom_natures[NATURES] {
	"directe",
	"  macd ",
	"chiffre",
	"   dx  ",
	"  dxdx "
};

ema_int_t * cree_ligne(uint source, uint nature, uint K_ema, uint intervalle, uint decale, uint params[MAX_PARAMS]) {
	ema_int_t * ret = alloc<ema_int_t>(1);
	//
	ret->source = source;
	ret->nature = nature;
	ret->K_ema  = K_ema;
	ret->intervalle = intervalle;
	ret->decale = decale;
	//
	memcpy(ret->params, params, sizeof(uint) * MAX_PARAMS);
	//
	ema_int_calc_ema(ret);
	fonctions_nature[nature](ret);
	//
	return ret;
};

void liberer_ligne(ema_int_t * ema_int) {

};

void charger_vram_nvidia() {
	CONTROLE_CUDA(cudaMalloc((void**)&  prixs__d, sizeof(float) * PRIXS));
	CONTROLE_CUDA(cudaMalloc((void**)&  macds__d, sizeof(float) * PRIXS));
	CONTROLE_CUDA(cudaMalloc((void**)&volumes__d, sizeof(float) * PRIXS));
	CONTROLE_CUDA(cudaMalloc((void**)&  hight__d, sizeof(float) * PRIXS));
	CONTROLE_CUDA(cudaMalloc((void**)&    low__d, sizeof(float) * PRIXS));
	//
	CONTROLE_CUDA(cudaMemcpy(  prixs__d,   prixs, sizeof(float) * PRIXS, cudaMemcpyHostToDevice));
	CONTROLE_CUDA(cudaMemcpy(  macds__d,   macds, sizeof(float) * PRIXS, cudaMemcpyHostToDevice));
	CONTROLE_CUDA(cudaMemcpy(volumes__d, volumes, sizeof(float) * PRIXS, cudaMemcpyHostToDevice));
	CONTROLE_CUDA(cudaMemcpy(  hight__d, volumes, sizeof(float) * PRIXS, cudaMemcpyHostToDevice));
	CONTROLE_CUDA(cudaMemcpy(    low__d, volumes, sizeof(float) * PRIXS, cudaMemcpyHostToDevice));
};

void     liberer_cudamalloc() {
	CONTROLE_CUDA(cudaFree(  prixs__d));
	CONTROLE_CUDA(cudaFree(  macds__d));
	CONTROLE_CUDA(cudaFree(volumes__d));
	CONTROLE_CUDA(cudaFree(  hight__d));
	CONTROLE_CUDA(cudaFree(    low__d));
};

void charger_tout() {
	printf("charger_les_prixs : ");    MESURER(charger_les_prixs());
	printf("charger_vram_nvidia : ");  MESURER(charger_vram_nvidia());
};

void liberer_tout() {
	titre("Liberer tout");
	liberer_cudamalloc();
};

ema_int_t * lire_ema_int(FILE * fp) {
	uint source, nature, K_ema, intervalle, decale;
	uint params[MAX_PARAMS];
	FREAD(&source,     sizeof(uint), 1, fp);
	FREAD(&nature,     sizeof(uint), 1, fp);
	FREAD(&K_ema,      sizeof(uint), 1, fp);
	FREAD(&intervalle, sizeof(uint), 1, fp);
	FREAD(&decale,     sizeof(uint), 1, fp);
	//
	FREAD(&params,     sizeof(uint), MAX_PARAMS, fp);
	//
	return cree_ligne(source, nature, K_ema, intervalle, decale, params);
};

void      ecrire_ema_int(ema_int_t * ema_int, FILE * fp) {
	FWRITE(&ema_int->source,     sizeof(uint), 1, fp);
	FWRITE(&ema_int->nature,     sizeof(uint), 1, fp);
	FWRITE(&ema_int->K_ema,      sizeof(uint), 1, fp);
	FWRITE(&ema_int->intervalle, sizeof(uint), 1, fp);
	FWRITE(&ema_int->decale,     sizeof(uint), 1, fp);
	//
	FWRITE(&ema_int->params,     sizeof(uint), MAX_PARAMS, fp);
};