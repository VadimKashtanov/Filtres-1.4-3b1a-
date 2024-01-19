#pragma once

#define DEBBUG false

#include "etc.cuh"

#define PRIXS 55548 //u += u*f*levier*(p[i+L]/p[i]-1)
#define P 1 		//[Nombre de sorties du Model]
#define P_INTERV 10	//(p[i+1+P*P_INTERV]/p[i]-1)

#define N_FLTR  8
#define N       N_FLTR

#define MAX_INTERVALLE 256
#define MAX_DECALES     32

#define DEPART ((N_FLTR+MAX_DECALES)*MAX_INTERVALLE)
#if (DEBBUG == false)
	#define FIN (PRIXS-P-1)
#else
	#define FIN (DEPART+1)
#endif

//	--- Sources ---

#define SOURCES 5

extern char * nom_sources[SOURCES];

//	Sources en CPU
extern float   prixs[PRIXS];	//  prixs.bin
extern float   macds[PRIXS];	//   macd.bin
extern float volumes[PRIXS];	// volume.bin
extern float   hight[PRIXS];	//  prixs.bin
extern float     low[PRIXS];	//  prixs.bin

extern float * sources[SOURCES];

//	Sources en GPU
extern float *   prixs__d;	//	nVidia
extern float *   macds__d;	//	nVidia
extern float * volumes__d;	//	nVidia
extern float *   hight__d;	//	nVidia
extern float *     low__d;	//	nVidia

extern float * sources__d[SOURCES];

void   charger_les_prixs();
void charger_vram_nvidia();
//
void  liberer_cudamalloc();
//
void charger_tout();
void liberer_tout();

//	---	analyse des sources ---

#define MAX_PARAMS 4
#define    NATURES 5

#define  DIRECT 0
#define    MACD 1
#define CHIFFRE 2
#define      DX 3
#define    DXDX 4

extern uint min_param[NATURES][MAX_PARAMS];
extern uint max_param[NATURES][MAX_PARAMS];

extern uint NATURE_PARAMS[NATURES];

extern char * nom_natures[NATURES];

#define       MAX_EMA 500
#define      MAX_PLUS 500
#define MAX_COEF_MACD 200

typedef struct {
	//	Intervalle
	uint      K_ema;	//ASSERT(1 <=      ema   <= inf           )
	uint intervalle;	//ASSERT(1 <= intervalle <= MAX_INTERVALLE)
	uint     decale;	//ASSERT(0 <=   decale   <= MAX_DECALES   )

	//	Nature
	uint nature;
	/*	Natures: ema-K, macd-k, chiffre-M, dx, dxdx, dxdxdx
			directe : {}							// Juste le Ema_int
			macd    : {coef }   					// le macd sera ema(9*c)-ema(26*c) sur ema(prixs,k)
			chiffre : {cible}						// Peut importe la cible, mais des chiffres comme 50, 100, 1.000 ... sont bien
			dx      : {plus0, ema0}					// dx(ema(arr, ema0), plus0)
			dxdx    : {plus0, ema0, plus1, ema1} 	// dx := r[i+plus0]-r[i])
	*/
	uint params[MAX_PARAMS];

	//	Valeurs
	float   ema[PRIXS];
	float brute[PRIXS];

	/*	Note : dans `normalisee` et `dif_normalisee`
	les intervalles sont deja calculee. Donc tout
	ce qui est avant DEPART n'est pas initialisee (car pas utilisee).
	*/
	uint source;
} ema_int_t;

void ema_int_calc_ema(ema_int_t * ema_int);

//	Outils qui composent les natures
void _outil_ema(float * y, float * x, uint K);
void _outil_macd(float * y, float * x, float coef);
void _outil_chiffre(float * y, float * x, float chiffre);
void _outil_dx(float * y, float * x, uint plus);

//	Les natures
void nature0__direct (ema_int_t * ema_int);
void nature1__macd   (ema_int_t * ema_int);
void nature2__chiffre(ema_int_t * ema_int);
void nature3__dx     (ema_int_t * ema_int);
void nature4__dxdx   (ema_int_t * ema_int);

typedef void (*nature_f)(ema_int_t*);
extern nature_f fonctions_nature[NATURES];

void      calculer_normalisee(ema_int_t * ema_int);
void calculer_diff_normalisee(ema_int_t * ema_int);

//	Mem
ema_int_t * cree_ligne(uint source, uint nature, uint K_ema, uint intervalle, uint decale, uint params[MAX_PARAMS]);
void     liberer_ligne(ema_int_t * ema_int);

//	IO
ema_int_t * lire_ema_int(FILE * fp);
void      ecrire_ema_int(ema_int_t * ema_int, FILE * fp);