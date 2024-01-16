#include "mdl.cuh"

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
		ema_int->ema[i] = sources[ema_int->source][i]*_K + ema_int->ema[i-1] * (1.0 - _K);
	}
};

//	===========================================================

void nature0__direct(ema_int_t * ema_int) {
	//			-- Parametres --
	//			-- Assertions --
	//	-- Transformation des Parametres --
	//		-- Calcule de la Nature --
	FOR(0, i, FIN) ema_int->brute[i] = ema_int->ema[i];
};

static float ema12[PRIXS], ema26[PRIXS], __macd[PRIXS], ema9_macd[PRIXS];

void nature1__macd(ema_int_t * ema_int) {
	//			-- Parametres --
	uint plus0 = ema_int->params[0];
	//			-- Assertions --
	ASSERT(min_param[MACD][0] <= plus0 && plus0 <= max_param[MACD][0]);
	//	-- Transformation des Parametres --
	float K12 = 1.0/(12.0), K26 = 1.0/(12.0), K9 = 1.0/(9.0);
	//		-- Calcule de la Nature --
	//ema12
	ema12[0] = ema_int->ema[0];
	FOR(1, t, FIN) ema12[t] = ema12[t-1]*(K12) + ema_int->ema[t]*(1-K12);
	//ema26
	ema26[0] = ema_int->ema[0];
	FOR(1, t, FIN) ema26[t] = ema26[t-1]*(K26) + ema_int->ema[t]*(1-K26);
	//__macd
	FOR(0, t, FIN) __macd[t] = ema12[t] - ema26[t];
	//ema9 du __macd
	ema9[0] = __macd[0];
	FOR(1, t, FIN) ema9[t] = ema9[t-1]*(K9) + __macd[t]*(1-K9);
	//MACD
	FOR(0, t, FIN) ema_int->brute[t] = __macd[t] - ema9[t];
};

void nature2__chiffre(ema_int_t * ema_int) {
	//			-- Parametres --
	uint cible = ema_int->params[0];
	//			-- Assertions --
	ASSERT(min_param[CHIFFRE][0] <= cible && cible <= max_param[CHIFFRE][0]);
	//	-- Transformation des Parametres --
	float chiffre = (float)cible;
	//		-- Calcule de la Nature --
	FOR(0, t, FIN) {
		float x = ema_int->ema[t];
		ema_int->brute[t] = 2*(chiffre-MIN2(fabs(x-chiffre*roundf((x+0)/chiffre)), fabs(x-chiffre*roundf((x+chiffre)/chiffre))))/chiffre-1
	}
};

void nature3__dx(ema_int_t * ema_int) {
	//			-- Parametres --
	uint plus0 = ema_int->params[0];
	//			-- Assertions --
	ASSERT(min_param[DX][0] <= plus0 && plus0 <= max_param[DX][0]);
	//	-- Transformation des Parametres --
	//		-- Calcule de la Nature --
	FOR(0, t, plus0+1) ema_int->brute[t] = 0;
	FOR(plus0+1, t, FIN) {
		float x = ema_int->ema[t];
		ema_int->brute[t] = ema_int->ema[t] - ema_int->ema[t-1-plus0];
	}
};

void nature4__dxdx(ema_int_t * ema_int) {
	//			-- Parametres --
	uint plus0 = ema_int->params[0];
	uint ema0  = ema_int->params[0];
	uint plus1 = ema_int->params[0];
	//			-- Assertions --
	ASSERT(min_param[DX][0] <= plus0 && plus0 <= max_param[DX][0]);
	ASSERT(min_param[DX][1] <= ema0  && ema0  <= max_param[DX][1]);
	ASSERT(min_param[DX][2] <= plus1 && plus1 <= max_param[DX][2]);
	//	-- Transformation des Parametres --
	float K = 1 / ((float)ema0);
	//		-- Calcule de la Nature --

	//	dx
	FOR(0, t, plus0+1) ema_int->brute[t] = 0;
	FOR(plus0+1, t, FIN) {
		float x = ema_int->ema[t];
		ema_int->brute[t] = ema_int->ema[t] - ema_int->ema[t-1-plus0];
	}

	//	ema0
	ema_int->ema[0] = ema_int->brute[0];
	FOR(1, t, FIN) ema_int->ema[t] = ema_int->ema[t-1]*(K) + ema_int->brute[t]*(1-K);

	//	dxdx
	FOR(0, t, plus1+1) ema_int->brute[t] = 0;
	FOR(plus1+1, t, FIN) {
		float x = ema_int->ema[t];
		ema_int->brute[t] = ema_int->ema[t] - ema_int->ema[t-1-plus1];
	}
};	//dx(ema(dx(ema(arr, ema0), plus0), ema1), plus1)

nature_f fonctions_nature[NATURES] = {
	nature0__direct,
	nature1__macd,
	nature2__chiffre,
	nature3__dx,
	nature4__dxdx
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
	memcpy(mdl->params, params, sizeof(uint) * MAX_PARAMS);
	//
	ema_int_calc_ema(ret);
	fonctions_nature[nature](ret);
	//
	return ret;
};

void liberer_ligne(ema_int_t * ema_int) {
	CONTROLE_CUDA(cudaFree(ema_int->    normalisee__d));
	CONTROLE_CUDA(cudaFree(ema_int->dif_normalisee__d));
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
	printf("charger_les_prixs : ");      MESURER(charger_les_prixs());
	printf("calculer_ema_norm_diff : "); MESURER(calculer_ema_norm_diff());
	printf("charger_les_prixs : ");      MESURER(charger_vram_nvidia());
};

void liberer_tout() {
	titre("Liberer tout");
	liberer_cudamalloc();
};