#include "main.cuh"

void ecrire_structure_generale(char * file) {
	FILE * fp = fopen(file, "wb");
	//
	uint constantes[18] = {
		P,
		P_INTERV,
		N,
		MAX_INTERVALLE,
		MAX_DECALES,
		SOURCES,
		MAX_PARAMS, NATURES,
		MAX_EMA, MAX_PLUS, MAX_COEF_MACD,
		NORMER_LES_FILTRES, BORNER_LES_FILTRES,
		C, MAX_Y, BLOQUES, F_PAR_BLOQUES,
		INSTS
	};
	//
	FWRITE(constantes, sizeof(uint), 18, fp);
	//
	FWRITE(min_param,  sizeof(uint), NATURES*MAX_PARAMS, fp);
	FWRITE(max_param,  sizeof(uint), NATURES*MAX_PARAMS, fp);
	//
	FWRITE(NATURE_PARAMS,  sizeof(uint), NATURES, fp);
	//
	fclose(fp);
};