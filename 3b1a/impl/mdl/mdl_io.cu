#include "mdl.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

PAS_OPTIMISER()
Mdl_t * ouvrire_mdl(char * fichier) {
	FILE * fp = fopen(fichier, "rb");
	//
	uint Y[C], insts[C];
	FREAD(    Y, sizeof(uint), C, fp);
	FREAD(insts, sizeof(uint), C, fp);
	//
	ema_int_t * bloques[BLOQUES];
	FOR(0, b, BLOQUES) bloques[b] = lire_ema_int(fp);
	//
	Mdl_t * mdl = cree_mdl(Y, insts, bloques);
	//
	FOR(0, c, C) {
		FREAD(mdl->p[c], sizeof(float), mdl->inst_POIDS[c], fp);
	}
	//
	mdl_cpu_vers_gpu(mdl);
	fclose(fp);
	OK("Model chargé");
	//
	return mdl;
};

PAS_OPTIMISER()
void ecrire_mdl(Mdl_t * mdl, char * fichier) {
	FILE * fp = fopen(fichier, "wb");
	//
	FWRITE(mdl->    Y, sizeof(uint), C, fp);
	FWRITE(mdl->insts, sizeof(uint), C, fp);
	//
	FOR(0, b, BLOQUES) ecrire_ema_int(mdl->bloque[b], fp);
	//
	FOR(0, c, C) {
		FWRITE(mdl->p[c], sizeof(float), mdl->inst_POIDS[c], fp);
	}
	//
	fclose(fp);
};

PAS_OPTIMISER()
void enregistrer_les_lignes_brute(Mdl_t * mdl, char * fichier) {
	FILE * fp = fopen(fichier, "wb");
	//
	uint _BLOQUES = BLOQUES, _PRIXS = PRIXS;
	FWRITE(&_BLOQUES, sizeof(uint), 1, fp);
	FWRITE(&_PRIXS  , sizeof(uint), 1, fp);
	FOR(0, b, BLOQUES) {
		FWRITE(mdl->bloque[b]->brute, sizeof(float), PRIXS, fp);
	}
	//
	fclose(fp);
};