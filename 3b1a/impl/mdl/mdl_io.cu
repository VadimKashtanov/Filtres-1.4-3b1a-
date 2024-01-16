#include "mdl.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

PAS_OPTIMISER()
Mdl_t * ouvrire_mdl(char * fichier) {
	FILE * fp = fopen(fichier, "rb");
	//
	uint Y[C], insts[C], lignes[BLOQUES], decales[BLOQUES];
	FREAD(      Y, sizeof(uint), C, fp);
	FREAD(  insts, sizeof(uint), C, fp);
	FREAD( lignes, sizeof(uint), BLOQUES, fp);
	FREAD(decales, sizeof(uint), BLOQUES, fp);
	//
	Mdl_t * mdl = cree_mdl(Y, insts, lignes, decales);
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
	FWRITE(mdl->      Y, sizeof(uint), C, fp);
	FWRITE(mdl->  insts, sizeof(uint), C, fp);
	FWRITE(mdl-> lignes, sizeof(uint), BLOQUES, fp);
	FWRITE(mdl->decales, sizeof(uint), BLOQUES, fp);
	//
	FOR(0, c, C) {
		FWRITE(mdl->p[c], sizeof(float), mdl->inst_POIDS[c], fp);
	}
	//
	fclose(fp);
};