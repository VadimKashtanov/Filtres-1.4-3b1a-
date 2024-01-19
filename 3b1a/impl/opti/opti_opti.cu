#include "opti.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

PAS_OPTIMISER()
void __interne_optimiser(
	Mdl_t * mdl,
	uint t0, uint t1,
	float * alpha, float div,
	uint methode, uint I,
	uint ** masque)
{
	//	Cree les listes pour les `hist` si un opti en a besoin 
	Opti_classe_t opti_classe;
	if      (methode == 0) opti_classe.sgd = (uint)NULL;
	else if (methode == 1) opti_classe.rmsprop = cree_rmsprop(mdl);
	else ERR("Pas de methode %i d'optimisation", methode);
	
	//	Plumer grad pour mieux y voire
	mdl_plume_grad(mdl, t0, t1);
	
	/* ------- Optimisation ----------- */
	FOR(0, i, I) {
		mdl_aller_retour(mdl, t0, t1, 3);
		//
		if (methode == 0) opti_simple(mdl, alpha, div, masque);
		if (methode == 1) opti_rmsprop(mdl, opti_classe.rmsprop, alpha, div, masque);
		//
		if (NORMER_LES_FILTRES) mdl_normer_les_filtres(mdl);
		if (BORNER_LES_FILTRES) mdl_borner_les_filtres(mdl);
		//
		if (i % 50 == 0) {
			float* __pred = mdl_pred(mdl, t0, t1, 3);
			float _score = mdl_score(mdl, t0, t1, 3);
			printf("%3.i/%3.i| perf={", i, I);
			FOR(0, p, P) printf("%+f%%, ", 100*__pred[p]);
			free(__pred);
			printf("} score=\033[93m%+f\033[0m\n", _score);
			if (_score < 0.001) {
				printf("Score < 0.0001 => Fin d'optimisation\n");
				break;
			}
		}
	}

	//	Liberer
	if (methode == 0) opti_classe.sgd = 0;
	else if (methode == 1) liberer_rmsprop(opti_classe.rmsprop);
};

void optimiser(
	Mdl_t * mdl,
	uint t0, uint t1,
	float * alpha, float div,
	uint methode, uint I,
	float * pourcent_masque)
{
	Masque_t * masque = cree_masque(mdl, pourcent_masque);
	//
	__interne_optimiser(
		mdl,
		t0, t1,
		alpha, div,
		methode, I,
		masque->masque);
	//
	sortire_masque(mdl, masque);
};