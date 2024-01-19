#include "main.cuh"

#include "../impl_tmpl/tmpl_etc.cu"

static void plume_pred(Mdl_t * mdl, uint t0, uint t1) {
	float * ancien = mdl_pred(mdl, t0, t1, 3);
	printf("PRED GENERALE = ");
	FOR(0, p, P) printf(" %f%% ", 100*ancien[p]);
	printf("\n");
	free(ancien);
};

float pourcent_masque_nulle[C] = {0};

float * pourcent_masque = de_a(0.20, 0.00, C);

float * alpha = de_a(2e-4, 2e-4, C);

PAS_OPTIMISER()
int main(int argc, char ** argv) {
	//	-- Init --
	srand(0);
	cudaSetDevice(0);
	titre(" Charger tout ");   charger_tout();

	//===============
	titre("  Programme Generale  ");

	ASSERT(argc == 2);
	Mdl_t * mdl = ouvrire_mdl(argv[1]);

	plumer_mdl(mdl);

	//	================= Initialisation ==============
	uint t0 = DEPART;
	uint t1 = ROND_MODULO(FIN, (16*16));
	printf("t0=%i t1=%i FIN=%i (t1-t0=%i, %%(16*16)=%i)\n", t0, t1, FIN, t1-t0, (t1-t0)%(16*16));
	//
	uint REP = 3;
	FOR(0, rep, REP) {
		optimisation_mini_packet(
			mdl,
			t0, t1, 16*16*1,
			alpha, 1.0,
			RMSPROP, 200,
			pourcent_masque);
		plume_pred(mdl, t0, t1);
		mdl_gpu_vers_cpu(mdl);
		ecrire_mdl(mdl, argv[1]);
		//
		printf("===================================================\n");
		printf("================ TERMINE %i/%i  =================\n", rep+1, REP);
		printf("===================================================\n");
	}
	optimiser(
		mdl,
		t0, t1,
		alpha, 1.0,
		RMSPROP, 500,
		pourcent_masque_nulle);
	//
	float _pred = mdl_pred(mdl, t0, t1, 3)[0];
	ecrire<float>("resultat.bin", &_pred, 1);
	//
	mdl_gpu_vers_cpu(mdl);
	ecrire_mdl(mdl, argv[1]);
	liberer_mdl(mdl);

	//	-- Fin --
	liberer_tout();
};