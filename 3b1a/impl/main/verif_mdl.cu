#include "main.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

/*static float _pourcent_masque_nulle[C] = {0};
static float _alpha[C] = {0.01};

__global__
static void kerd_p1e5(float * p, uint i, float _1E5) {
	p[i] += _1E5;
};

static void p1e5(Mdl_t * mdl, uint c, uint i, float _1E5, uint _MODE) {
	if (_MODE == 0) {
		mdl->p[c][i] += _1E5;
	} else {
		kerd_p1e5<<<1,1>>>(mdl->p__d[c], i, _1E5);
		ATTENDRE_CUDA();
	}
};*/

static void __performance() {
	/*ASSERT(C == 11);
	titre("Performance");
	//
	uint Y[C] = {
		512,
		256,
		256,
		256,
		128,
		64,
		32,
		16,
		8,
		4,
		P
	};
	uint insts[C] = {
		FILTRES_PRIXS,
		LSTM1D,
		LSTM1D,
		LSTM1D,
		DOT1D,
		DOT1D,
		DOT1D,
		DOT1D,
		DOT1D,
		DOT1D,
		DOT1D
	};
	uint lignes[BLOQUES] = {0};
	FOR(0, i, BLOQUES) lignes[i] = rand() % EMA_INTS;
	Mdl_t * mdl = cree_mdl(Y, insts, lignes);
	plumer_mdl(mdl);
	//
	uint plus_T = 16*16*25;
	//
	mdl_plume_grad(mdl, DEPART, DEPART+plus_T);
	//
	printf("TEMPS MODEL = ");
	MESURER(mdl_aller_retour(mdl, DEPART, DEPART+plus_T, 3));
	//
	liberer_mdl(mdl);*/
};

static void __verif_mdl_1e5() {
	/*ASSERT(C == 5);
	titre("Comparer MODEL 1e-5");
	//
	uint Y[C] = {
		64,
		64,
		32,
		8,
		P
	};
	uint insts[C] = {
		FILTRES_PRIXS,
		LSTM1D,
		LSTM1D,
		LSTM1D,
		DOT1D
	};
	uint lignes[BLOQUES] = {
		0
	};
	Mdl_t * mdl = cree_mdl(Y, insts, lignes);
	plumer_mdl(mdl);
	//
	uint plus_T = 16*16*1;
	//
	uint t0 = DEPART;
	uint t1 = ROND_MODULO(FIN, 16*16);
	//
	//mdl_plume_poids(mdl);
	//
	//comportement(mdl, DEPART, DEPART+16*16);
#define MODE 0 //0,1,2,3
	//
	MESURER(mdl_aller_retour(mdl, DEPART, DEPART+plus_T, MODE));
	//mdl_gpu_vers_cpu(mdl);
	//
	//	1e-5
	//
	mdl_zero_gpu(mdl);
	float _f = mdl_score(mdl, DEPART, DEPART+plus_T, MODE);
	float _1E5 = 1e-5;
	FOR(0, c, C) {
		printf("###############################################################\n");
		printf("                       C = %2.i (%s)    \n", c, nom_inst[mdl->insts[c]]);
		printf("#######################vvvvvvvvvvvvvv##########################\n");
		//
		FOR(0, i, mdl->inst_POIDS[c]) {
			p1e5(mdl, c, i, +_1E5, MODE);
			float grad_1e5 = (mdl_score(mdl, DEPART, DEPART+plus_T, MODE) - _f)/_1E5;
			p1e5(mdl, c, i, -_1E5, MODE);
			//
			float a=grad_1e5, b=mdl->dp[c][i];
			printf("%i| ", i);
			PLUME_CMP(a, b);
			printf("\n");
		}
	};
	printf("  1e5 === df(x)  \n");

	//
	liberer_mdl(mdl);*/
};

void verif_mdl_1e5() {
	__performance();
	__verif_mdl_1e5();
};