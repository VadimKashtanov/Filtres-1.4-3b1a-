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

float * pourcent_masque = de_a(0.10, 0.00, C);

//	! A FAIRE ! :
//		selection (mutation de +/- 1 ligne (de meme source))
//

//	# Un jour reflechire a f(x@p0 + b0) * f(x@p1 + b1) + f(x@p2 + b2)

float * alpha = de_a(2e-4, 2e-4, C);

//	## (x/3) * (x-2)**2                     ##
//	## score(x) + rnd()*abs(score(x))*0.05  ##

PAS_OPTIMISER()
int main(int argc, char ** argv) {
	MSG("S(x) Ajouter un peut d'aléatoire");
	MSG("S(x) Eventuellement faire des prediction plus lointaines");
	//	-- Init --
	srand(0);
	cudaSetDevice(0);
	titre(" Charger tout ");   charger_tout();

	//	-- Verification --
	//titre("Verifier MDL");     verif_mdl_1e5();

	//===============
	titre("  Programme Generale  ");
	ecrire_structure_generale("structure_generale.bin");

	uint Y[C] = {
		512,
		512,
		256,
		128,
		64,
		32,
		16,
		8,
		P
	};
	uint insts[C] = {
		FILTRES_PRIXS,
		DOT1D,
		DOT1D,
		DOT1D,
		DOT1D,
		DOT1D,
		DOT1D,
		DOT1D,
		DOT1D
	};
	//	Assurances :
	ema_int_t * bloque[BLOQUES];
	uint params[MAX_PARAMS];
	FOR(0, i, BLOQUES) {
		uint source     = rand() % SOURCES;
		uint nature     = 0;//rand() % NATURES;
		uint K_ema      = 1;//rand() % MAX_EMA,
		uint intervalle = 1;//rand() % MAX_INTERVALLE,
		uint decale     = 0;//rand() % MAX_DECALES,
		//
		FOR(0, j, MAX_PARAMS) {
			if (max_param[nature][j]-min_param[nature][j] != 0)
				params[j] = min_param[nature][j] + (rand() % (max_param[nature][j]-min_param[nature][j]));
			else
				params[j] = max_param[nature][j];
		}
		//
		bloque[i] = cree_ligne(
			source,
			nature,
			K_ema,
			intervalle,
			decale,
			params
		);
	}
	//
	Mdl_t * mdl = cree_mdl(Y, insts, bloque);

	//Mdl_t * mdl = ouvrire_mdl("mdl.bin");

	enregistrer_les_lignes_brute(mdl, "lignes_brute.bin");


	/*

		De temps en temps. Echanger 2 connections.
	
	#	Juste avant aller_retour(mdl, t0, t1);
	for i in range(QUANTITE aleatoire):
		p[c][i], p[c][j] = p[c][j], p[c][i]

		Sans contre parties. Si l'optimisation aime, elle gardera (et modifira).
	Si elle aime pas elle annulera ce poids ou en fera autre chose.

		C'est un perturbateur qui ralentie mais qui propose des alternatives
	que la descente du gradient ne prendrait pas forcement d'elle meme.

		Ca ralentie certe, mais au moins ca decouvre plus (sans devoire meme de l'aleatoire).

	!! -> C'est comme le DROP_out. Ca se reequilibrera mais ca permet de decouvrire.

	*/

	plumer_mdl(mdl);

	//	================= Initialisation ==============
	uint t0 = DEPART;
	uint t1 = ROND_MODULO(FIN, (16*16));
	printf("t0=%i t1=%i FIN=%i (t1-t0=%i, %%(16*16)=%i)\n", t0, t1, FIN, t1-t0, (t1-t0)%(16*16));
	//
	plume_pred(mdl, t0, t1);
	//comportement(mdl, t0, t0+16*16);
	//
	uint REP = 150;
	FOR(0, rep, REP) {
		FOR(0, i, 10) {
			printf(" ================== %i/10 ================\n", i);
			optimisation_mini_packet(
				mdl,
				t0, t1, 16*16*1,
				alpha, 1.0,
				RMSPROP, 300,
				pourcent_masque);
			plume_pred(mdl, t0, t1);
			mdl_gpu_vers_cpu(mdl);
			ecrire_mdl(mdl, "mdl.bin");
		}
		//
		/*optimiser(
			mdl,
			t0, t1,
			alpha, 1.0,
			RMSPROP, 2000,
			//pourcent_masque_nulle);
			pourcent_masque);*/
		//
		mdl_gpu_vers_cpu(mdl);
		ecrire_mdl(mdl, "mdl.bin");
		plume_pred(mdl, t0, t1);
		printf("===================================================\n");
		printf("==================TERMINE %i/%i=======================\n", rep+1, REP);
		printf("===================================================\n");
	}
	//
	mdl_gpu_vers_cpu(mdl);
	ecrire_mdl(mdl, "mdl.bin");
	liberer_mdl(mdl);

	//	-- Fin --
	liberer_tout();
};