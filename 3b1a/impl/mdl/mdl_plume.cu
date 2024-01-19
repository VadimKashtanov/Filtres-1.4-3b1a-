#include "mdl.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

void mdl_plume_poids(Mdl_t * mdl) {
	FOR(0, c, C) {
		printf("=========== C= %i ==========\n", c);
		plume_inst[mdl->insts[c]](mdl, c);
	}
};

void plumer_mdl(Mdl_t * mdl) {
	printf("\033[4m Plume mdl\033[0m\n");
	printf("Filtres : %i\n", mdl->Y[0]);
	printf("Lignes des bloques (pour filtres_prixs): \n");
	FOR(0, i, BLOQUES) {
		printf("%3.i| %s  source=%s decale=%3.i intervalle=%3.i K_ema=%3.i ",
			i,
			nom_natures[mdl->bloque[i]->nature],
			nom_sources[mdl->bloque[i]->source],
			mdl->decales[i],
			mdl->intervalles[i],
			mdl->bloque[i]->K_ema
		);
		FOR(0, j, MAX_PARAMS) printf("P%i=%3.i ", j, mdl->bloque[i]->params[j]);
		printf("\n");
	}
	printf("Instructions : \n");
	uint POIDS = 0;
	FOR(0, c, C) {
		POIDS += mdl->inst_POIDS[c];
		printf("%2.i| %s:<Y=%3.i> [poids=%6.i]\n", c, nom_inst[mdl->insts[c]], mdl->Y[c], mdl->inst_POIDS[c]);
	}
	printf("Quantitée poids = %i\n", POIDS);
	printf(" --- fin plume mdl ---\n");
};

void comportement(Mdl_t * mdl, uint t0, uint t1) {
	mdl_f(mdl, t0, t1, 3);
	mdl_gpu_vers_cpu(mdl);
	FOR(0, c, C) {
		printf("###=== c=%i ===###\n", c);
		FOR(0, i, mdl->inst_VARS[c]) {
			printf("%3.i| ", i);
			FOR(t0, t, t1) printf("%+f ", mdl->y[c][t*mdl->inst_VARS[c] + i]);
			printf("\n");
		}
	}
};

void cmp_dy_dp(Mdl_t * mdl, uint t0, uint t1) {
	printf(" ########## COMPARER DY #########\n");
	FOR(0, c, C) {
		float * m = gpu_vers_cpu<float>(mdl->dy__d[c], mdl->inst_VARS[c]*t1);
		printf("###=== c=%i ===###\n", c);
		FOR(0, i, mdl->inst_VARS[c]) {
			printf("%3.i| cpu : ", i);
			FOR(t0, t, t1) printf("%+f ", mdl->dy[c][i+t*mdl->inst_VARS[c]]);

			printf(" gpu : ");

			FOR(t0, t, t1) printf("%+f ", m[i+t*mdl->inst_VARS[c]]);
			printf("\n");
		}
		free(m);
	}

	printf(" ########## COMPARER DP #########\n");
	FOR(1, c, C) {
		float * m = gpu_vers_cpu<float>(mdl->dp__d[c], mdl->inst_POIDS[c]);
		printf("###=== c=%i ===###\n", c);
		FOR(0, i, mdl->inst_POIDS[c]) {
			printf("%3.i| ", i);

			printf("cpu : ");
			printf("%+f ", mdl->dp[c][i]);

			printf(" gpu : ");
			printf("%+f ", m[i]);

			printf("\n");
		}
		free(m);
	}
};

void mdl_plume_grad(Mdl_t * mdl, uint t0, uint t1) {
	mdl_aller_retour(mdl, t0, t1, 3);
	//
	FOR(0, c, C) {
		printf(" Couche %i \033[93mY=%i poids=%i\033[0m", c, mdl->Y[c], mdl->inst_POIDS[c]);
		uint POIDS = mdl->inst_POIDS[c];
		float * _grad = gpu_vers_cpu<float>(mdl->dp__d[c], POIDS);
		float * ____p = gpu_vers_cpu<float>(mdl->p__d[c], POIDS);
		//
		//	Grad
		//
		float moyenne = 0;
		float min=_grad[0], max=_grad[0];
		FOR(0, i, POIDS) {
			if (_grad[i] > max) max = _grad[i];
			if (_grad[i] < min) min = _grad[i];
			moyenne += fabs(_grad[i]);
		}
		printf("[Grad:Min=%+f;Max=%+f;Moyenne=(-/+)%f]",
			min, max, moyenne / POIDS
		);
		free(_grad);
		//
		//	P
		//
		moyenne = 0;
		min=____p[0], max=____p[0];
		FOR(0, i, POIDS) {
			if (____p[i] > max) max = ____p[i];
			if (____p[i] < min) min = ____p[i];
			moyenne += fabs(____p[i]);
		}
		printf("[P:Min=%+f;Max=%+f;Moyenne=(-/+)%f]",
			min, max, moyenne / POIDS
		);
		free(____p);
		printf("  %s\n", nom_inst[mdl->insts[c]]);
	}
};