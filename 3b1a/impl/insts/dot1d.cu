#include "dot1d.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

void cree_dot1d(Mdl_t * mdl, uint c)
{
	uint X=mdl->Y[c-1], Y=mdl->Y[c];
	mdl->inst_POIDS        [c] = (mdl->Y[c-1]+1)*mdl->Y[c];
	mdl->inst_VARS         [c] = mdl->Y[c];
	mdl->inst_LOCDS        [c] = mdl->Y[c];
	mdl->inst_SORTIES      [c] = mdl->Y[c];
	mdl->inst_DEPART_SORTIE[c] = mdl->Y[c] - mdl->Y[c];
	//
	mdl->p[c] = alloc<float>(mdl->inst_POIDS[c]);
	//
	FOR(0, i, mdl->inst_POIDS[c])
		mdl->p[c][i] = (2*rnd()-1) * 0.01;
	//
	FOR(0, y, Y) {
		uint CONNECTIONS = 4 + (rand()%10);
		FOR(0, i, 8)
			mdl->p[c][y*(X+1) + (rand()%X)] = (2*rnd()-1);
		mdl->p[c][y*(X+1)+(X-1)+1] = (2*rnd()-1);
	};
};

//	::::: Normer la sortie ?? (non)

//	supprimer EMA_INTS
//		et ajouter BLOQUES outils pour chaque bloque (si un est deja calculer ne pas recalculer evidement)
//		chaque outil a des parametres (macd, chiffre, ema, interv, delta)

//	La selection naturelle se fera sur ces parametres (et aleatoire forcé)

//r = ema([x for x in prixs[-500:]], 10);dr = ema([0] + [r[i+1]-r[i] for i in range(len(r)-1)], 10); ddr = [0,0] + [dr[i+1]-dr[i] for i in range(len(dr)-1)]


void plume_dot1d(Mdl_t * mdl, uint c)
{
	printf("POIDS DOT1D: \n");
	uint X=mdl->Y[c-1], Y=mdl->Y[c];
	FOR(0, y, Y) {
		printf("y=%i : ", y);
		FOR(0, x, X) {
			printf("%+f,", mdl->p[c][y*(X+1)+x]);
		}
		printf(" biais=%+f\n", mdl->p[c][y*(X+1)+X+1-1]);
	}
};

void intel_dot1d(
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd)
{
#pragma omp parallel
#pragma omp for
	FOR(0, t, T) {
		FOR(0, _y, Y) {
			float s = p[_y*(X+1)+(X+1-1)];
			FOR(0, k, X) {
				float __x = x[(depart+t)*X_vars+DEPART_x+k];
				float __p = p[_y*(X+1)+k];
				s += __x * __p;
			}
			float a = ACTIV(ACTIVATION, s);
			y[(depart+t)*Y+_y]    = a;
			locd[(depart+t)*Y+_y] = dACTIV(ACTIVATION, s, a);
		}
	}
}

void d_intel_dot1d(
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd,
	float * dy,
	float * dx,
	float * dp)
{
//#pragma omp parallel
//#pragma omp for
/*	FOR(0, t, T) {
		FOR(0, _y, Y) {
			float _locd = locd[(depart+t)*Y+_y] * dy[(depart+t)*Y+_y];
			dp[_y*(X+1)+(X+1-1)] += _locd;
			FOR(0, k, X) {
				//s += x[t*X+k] * p[y*(X+1)+k];
				dx[(depart+t)*X+k]      += _locd * p[_y*(X+1)+k];
				dp[_y*(X+1)+k] += _locd * x[(depart+t)*X+k];
			}
		}
	}*/

	//dx = (p @ ((y-_y)*dtanh(x@p)).T).T
#pragma omp parallel
#pragma omp for
	FOR(0, t, T) {
		FOR(0, _x, X) {
			//float _locd = locd[(depart+t)*Y+_y] * dy[(depart+t)*Y+_y];
			float s = 0;
			FOR(0, k, Y) {
				float __x = p[k*(X+1)+_x];//x[(depart+t)*X+k];
				float __p = locd[(depart+t)*Y+k] * dy[(depart+t)*Y+k];//p[_y*(X+1)+k];
				s += __x * __p;
			}
			dx[(depart+t)*X_vars+DEPART_x+_x]   = s;
		}
	}

	//dp = x.T @ ((y-_y)*dtanh(x@p))
#pragma omp parallel
#pragma omp for
	FOR(0, _y, Y) {
		float dbiais = 0;
		FOR(0, _x, X) {
			float s = 0;
			FOR(0, t, T) {
				float __x = locd[(depart+t)*Y+_y] * dy[(depart+t)*Y+_y];//x[(depart+t)*X+k];
				float __p = x[(depart+t)*X_vars+DEPART_x+_x];//p[_y*(X+1)+k];
				s += __x * __p;
				if (_x == 0) {	//	Biais
					dbiais += __x;
				}
			}
			dp[_y*(X+1)+_x] = s;
		}
		dp[_y*(X+1)+(X+1-1)] = dbiais;
	}
}

//	=========================================================

void f_dot1d(Mdl_t * mdl, uint inst, uint mode, uint t0, uint t1) {
	uint X=mdl->Y[inst-1], Y=mdl->Y[inst];
	uint X_vars=mdl->inst_VARS[inst-1], Y_vars=mdl->inst_VARS[inst];
	uint depart = t0;
	uint T = (t1-t0);
	uint DEPART_x = mdl->inst_DEPART_SORTIE[inst-1];
	if (mode == 0) {
		intel_dot1d(
			X_vars, Y_vars,
			X, Y,
			depart, T,
			DEPART_x,
			mdl->y[inst-1], mdl->y[inst],
			mdl->p[inst],
			mdl->l[inst]);
	} else if (mode == 1) {
		nvidia_dot1d_naive(
			X_vars, Y_vars,
			X, Y,
			depart, T,
			DEPART_x,
			mdl->y__d[inst-1], mdl->y__d[inst],
			mdl->p__d[inst],
			mdl->l__d[inst]);
	} else if (mode == 2) {
		nvidia_dot1d_shared(
			X_vars, Y_vars,
			X, Y,
			depart, T,
			DEPART_x,
			mdl->y__d[inst-1], mdl->y__d[inst],
			mdl->p__d[inst],
			mdl->l__d[inst]);
	} else if (mode == 3) {
		nvidia_dot1d_shared_2_16(
			X_vars, Y_vars,
			X, Y,
			depart, T,
			DEPART_x,
			mdl->y__d[inst-1], mdl->y__d[inst],
			mdl->p__d[inst],
			mdl->l__d[inst]);
	} else {
		ERR("Pas de mode %i pour cuda f(x)", mode);
	}
}

//	----------------------------

void df_dot1d(Mdl_t * mdl, uint inst, uint mode, uint t0, uint t1) {
	uint X=mdl->Y[inst-1], Y=mdl->Y[inst];
	uint X_vars=mdl->inst_VARS[inst-1], Y_vars=mdl->inst_VARS[inst];
	uint depart = t0;
	uint T = (t1-t0);
	uint DEPART_x = mdl->inst_DEPART_SORTIE[inst-1];
	if (mode == 0) {
		d_intel_dot1d(
			X_vars, Y_vars,
			X, Y,
			depart, T,
			DEPART_x,
			mdl->y[inst-1], mdl->y[inst],
			mdl->p[inst],
			mdl->l[inst],
			mdl->dy[inst],
			mdl->dy[inst-1],
			mdl->dp[inst]);
	} else if (mode == 1) {
		d_nvidia_dot1d_naive(
			X_vars, Y_vars,
			X, Y,
			depart, T,
			DEPART_x,
			mdl->y__d[inst-1], mdl->y__d[inst],
			mdl->p__d[inst],
			mdl->l__d[inst],
			mdl->dy__d[inst],
			mdl->dy__d[inst-1],
			mdl->dp__d[inst]);
	} else if (mode == 2) {
		d_nvidia_dot1d_shared(
			X_vars, Y_vars,
			X, Y,
			depart, T,
			DEPART_x,
			mdl->y__d[inst-1], mdl->y__d[inst],
			mdl->p__d[inst],
			mdl->l__d[inst],
			mdl->dy__d[inst],
			mdl->dy__d[inst-1],
			mdl->dp__d[inst]);
	} else if (mode == 3) {
		d_nvidia_dot1d_shared_2_16(
			X_vars, Y_vars,
			X, Y,
			depart, T,
			DEPART_x,
			mdl->y__d[inst-1], mdl->y__d[inst],
			mdl->p__d[inst],
			mdl->l__d[inst],
			mdl->dy__d[inst],
			mdl->dy__d[inst-1],
			mdl->dp__d[inst]);
	} else {
		ERR("Pas de mode %i pour cuda df(x)", mode);
	}
}