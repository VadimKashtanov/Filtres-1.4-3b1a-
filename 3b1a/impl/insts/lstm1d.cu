#include "lstm1d.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

void cree_lstm1d(Mdl_t * mdl, uint c)
{
	uint X=mdl->Y[c-1], Y=mdl->Y[c];
	mdl->inst_POIDS        [c] = 3*(X*Y + Y*Y + Y) + 1*(X*Y + Y);
	mdl->inst_VARS         [c] = mdl->Y[c]*6;	//Ft, It, Ot, Tt, Ct, Ht
	mdl->inst_LOCDS        [c] = 0;
	mdl->inst_SORTIES      [c] = mdl->Y[c];		//Ht
	mdl->inst_DEPART_SORTIE[c] = mdl->inst_VARS[c] - mdl->inst_SORTIES[c];
	//
	mdl->p[c] = alloc<float>(mdl->inst_POIDS[c]);
	FOR(0, i, mdl->inst_POIDS[c])
		mdl->p[c][i] = (2*rnd()-1) * sqrtf(2.0 / (float)X);
};

void plume_lstm1d(Mdl_t * mdl, uint c)
{
	printf("POIDS LSTM: \n");
	uint X=mdl->Y[c-1], Y=mdl->Y[c];
	//
	float * p = mdl->p[c];
	//
	printf("Wf : "); FOR(0, i, X*Y) {printf("%+f,", p[Wf+i]);}; printf("\n");
	printf("Wi : "); FOR(0, i, X*Y) {printf("%+f,", p[Wi+i]);}; printf("\n");
	printf("Wo : "); FOR(0, i, X*Y) {printf("%+f,", p[Wo+i]);}; printf("\n");
	//
	printf("Uf : "); FOR(0, i, Y*Y) {printf("%+f,", p[Uf+i]);}; printf("\n");
	printf("Ui : "); FOR(0, i, Y*Y) {printf("%+f,", p[Ui+i]);}; printf("\n");
	printf("Uo : "); FOR(0, i, Y*Y) {printf("%+f,", p[Uo+i]);}; printf("\n");
	//
	printf("Bf : "); FOR(0, i,   Y) {printf("%+f,", p[Bf+i]);}; printf("\n");
	printf("Bi : "); FOR(0, i,   Y) {printf("%+f,", p[Bi+i]);}; printf("\n");
	printf("Bo : "); FOR(0, i,   Y) {printf("%+f,", p[Bo+i]);}; printf("\n");
	//
	printf("Wt : "); FOR(0, i, X*Y) {printf("%+f,", p[Wt+i]);}; printf("\n");
	printf("Bt : "); FOR(0, i,   Y) {printf("%+f,", p[Bt+i]);}; printf("\n");
};

void intel_lstm1d(
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint t,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd)
{
	float * _x = x + t*X_vars + DEPART_x;
	//(0) Ft[t] = logistic(x@Wf + Ct[t-1]@Uf + Bf)
	//(1) It[t] = logistic(x@Wi + Ct[t-1]@Ui + Bi)
	//(2) Ot[t] = logistic(x@Wo + Ct[t-1]@Uo + Bo)
	//(3) Tt[t] = tanh    (x@Wt + Bt)
	//(4) Ct[t] = Ft[t]*Ct[-1] + It[t]*Tt[t]
	//(5) Ht[t] = Ot[t]*Ct[t]
	FOR(0, _y, Y) {
		float sF=p[Bf+_y];
		float sI=p[Bi+_y];
		float sO=p[Bo+_y];
		float sT=p[Bt+_y];
		//	--- x@W ---
		FOR(0, k, X) {
			float __x = _x[k];
			sF += __x * p[Wf+_y*X+k];
			sI += __x * p[Wi+_y*X+k];
			sO += __x * p[Wo+_y*X+k];
			sT += __x * p[Wt+_y*X+k];
		}
		//	--- Ct[t-1]@U ---
		FOR(0, k, Y) {
			float __c = y[Ct-Y_vars+k];
			sF += __c * p[Uf+_y*Y+k];
			sI += __c * p[Ui+_y*Y+k];
			sO += __c * p[Uo+_y*Y+k];
		}
		//	--- logistic && tanh ---
		y[Ft+_y] = 1 / (1 + expf(-sF));
		y[It+_y] = 1 / (1 + expf(-sI));
		y[Ot+_y] = 1 / (1 + expf(-sO));
		y[Tt+_y] = tanh(sT);
	}
	FOR(0, _y, Y) {
		y[Ct+_y] = y[Ft+_y] * y[Ct-1*Y_vars+_y] + y[It+_y]*y[Tt+_y];
		y[Ht+_y] = y[Ct+_y] * y[Ot+_y];
	};
}

void d_intel_lstm1d(
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint t,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd,
	float * dy,
	float * dx,
	float * dp)
{
	float * _x  = x  + t*X_vars + DEPART_x;
	float * _dx = dx + t*X_vars + DEPART_x;

	FOR(0, _y, Y) {
		dy[Ct+_y] += y[Ot+_y] * dy[Ht+_y];
		dy[Ot+_y] += y[Ct+_y] * dy[Ht+_y];
		//
		dy[Ft+_y] += y[Ct-1*Y_vars+_y] * dy[Ct+_y];
		dy[Ct-1*Y_vars+_y] += y[Ft+_y] * dy[Ct+_y];
		dy[It+_y] += y[Tt+_y] * dy[Ct+_y];
		dy[Tt+_y] += y[It+_y] * dy[Ct+_y];
	}

	FOR(0, _y, Y) {
		//	--- logistic && tanh ---
	//	Ft[_y] = 1 / (1 + expf(-sF));
	//	It[_y] = 1 / (1 + expf(-sI));
	//	Ot[_y] = 1 / (1 + expf(-sO));
	//	Tt[_y] = tanh(sT);
		//
		float dsF = dy[Ft+_y] * (y[Ft+_y] * (1 - y[Ft+_y]));
		float dsI = dy[It+_y] * (y[It+_y] * (1 - y[It+_y]));
		float dsO = dy[Ot+_y] * (y[Ot+_y] * (1 - y[Ot+_y]));
		float dsT = dy[Tt+_y] * (  1 - y[Tt+_y]*y[Tt+_y]  );

		//	--- Ct[t-1]@U ---
		FOR(0, k, Y) {
			float d__c = 0;
			float __c = y[Ct-1*Y_vars+k];	//t-1
	//		sF += __c * Uf[_y*Y+k];
			d__c += dsF * p[Uf+_y*Y+k];
			dp[Uf+_y*Y+k] += dsF * __c;
	//		sI += __c * Ui[_y*Y+k];
			d__c += dsI * p[Ui+_y*Y+k];
			dp[Ui+_y*Y+k] += dsI * __c;
	//		sO += __c * Uo[_y*Y+k];
			d__c += dsO * p[Uo+_y*Y+k];
			dp[Uo+_y*Y+k] += dsO * __c;
			//
			dy[Ct-1*Y_vars+k] += d__c;
		}
		//	--- x@W ---
		FOR(0, k, X) {
	//		float __x = _x[k];
			float d__x = 0;
			float __x = _x[k];	//t-1
	//		sF += __x * Wf[_y*X+k];
			d__x += dsF * p[Wf+_y*X+k];
			dp[Wf+_y*X+k] += dsF * __x;
	//		sI += __x * Wi[_y*X+k];
			d__x += dsI * p[Wi+_y*X+k];
			dp[Wi+_y*X+k] += dsI * __x;
	//		sO += __x * Wo[_y*X+k];
			d__x += dsO * p[Wo+_y*X+k];
			dp[Wo+_y*X+k] += dsO * __x;
	//		sT += __x * Wt[_y*X+k];
			d__x += dsT * p[Wt+_y*X+k];
			dp[Wt+_y*X+k] += dsT * __x;
			//
			_dx[k] += d__x;
		}
		//
	//	float sF=Bf[_y], sI=Bi[_y], sO=Bo[_y], sT=Bt[_y];
		dp[Bf+_y] += dsF;
		dp[Bi+_y] += dsI;
		dp[Bo+_y] += dsO;
		dp[Bt+_y] += dsT;
	}
}

//	=========================================================
__global__
static void kerd_cuda_memset_t(float * v, uint t, uint vars) {
	uint thx = threadIdx.x + blockIdx.x * blockDim.x;
	if (thx < vars) {
		v[t*vars + thx] = 0;
	}
};
void cuda_memset_t(float * v, uint t, uint vars) {
	kerd_cuda_memset_t<<<dim3(KERD(vars,32)), dim3(32)>>>(v, t, vars);
	ATTENDRE_CUDA();
}

void f_lstm1d(Mdl_t * mdl, uint inst, uint mode, uint t0, uint t1) {
	uint X=mdl->Y[inst-1], Y=mdl->Y[inst];
	uint X_vars=mdl->inst_VARS[inst-1], Y_vars=mdl->inst_VARS[inst];
	uint DEPART_x = mdl->inst_DEPART_SORTIE[inst-1];
	//
	if (mode == 0) {
		memset(
			mdl->y[inst]+(t0-1)*mdl->inst_VARS[inst],
			0,
			sizeof(float)*mdl->inst_VARS[inst]
		);
		FOR(t0, t, t1) {
			intel_lstm1d(
				X_vars, Y_vars,
				X, Y,
				t,
				DEPART_x,
				mdl->y[inst-1], mdl->y[inst],
				mdl->p[inst],
				mdl->l[inst]);
		}
	} else if (mode == 1) {
		cuda_memset_t(
			mdl->y__d[inst],
			t0-1, mdl->inst_VARS[inst]
		);
		FOR(t0, t, t1) {
			nvidia_lstm1d_naive(
				X_vars, Y_vars,
				X, Y,
				t,
				DEPART_x,
				mdl->y__d[inst-1], mdl->y__d[inst],
				mdl->p__d[inst],
				mdl->l__d[inst]);
		}
	}  else if (mode == 2 || mode == 3) {
		cuda_memset_t(
			mdl->y__d[inst],
			t0-1,
			mdl->inst_VARS[inst]
		);
		FOR(t0, t, t1) {
			nvidia_lstm1d__shared(
				X_vars, Y_vars,
				X, Y,
				t,
				DEPART_x,
				mdl->y__d[inst-1], mdl->y__d[inst],
				mdl->p__d[inst],
				mdl->l__d[inst]);
		}
	} else {
		ERR("Pas de mode %i pour cuda f(x)", mode);
	}
}

//	----------------------------

void df_lstm1d(Mdl_t * mdl, uint inst, uint mode, uint t0, uint t1) {
	uint X=mdl->Y[inst-1], Y=mdl->Y[inst];
	uint X_vars=mdl->inst_VARS[inst-1], Y_vars=mdl->inst_VARS[inst];
	uint DEPART_x = mdl->inst_DEPART_SORTIE[inst-1];
	//
	if (mode == 0) {
		RETRO_FOR(t0, t, t1) {
			d_intel_lstm1d(
				X_vars, Y_vars,
				X, Y,
				t,
				DEPART_x,
				mdl->y[inst-1], mdl->y[inst],
				mdl->p[inst],
				mdl->l[inst],
				mdl->dy[inst],
				mdl->dy[inst-1],
				mdl->dp[inst]);
		}
	} else if (mode == 1) {
		RETRO_FOR(t0, t, t1) {
			d_nvidia_lstm1d_naive(
				X_vars, Y_vars,
				X, Y,
				t,
				DEPART_x,
				mdl->y__d[inst-1], mdl->y__d[inst],
				mdl->p__d[inst],
				mdl->l__d[inst],
				mdl->dy__d[inst],
				mdl->dy__d[inst-1],
				mdl->dp__d[inst]);
		}
	}  else if (mode == 2 || mode == 3) {
		RETRO_FOR(t0, t, t1) {
			d_nvidia_lstm1d__shared(
				X_vars, Y_vars,
				X, Y,
				t,
				DEPART_x,
				mdl->y__d[inst-1], mdl->y__d[inst],
				mdl->p__d[inst],
				mdl->l__d[inst],
				mdl->dy__d[inst],
				mdl->dy__d[inst-1],
				mdl->dp__d[inst]);
		}
	} else {
		ERR("Pas de mode %i pour cuda f(x)", mode);
	}
}