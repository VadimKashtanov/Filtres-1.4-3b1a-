#include "lstm1d.cuh"

//(0) Ft[t] = logistic(x@Wf + Ct[t-1]@Uf + Bf)
//(1) It[t] = logistic(x@Wi + Ct[t-1]@Ui + Bi)
//(2) Ot[t] = logistic(x@Wo + Ct[t-1]@Uo + Bo)
//(3) Tt[t] = tanh    (x@Wt + Bt)
//(4) Ct[t] = Ft[t]*Ct[-1] + It[t]*Tt[t]
//(5) Ht[t] = Ot[t]*Ct[t]

#define BLOQUE_Y 32

static __global__ void kerd_lstm1d_naive__0123(
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint t,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd)
{
	uint _y = threadIdx.x + blockIdx.x * blockDim.x;

	float * _x = x + t*X_vars + DEPART_x;

	if (_y < Y) {
		float sF = p[Bf+_y];
		float sI = p[Bi+_y];
		float sO = p[Bo+_y];
		float sT = p[Bt+_y];
		// --- x@W ---
		FOR(0, k, X) {
			float __x = _x[k];
			sF += __x * p[Wf+_y*X+k];
			sI += __x * p[Wi+_y*X+k];
			sO += __x * p[Wo+_y*X+k];
			sT += __x * p[Wt+_y*X+k];
		}
		// --- Ct[t-1]@U ---
		FOR(0, k, Y) {
			float __c = y[Ct-Y_vars+k];
			sF += __c * p[Uf+_y*Y+k];
			sI += __c * p[Ui+_y*Y+k];
			sO += __c * p[Uo+_y*Y+k];
		}
		// --- logistic && tanh ---
		y[Ft+_y] = 1 / (1 + expf(-sF));
		y[It+_y] = 1 / (1 + expf(-sI));
		y[Ot+_y] = 1 / (1 + expf(-sO));
		y[Tt+_y] = tanh(sT);
	}
};

static __global__ void kerd_lstm1d_naive__45(
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint t,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd)
{
	uint _y = threadIdx.x + blockIdx.x * blockDim.x;

	if (_y < Y) {
		y[Ct+_y] = y[Ft+_y] * y[Ct-Y_vars+_y] + y[It+_y]*y[Tt+_y];
		y[Ht+_y] = y[Ct+_y] * y[Ot+_y];
	}
};

void nvidia_lstm1d_naive(
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint t,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd)
{
	kerd_lstm1d_naive__0123<<<dim3(KERD(Y, BLOQUE_Y)), dim3(BLOQUE_Y)>>>(
		X_vars, Y_vars,
		X, Y,
		t,
		DEPART_x,
		x, y,
		p,
		locd);
	ATTENDRE_CUDA();
	kerd_lstm1d_naive__45<<<dim3(KERD(Y, BLOQUE_Y)), dim3(BLOQUE_Y)>>>(
		X_vars, Y_vars,
		X, Y,
		t,
		DEPART_x,
		x, y,
		p,
		locd);
	ATTENDRE_CUDA();
};

//	===========================================================================
//	===========================================================================
//	===========================================================================

static __global__ void deriv_kerd_lstm1d_naive__45(
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
	uint _y = threadIdx.x + blockIdx.x * blockDim.x;

	if (_y < Y) {
		//Ht[_y] = Ct[_y] * Ot[_y];
		dy[Ct+_y] += dy[Ht+_y] * y[Ot+_y];
		dy[Ot+_y] += dy[Ht+_y] * y[Ct+_y];

		//Ct[_y] = Ft[_y]*Ct[_y - 1*(6*Y)] + It[_y]*Tt[_y];
		dy[Ft +_y]  += dy[Ct+_y] * y[Ct-Y_vars+_y];
		dy[Ct-Y_vars+_y] += dy[Ct+_y] * y[Ft+_y];
		//
		dy[It +_y] += dy[Ct+_y] * y[Tt+_y];
		dy[Tt +_y] += dy[Ct+_y] * y[It+_y];
	}
};

static __global__ void deriv_kerd_lstm1d_naive__0123(
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
	uint _y = threadIdx.x + blockIdx.x * blockDim.x;

	float *  _x =  x + t*X_vars + DEPART_x;
	float * _dx = dx + t*X_vars + DEPART_x;

	if (_y < Y) {
		float dsF = dy[Ft+_y] * (y[Ft+_y] * (1 - y[Ft+_y]));
		float dsI = dy[It+_y] * (y[It+_y] * (1 - y[It+_y]));
		float dsO = dy[Ot+_y] * (y[Ot+_y] * (1 - y[Ot+_y]));
		float dsT = dy[Tt+_y] * (  1 - y[Tt+_y]*y[Tt+_y]  );
		//	--- Ct[t-1]@U ---
		FOR(0, k, Y) {
			float d__c = 0;
			float __c = y[Ct-Y_vars+k];
	//		sF += __c * Uf[_y*Y+k];
			d__c += dsF * p[Uf+_y*Y+k];
			atomicAdd(&dp[Uf+_y*Y+k], dsF * __c);
	//		sI += __c * Ui[_y*Y+k];
			d__c += dsI * p[Ui+_y*Y+k];
			atomicAdd(&dp[Ui+_y*Y+k], dsI * __c);
	//		sO += __c * Uo[_y*Y+k];
			d__c += dsO * p[Uo+_y*Y+k];
			atomicAdd(&dp[Uo+_y*Y+k], dsO * __c);
			//
			atomicAdd(&dy[Ct-Y_vars+k], d__c);
		}
		//	--- x@W ---
		FOR(0, k, X) {
			float d__x = 0;
			float __x = _x[k];
	//		sF += __x * Wf[_y*X+k];
			d__x += dsF * p[Wf+_y*X+k];
			atomicAdd(&dp[Wf+_y*X+k], dsF * __x);
	//		sI += __x * Wi[_y*X+k];
			d__x += dsI * p[Wi+_y*X+k];
			atomicAdd(&dp[Wi+_y*X+k], dsI * __x);
	//		sO += __x * Wo[_y*X+k];
			d__x += dsO * p[Wo+_y*X+k];
			atomicAdd(&dp[Wo+_y*X+k], dsO * __x);
	//		sT += __x * Wt[_y*X+k];
			d__x += dsT * p[Wt+_y*X+k];
			atomicAdd(&dp[Wt+_y*X+k], dsT * __x);
			//
			atomicAdd(&_dx[k], d__x);
		}
		//
	//	float sF=Bf[_y], sI=Bi[_y], sO=Bo[_y], sT=Bt[_y];
		atomicAdd(&dp[Bf+_y], dsF);
		atomicAdd(&dp[Bi+_y], dsI);
		atomicAdd(&dp[Bo+_y], dsO);
		atomicAdd(&dp[Bt+_y], dsT);
	}
};

void d_nvidia_lstm1d_naive(
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
	deriv_kerd_lstm1d_naive__45<<<dim3(KERD(Y, BLOQUE_Y)), dim3(BLOQUE_Y)>>>(
		X_vars, Y_vars,
		X, Y,
		t,
		DEPART_x,
		x, y,
		p,
		locd,
		dy,
		dx,
		dp);
	ATTENDRE_CUDA();
	deriv_kerd_lstm1d_naive__0123<<<dim3(KERD(Y, BLOQUE_Y)), dim3(BLOQUE_Y)>>>(
		X_vars, Y_vars,
		X, Y,
		t,
		DEPART_x,
		x, y,
		p,
		locd,
		dy,
		dx,
		dp);
	ATTENDRE_CUDA();
};