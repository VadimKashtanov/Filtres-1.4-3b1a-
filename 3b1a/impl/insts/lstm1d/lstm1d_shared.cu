#include "lstm1d.cuh"

//(0) Ft[t] = logistic(x@Wf + Ct[t-1]@Uf + Bf)
//(1) It[t] = logistic(x@Wi + Ct[t-1]@Ui + Bi)
//(2) Ot[t] = logistic(x@Wo + Ct[t-1]@Uo + Bo)
//(3) Tt[t] = tanh    (x@Wt + Bt)
//(4) Ct[t] = Ft[t]*Ct[-1] + It[t]*Tt[t]
//(5) Ht[t] = Ot[t]*Ct[t]

#define BLOQUE_Y 8

#define K16 64

static __global__ void kerd_lstm1d__shared__0123(
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint t,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd)
{
	uint _y = threadIdx.x + blockIdx.x * blockDim.x;

	uint thx = threadIdx.x;
	uint thy = threadIdx.y;

	float * _x = x + t*X_vars + DEPART_x;
	//
	__shared__ float sF[BLOQUE_Y], sI[BLOQUE_Y], sO[BLOQUE_Y], sT[BLOQUE_Y];
	if (thy == 0) {
		sF[thx] = 0;
		sI[thx] = 0;
		sO[thx] = 0;
		sT[thx] = 0;
	}
	__syncthreads();
	//
	uint k;
	if (_y < Y) {
		//	--- x@W ---
		FOR(0, _k, X/K16) {
			k = _k*K16 + thy;
			float __x = _x[k];
			atomicAdd(&sF[thx], __x * p[Wf+_y*X+k]);
			atomicAdd(&sI[thx], __x * p[Wi+_y*X+k]);
			atomicAdd(&sO[thx], __x * p[Wo+_y*X+k]);
			atomicAdd(&sT[thx], __x * p[Wt+_y*X+k]);
		}
		//	--- Ct[t-1]@U ---
		FOR(0, _k, Y/K16) {
			k = _k*K16 + thy;
			float __c = y[Ct-Y_vars+k];
			atomicAdd(&sF[thx], __c * p[Uf+_y*Y+k]);
			atomicAdd(&sI[thx], __c * p[Ui+_y*Y+k]);
			atomicAdd(&sO[thx], __c * p[Uo+_y*Y+k]);
		}
		__syncthreads();

		if (thy == 0) {
			y[Ft+_y] = sF[thx];
			y[It+_y] = sI[thx];
			y[Ot+_y] = sO[thx];
			y[Tt+_y] = sT[thx];
		}
	}
};

static __global__ void kerd_lstm1d__shared__45(
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint t,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd)
{
	uint _y = threadIdx.x + blockIdx.x * blockDim.x;
	//
	if (_y < Y) {
		float ft=y[Ft+_y]+p[Bf+_y], it=y[It+_y]+p[Bi+_y], ot=y[Ot+_y]+p[Bo+_y], tt=y[Tt+_y]+p[Bt+_y];
		//
		ft = 1 / (1 + expf(-ft));
		it = 1 / (1 + expf(-it));
		ot = 1 / (1 + expf(-ot));
		tt = tanh(tt);
		//
		float __Ct = ft*y[Ct-Y_vars+_y] + it*tt;
		y[Ct+_y] = __Ct;
		y[Ht+_y] = __Ct * ot;
	}
};

void nvidia_lstm1d__shared(
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint t,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd)
{
	ASSERT(X  % K16 == 0);
	ASSERT(Y  % K16 == 0);
	kerd_lstm1d__shared__0123<<<dim3(KERD(Y, BLOQUE_Y)), dim3(BLOQUE_Y, K16)>>>(
		X_vars, Y_vars,
		X, Y,
		t,
		DEPART_x,
		x, y,
		p,
		locd);
	ATTENDRE_CUDA();
	kerd_lstm1d__shared__45<<<dim3(KERD(Y, BLOQUE_Y)), dim3(BLOQUE_Y)>>>(
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

static __global__ void deriv_kerd_lstm1d__shared__45(
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
	//
	if (_y < Y) {
		/*float ft=y[Ft+_y]+p[Bf+_y], it=y[It+_y]+p[Bi+_y], ot=y[Ot+_y]+p[Bo+_y], tt=y[Tt+_y]+p[Bt+_y];
		//
		ft = 1 / (1 + expf(-ft));
		it = 1 / (1 + expf(-it));
		ot = 1 / (1 + expf(-ot));
		tt = tanh(tt);
		//
		float __Ct = ft*y[Ct-Y_vars+_y] + it*tt;
		y[Ct+_y] = __Ct;
		y[Ht+_y] = __Ct * ot;*/
		float ft=y[Ft+_y], it=y[It+_y], ot=y[Ot+_y], tt=y[Tt+_y];
		float __Ct = y[Ct+_y];
		float __Ct1 = y[Ct-Y_vars+_y];
		float dht = dy[Ht+_y];
		float d__Ct = dht * ot;
		float d__ot = dht * __Ct;
		//
		float d__ft = d__Ct * __Ct1;
		//float d__Ct1 = d__Ct * ft;
		float d__it = d__Ct * tt;
		float d__tt = d__Ct * it;
		dy[Ft+_y] += d__ft;
		dy[It+_y] += d__it;
		dy[Ot+_y] += d__ot;
		dy[Tt+_y] += d__tt;
		//
		float dsF = d__ft * (ft*(1-ft));
		float dsI = d__it * (it*(1-it));
		float dsO = d__ot * (ot*(1-ot));
		float dsT = d__tt * (1-tt*tt);
		//
		p[Bf+_y] += dsF;
		p[Bi+_y] += dsI;
		p[Bo+_y] += dsO;
		p[Bt+_y] += dsT;
	}
};

static __global__ void deriv_kerd_lstm1d__shared__0123(
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

	uint thx = threadIdx.x;
	uint thy = threadIdx.y;

	float * _x = x + t*X_vars + DEPART_x;
	float * _dx = dx + t*X_vars + DEPART_x;
	//
	__shared__ float dsF[BLOQUE_Y], dsI[BLOQUE_Y], dsO[BLOQUE_Y], dsT[BLOQUE_Y];
	//
	uint k;
	if (_y < Y) {
		if (thy == 0) {
			float ft=y[Ft+_y], it=y[It+_y], ot=y[Ot+_y], tt=y[Tt+_y];
			//
			float d__ft = dy[Ft+_y];
			float d__it = dy[It+_y];
			float d__ot = dy[Ot+_y];
			float d__tt = dy[Tt+_y];
			//
			dsF[thx] = d__ft * (ft*(1-ft));
			dsI[thx] = d__it * (it*(1-it));
			dsO[thx] = d__ot * (ot*(1-ot));
			dsT[thx] = d__tt * (1-tt*tt);
		}
		__syncthreads();
		//	--- x@W ---
		FOR(0, _k, X/K16) {
			k = _k*K16 + thy;
			/*float __x = _x[k];
			atomicAdd(&sF[thx], __x * p[Wf+_y*X+k]);
			atomicAdd(&sI[thx], __x * p[Wi+_y*X+k]);
			atomicAdd(&sO[thx], __x * p[Wo+_y*X+k]);
			atomicAdd(&sT[thx], __x * p[Wt+_y*X+k]);*/
			float d__x = 0;
			float __x = _x[k];
	//		sF += __x * Wf[_y*X+k];
			d__x += dsF[thx] * p[Wf+_y*X+k];
			atomicAdd(&dp[Wf+_y*X+k], dsF[thx] * __x);
	//		sI += __x * Wi[_y*X+k];
			d__x += dsI[thx] * p[Wi+_y*X+k];
			atomicAdd(&dp[Wi+_y*X+k], dsI[thx] * __x);
	//		sO += __x * Wo[_y*X+k];
			d__x += dsO[thx] * p[Wo+_y*X+k];
			atomicAdd(&dp[Wo+_y*X+k], dsO[thx] * __x);
	//		sT += __x * Wt[_y*X+k];
			d__x += dsT[thx] * p[Wt+_y*X+k];
			atomicAdd(&dp[Wt+_y*X+k], dsT[thx] * __x);
			//
			atomicAdd(&_dx[k], d__x);
		}
		//	--- Ct[t-1]@U ---
		FOR(0, _k, Y/K16) {
			k = _k*K16 + thy;
			/*float __c = y[Ct-Y_vars+k];
			atomicAdd(&sF[thx], __c * p[Uf+_y*Y+k]);
			atomicAdd(&sI[thx], __c * p[Ui+_y*Y+k]);
			atomicAdd(&sO[thx], __c * p[Uo+_y*Y+k]);*/
			//
			float d__c = 0;
			float __c = y[Ct-Y_vars+k];
	//		sF += __c * Uf[_y*Y+k];
			d__c += dsF[thx] * p[Uf+_y*Y+k];
			atomicAdd(&dp[Uf+_y*Y+k], dsF[thx] * __c);
	//		sI += __c * Ui[_y*Y+k];
			d__c += dsI[thx] * p[Ui+_y*Y+k];
			atomicAdd(&dp[Ui+_y*Y+k], dsI[thx] * __c);
	//		sO += __c * Uo[_y*Y+k];
			d__c += dsO[thx] * p[Uo+_y*Y+k];
			atomicAdd(&dp[Uo+_y*Y+k], dsO[thx] * __c);
			//
			atomicAdd(&dy[Ct-Y_vars+k], d__c);
		}
		//__syncthreads();
	}
};

void d_nvidia_lstm1d__shared(
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
	ASSERT(X  % K16 == 0);
	ASSERT(Y  % K16 == 0);
	deriv_kerd_lstm1d__shared__45<<<dim3(KERD(Y, BLOQUE_Y)), dim3(BLOQUE_Y)>>>(
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
	deriv_kerd_lstm1d__shared__0123<<<dim3(KERD(Y, BLOQUE_Y)), dim3(BLOQUE_Y, K16)>>>(
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