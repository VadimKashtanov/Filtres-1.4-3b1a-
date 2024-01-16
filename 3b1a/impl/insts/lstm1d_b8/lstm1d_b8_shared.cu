#include "lstm1d_b8.cuh"

//(0) Ft[t] = logistic(x@Wf + Ct[t-1]@Uf + Bf)
//(1) It[t] = logistic(x@Wi + Ct[t-1]@Ui + Bi)
//(2) Ot[t] = logistic(x@Wo + Ct[t-1]@Uo + Bo)
//(3) Tt[t] = tanh    (x@Wt + Bt)
//(4) Ct[t] = Ft[t]*Ct[-1] + It[t]*Tt[t]
//(5) Ht[t] = Ot[t]*Ct[t]

#define BLOQUE_Y 8

#define K16 64

/*static __global__ void kerd_lstm1d_b8__shared__0123(
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint t,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd)
{
	
};

static __global__ void kerd_lstm1d_b8__shared__45(
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint t,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd)
{
	
};*/

void nvidia_lstm1d_b8__shared(
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint t,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd)
{
	TODO()
};

//	===========================================================================
//	===========================================================================
//	===========================================================================

/*static __global__ void deriv_kerd_lstm1d_b8__shared__45(
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
	
};

static __global__ void deriv_kerd_lstm1d_b8__shared__0123(
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
	
};*/

void d_nvidia_lstm1d_b8__shared(
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
	TODO()
};