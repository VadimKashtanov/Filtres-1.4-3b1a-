#include "lstm1d_b8.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

//
//	//
//	//	Filtres LSTM 1D B8
//	//
//

void cree_lstm1d_b8(Mdl_t * mdl, uint c)
{
	TODO()
};

void plume_lstm1d_b8(Mdl_t * mdl, uint c)
{
	TODO()
};

void intel_lstm1d_b8(
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint t,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd)
{
	TODO()
}

void d_intel_lstm1d_b8(
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

//	=========================================================
/*__global__
static void kerd_cuda_memset_t(float * v, uint t, uint vars) {
	uint thx = threadIdx.x + blockIdx.x * blockDim.x;
	if (thx < vars) {
		v[t*vars + thx] = 0;
	}
};
static void cuda_memset_t(float * v, uint t, uint vars) {
	kerd_cuda_memset_t<<<dim3(KERD(vars,32)), dim3(32)>>>(v, t, vars);
	ATTENDRE_CUDA();
}*/

void f_lstm1d_b8(Mdl_t * mdl, uint inst, uint mode, uint t0, uint t1) {
	TODO()
}

//	----------------------------

void df_lstm1d_b8(Mdl_t * mdl, uint inst, uint mode, uint t0, uint t1) {
	TODO()
}