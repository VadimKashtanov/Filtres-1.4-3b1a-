#include "opti.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

static __global__ void kerd_opti_simple(
	float * p, float * dp, float alpha, uint POIDS, float div)
{
	uint thx = threadIdx.x + blockIdx.x * blockDim.x;

	if (thx < POIDS) {
		p[thx] -= alpha * dp[thx] / div;
	}
};

static __global__ void kerd_opti_simple_masque(
	float * p, float * dp, float alpha, uint POIDS, float div, uint * masque)
{
	uint thx = threadIdx.x + blockIdx.x * blockDim.x;

	if (thx < POIDS) {
		if (masque[thx] == NON_MASQUEE)
			p[thx] -= alpha * dp[thx] / div;
	}
};

void opti_simple(Mdl_t * mdl, float * alpha, float div, uint ** masque) {
	//	Filtres
	uint FILTRES = mdl->Y[0];	//pas de *N, car c'est le filtre qu'on ignore, pas les points
	if (masque == 0) {
		kerd_opti_simple<<<dim3(KERD(FILTRES, 256)), dim3(256)>>>(
			mdl->p__d[0], mdl->dp__d[0], alpha[0], FILTRES, div);
	} else {
		kerd_opti_simple_masque<<<dim3(KERD(FILTRES, 256)), dim3(256)>>>(
			mdl->p__d[0], mdl->dp__d[0], alpha[0], FILTRES, div, masque[0]
		);
	}
	//	Poids
	FOR(1, c, C) {
		uint POIDS = mdl->inst_POIDS[c];
		if (masque == 0) {
			kerd_opti_simple<<<dim3(KERD(POIDS, 1024)), dim3(1024)>>>(
				mdl->p__d[c], mdl->dp__d[c], alpha[c], POIDS, div
			);
		} else {
			kerd_opti_simple_masque<<<dim3(KERD(POIDS, 1024)), dim3(1024)>>>(
				mdl->p__d[c], mdl->dp__d[c], alpha[c], POIDS, div, masque[c]
			);
		}
	};
	ATTENDRE_CUDA();
	mdl_gpu_vers_cpu(mdl);
};