#include "opti.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

#define rms_alpha 0.9

static __global__ void kerd_opti_rmsprop(
	float * p, float * dp, float * g,
	float alpha, uint POIDS, float div)
{
	uint thx = threadIdx.x + blockIdx.x * blockDim.x;

	if (thx < POIDS) {
		float _grad = dp[thx] / div;
		float _g = rms_alpha*g[thx] + _grad*_grad;
		p[thx] -= alpha * _grad / (sqrtf(_g) + 1e-5);
		g[thx] = _g;
	}
};

static __global__ void kerd_opti_rmsprop_masque(
	float * p, float * dp, float * g,
	float alpha, uint POIDS, float div, uint * masque)
{
	uint thx = threadIdx.x + blockIdx.x * blockDim.x;

	if (thx < POIDS) {
		if (masque[thx] == NON_MASQUEE) {
			float _grad = dp[thx] / div;
			float _g = rms_alpha*g[thx] + _grad*_grad;
			p[thx] -= alpha * _grad / (sqrtf(_g) + 1e-5);
			g[thx] = _g;
		}
	}
};

Rmsprop_t * cree_rmsprop(
	Mdl_t * mdl)
{
	Rmsprop_t * ret = alloc<Rmsprop_t>(1);
	ret->g[0] = cudazero<float>(mdl->Y[0]*N);
	FOR(1, c, C) ret->g[c] = cudazero<float>(mdl->inst_POIDS[c]);
	return ret;
};

void liberer_rmsprop(Rmsprop_t * rmsprop) {
	FOR(0, c, C) cudafree<float>(rmsprop->g[c]);
	free(rmsprop);
};

void opti_rmsprop(
	Mdl_t * mdl, Rmsprop_t * rmsprop,
	float * alpha, float div, uint ** masque)
{
	//	Filtres
	uint FILTRES = mdl->Y[0];	//pas de *N, car c'est le filtre qu'on ignore, pas les points
	if (masque == 0) {
		kerd_opti_rmsprop<<<dim3(KERD(FILTRES, 256)), dim3(256)>>>(
			mdl->p__d[0], mdl->dp__d[0], rmsprop->g[0], alpha[0], FILTRES, div);
	} else {
		kerd_opti_rmsprop_masque<<<dim3(KERD(FILTRES, 256)), dim3(256)>>>(
			mdl->p__d[0], mdl->dp__d[0], rmsprop->g[0], alpha[0], FILTRES, div, masque[0]
		);
	}
	//	Poids
	FOR(1, c, C) {
		uint POIDS = mdl->inst_POIDS[c];
		
		if (masque == 0) {
			kerd_opti_rmsprop<<<dim3(KERD(POIDS, 1024)), dim3(1024)>>>(
				mdl->p__d[c], mdl->dp__d[c], rmsprop->g[c],
				alpha[c], POIDS, div
			);
		} else {
			kerd_opti_rmsprop_masque<<<dim3(KERD(POIDS, 1024)), dim3(1024)>>>(
				mdl->p__d[c], mdl->dp__d[c], rmsprop->g[c],
				alpha[c], POIDS, div, masque[c]
			);
		}
	};
	ATTENDRE_CUDA();
	mdl_gpu_vers_cpu(mdl);
};