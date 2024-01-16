#pragma once

#include "marchee.cuh"

//	Aleatoire

static float sng(float x) {
	return (x>=0) ? 1.0 : -1.0;
};

static float SCORE(float y, float p1, float p0) {
	return powf((y) - sng(p1/p0 - 1), 2)/2;
	//return powf((y) - sng(p1/p0 - 1), 4)/4;
};

static float dSCORE(float y, float p1, float p0) {
	return (y) - sng(p1/p0 - 1);
	//return powf((y) - sng(p1/p0 - 1), 3);
};


static __device__ float cuda_sng(float x) {
	return x>=0 ? 1.0 : -1.0;
};

static __device__ float cuda_SCORE(float y, float p1, float p0) {
	return powf((y) - cuda_sng(p1/p0 - 1), 2)/2;
	//return powf((y) - cuda_sng(p1/p0 - 1), 4)/4;
};

static __device__ float cuda_dSCORE(float y, float p1, float p0) {
	return (y) - cuda_sng(p1/p0 - 1);
	//return powf((y) - cuda_sng(p1/p0 - 1), 3);
};

/*	Score	*/

float  intel_score(float * y, uint depart, uint T);
float nvidia_score(float * y, uint depart, uint T);

float* intel_prediction(float * y, uint depart, uint T);
float* nvidia_prediction(float * y, uint depart, uint T);

void  d_intel_score(float * y, float * dy, uint depart, uint T);
void d_nvidia_score(float * y, float * dy, uint depart, uint T);

void verifier_S();