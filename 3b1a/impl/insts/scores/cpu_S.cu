#include "S.cuh"

#include "../../../impl_tmpl/tmpl_etc.cu"

float  intel_score(float * y, uint depart, uint T) {
	float score = 0;
	FOR(0, t, T) {
		//#pragma unroll
		FOR(0, p, P)
			score += (P-p)*SCORE(y[(depart+t)*P+p], prixs[depart+t+p+1], prixs[depart+t/*+p*/]);
	}
	return score / (float)(P * T);
};

float* intel_prediction(float * y, uint depart, uint T) {
	float * pourcent = (float*)calloc(P, sizeof(float));
	//
	FOR(0, i, T) {
		FOR(0, p, P) {
			if (signe(y[(depart+i)*P+p]) == signe(prixs[depart+i+p+1]/prixs[depart+i/*+p*/]-1)) {
				pourcent[p] += 1.0;
			}
		}
	}
	//
	FOR(0, p, P)
		pourcent[p] /= (float)T;
	//
	return pourcent;
};

void d_intel_score(float * y, float * dy, uint depart, uint T) {
	FOR(0, t, T) {
		//#pragma unroll
		FOR(0, p, P) {
			dy[(depart+t)*P+p] = (P-p)*dSCORE(y[(depart+t)*P+p], prixs[depart+t+p+1], prixs[depart+t/*+p*/]) / (T*P);
		}
	}
};