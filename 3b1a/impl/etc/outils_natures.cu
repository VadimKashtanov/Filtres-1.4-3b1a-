#include "marchee.cuh"

void _outil_ema(float * y, float * x, uint K) {
	float _K = 1 / ((float)K);
	y[0] = x[0];
	FOR(1, t, FIN) {
		y[t] = y[t-1]*(1 - _K) + x[t] * _K;
	}
};

void _outil_macd(float * y, float * x, float coef) {
	ASSERT(coef > 0.0);
	float ema12[PRIXS], ema26[PRIXS], ema9[PRIXS], __macd[PRIXS];
	_outil_ema(ema12, x, 12*coef);
	_outil_ema(ema26, x, 26*coef);
	FOR(0, i, PRIXS) __macd[i] = ema12[i] - ema26[i];
	_outil_ema(ema9, __macd, 12*coef);
	FOR(0, i, PRIXS) y[i] = __macd[i] - ema9[i];
};

void _outil_chiffre(float * y, float * x, float chiffre) {
	FOR(0, t, PRIXS) {
		y[t] = 2*(chiffre-MIN2(fabs(x[t]-chiffre*roundf((x[t]+0)/chiffre)), fabs(x[t]-chiffre*roundf((x[t]+chiffre)/chiffre))))/chiffre-1;
	}
};

void _outil_dx(float * y, float * x, uint plus) {
	FOR(0, t, plus+1) y[t] = 0;
	FOR(plus+1, t, FIN) {
		y[t] = x[t] - x[t-1-plus];
	}
};