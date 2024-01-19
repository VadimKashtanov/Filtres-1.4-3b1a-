#include "marchee.cuh"

void nature1__macd(ema_int_t * ema_int) {
	//			-- Parametres --
	uint coef = ema_int->params[0];
	//			-- Assertions --
	ASSERT(min_param[MACD][0] <= coef && coef <= max_param[MACD][0]);
	//	-- Transformation des Parametres --
	float _coef = coef;
	//		-- Calcule de la Nature --
	_outil_macd(ema_int->brute, ema_int->ema, _coef);
};