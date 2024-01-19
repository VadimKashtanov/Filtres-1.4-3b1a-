#include "marchee.cuh"

void nature0__direct (ema_int_t * ema_int) {
	//			-- Parametres --
	//			-- Assertions --
	//	-- Transformation des Parametres --
	//		-- Calcule de la Nature --
	FOR(0, i, FIN) ema_int->brute[i] = ema_int->ema[i];
};