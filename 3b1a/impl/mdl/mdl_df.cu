#include "mdl.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

void mdl_df(Mdl_t * mdl, uint t0, uint t1, uint mode) {
	RETRO_FOR(0, c, C) {
		inst_df[mdl->insts[c]](mdl, c, mode, t0, t1);
	};
};