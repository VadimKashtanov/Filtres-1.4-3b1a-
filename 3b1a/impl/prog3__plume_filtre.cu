#include "main.cuh"

#include "../impl_tmpl/tmpl_etc.cu"

/*
Utilisation :
	./prog0__plume_filtre mdl.bin bloque f_dans_bloque
*/

int main(int argc, char ** argv) {
	srand(0);
	cudaSetDevice(0);
	titre(" Charger tout ");  charger_tout();
	//
	if (argc == 4) {
		Mdl_t * mdl = ouvrire_mdl(argv[1]);
		//
		char cmd[1000];
		//
		uint depart = atoi(argv[2])*F_PAR_BLOQUES*N + atoi(argv[3])*N;
		snprintf(cmd, 1000, "python3 -c \"import matplotlib.pyplot as plt;plt.plot([%f,%f,%f,%f,%f,%f,%f,%f]);plt.show()\"",
			mdl->p[0][depart + 0],
			mdl->p[0][depart + 1],
			mdl->p[0][depart + 2],
			mdl->p[0][depart + 3],
			mdl->p[0][depart + 4],
			mdl->p[0][depart + 5],
			mdl->p[0][depart + 6],
			mdl->p[0][depart + 7]
		);
		//
		SYSTEM(cmd);
		liberer_mdl(mdl);
		//
	} else {
		ERR("./prog0__plume_filtre mdl.bin bloque f_dans_bloque")
	}
	liberer_tout();
};