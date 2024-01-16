#include "main.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

float rnd()
{
#define PROFONDEURE 10000
	return (float)(rand() % PROFONDEURE) / (float)PROFONDEURE;
}

float signe(float x)
{
	return (x>=0 ? 1:-1);
};

double secondes()
{
	struct timespec now;
	timespec_get(&now, TIME_UTC);
	return 1000.0*(((int64_t) now.tv_sec) * 1000 + ((int64_t) now.tv_nsec) / 1000000);
};

PAS_OPTIMISER()
void titre(char * str) {
	printf("\033[93m=========\033[0m %s \033[93m=========\033[0m\n", str);
};