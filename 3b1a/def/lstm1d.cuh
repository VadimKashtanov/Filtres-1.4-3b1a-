#pragma once

#include "mdl.cuh"

void cree_lstm1d(Mdl_t * mdl, uint inst);
void plume_lstm1d(Mdl_t * mdl, uint c);

#define Ft (t*(6*Y) + 0*Y)
#define It (t*(6*Y) + 1*Y)
#define Ot (t*(6*Y) + 2*Y)
#define Tt (t*(6*Y) + 3*Y)
#define Ct (t*(6*Y) + 4*Y)
#define Ht (t*(6*Y) + 5*Y)
	//
#define Wf ((0) + 0*X*Y)
#define Wi ((0) + 1*X*Y)
#define Wo ((0) + 2*X*Y)
	//
#define Uf ((3*X*Y) + 0*Y*Y)
#define Ui ((3*X*Y) + 1*Y*Y)
#define Uo ((3*X*Y) + 2*Y*Y)
	//
#define Bf ((3*X*Y+3*Y*Y) + 0*Y)
#define Bi ((3*X*Y+3*Y*Y) + 1*Y)
#define Bo ((3*X*Y+3*Y*Y) + 2*Y)
	//
#define Wt ((3*X*Y+3*Y*Y+3*Y)+0)
#define Bt ((3*X*Y+3*Y*Y+3*Y+1*X*Y)+0)

//	============================================

void intel_lstm1d(					//mode = 0
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint t,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd);

void nvidia_lstm1d_naive(			//mode = 1
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint t,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd);

void nvidia_lstm1d__shared(			//mode = 2
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint t,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd);

//	faire tout en dot1d et mul

void f_lstm1d(Mdl_t * mdl, uint inst, uint mode, uint t0, uint t1);

//	============================================

void d_intel_lstm1d(					//mode = 0
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint t,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd,
	float * dy,
	float * dx,
	float * dp);

void d_nvidia_lstm1d_naive(			//mode = 1
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint t,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd,
	float * dy,
	float * dx,
	float * dp);

void d_nvidia_lstm1d__shared(			//mode = 2
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint t,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd,
	float * dy,
	float * dx,
	float * dp);

void df_lstm1d(Mdl_t * mdl, uint inst, uint mode, uint t0, uint t1);