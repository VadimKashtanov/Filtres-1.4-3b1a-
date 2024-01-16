#pragma once

#define TANH 0
#define LOGISTIC 1
#define SPECIALE 2

#define ACTIVATION TANH

#define tanh_f(s)          (tanh(s))
#define tanh_df(s,a)       (1 - a*a)

#define logistique_f(s)    (1.f/(1.f + expf(-s)))
#define logistique_df(s,a) (a * (a - 1.f))

#define speciale_f(s)      (expf(-powf(s-1.5,4)) - expf(-powf(s+1.5,4)))
#define speciale_df(s,a)   (-4*powf(s-1.5, 3)*speciale_f(s-1.5) + 4*powf(s+1.5,3)*speciale_f(s+1.5))

#define ACTIV(mode, s) (\
	(mode == TANH ? tanh_f(s) \
	: (mode == LOGISTIC ? logistique_f(s) \
	: (mode == SPECIALE ? speciale_f(s) \
	: 0 ))))
#define dACTIV(mode, s,a) (\
	(mode == TANH ? tanh_df(s,a) \
	: (mode == LOGISTIC ? logistique_df(s,a) \
	: (mode == SPECIALE ? speciale_df(s,a) \
	: 0 ))))

#include "mdl.cuh"

void cree_dot1d(Mdl_t * mdl, uint inst);
void plume_dot1d(Mdl_t * mdl, uint c);

//	============================================

void intel_dot1d(
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd);

void nvidia_dot1d_naive(
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd);

void nvidia_dot1d_shared(
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd);

void nvidia_dot1d_shared_2_16(
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd);

void f_dot1d(Mdl_t * mdl, uint inst, uint mode, uint t0, uint t1);

//	============================================

void d_intel_dot1d(
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd,
	float * dy,
	float * dx,
	float * dp);

void d_nvidia_dot1d_naive(
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd,
	float * dy,
	float * dx,
	float * dp);

void d_nvidia_dot1d_shared(
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd,
	float * dy,
	float * dx,
	float * dp);

void d_nvidia_dot1d_shared_2_16(
	uint X_vars, uint Y_vars,
	uint X, uint Y,
	uint depart, uint T,
	uint DEPART_x,
	float * x, float * y,
	float * p,
	float * locd,
	float * dy,
	float * dx,
	float * dp);

void df_dot1d(Mdl_t * mdl, uint inst, uint mode, uint t0, uint t1);