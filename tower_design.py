import numpy as np
from colored import fg
from math import *
from scipy import interpolate
from scipy import optimize
from functools import partial

def max_load(D,W,E):
	n = np.size(D)
	Uv = np.zeros((n,7))
	Pu = np.zeros(n)

	for i in range(n):
		Uv[i,0] = 1.4*D[i]
		Uv[i,1] = 0.9*D[i] + 1.6*W[i]
		Uv[i,2] = 1.2*D[i] + 1.6*W[i]
		Uv[i,3] = 0.9*D[i] + 1.4*W[i]
		Uv[i,4] = 1.2*D[i] + 1.4*W[i]
		Uv[i,5] = 0.9*D[i] + E[i]
		Uv[i,6] = 1.2*D[i] + E[i]
		Pu[i] = np.max(Uv[i,:])
	return Pu

def salt(T):
	Temp = T + 273.15
	density = -0.5786666667*Temp + 2124.1516888889                           # Salt density [kg/m3]
	enthalpy = 1/1000.*(1396.0182 * Temp + 0.086 * Temp**2)
	return density,enthalpy

def sodium(T):
	Temp = T + 273.15
	density = 219 + 275.32 * (1-Temp/2503.7) + 511.58*sqrt(1-Temp/2503.7)    # Sodium density [kg/m3]
	enthalpy = -365.77 + 1.6582 * Temp - 4.2395e-4 * Temp**2 + 1.4847e-7 * Temp**3 + 2992.6 /Temp - 104.90817873321107
	return density,enthalpy

def scaling_w_receiver(D_receiver, H_receiver, Q_rec_out):
	rho_salt_hot,h_salt_hot = salt(566.)
	rho_salt_cold,h_salt_cold = salt(290.)
	rho_na_hot,h_na_hot = sodium(740.)
	rho_na_cold,h_na_cold = sodium(520.)
	rho_na = 0.5*(rho_na_cold + rho_na_hot)
	rho_salt = 0.5*(rho_salt_cold + rho_salt_hot)
	delta_h_salt = h_salt_hot - h_salt_cold
	delta_h_na = h_na_hot - h_na_cold
	Q_rec_out_ref = 700.                                                     # CMI Receiver thermal output at design [MWth]
	scaffholding_ref = 1000.                                                 # CMI Steel scaffholding weight [tons]
	panels_ref = 100.                                                        # CMI panels weight [tons]
	installation_ref = 650.                                                  # CMI internal installation weight [tons]
	fluid_ref = 500.                                                         # CMI salt weight [tons]
	Vss_ref = 0.25*pi*17.**2*18.                                             # CMI receiver volume [m3]
	Apa_ref = pi*17.*18.                                                     # CMI panel area [m2]
	scaffholding = scaffholding_ref/Vss_ref*0.25*pi*D_receiver**2*H_receiver # Steel scaffholding weight [tons]
	panels = panels_ref/Apa_ref*pi*D_receiver*H_receiver                     # Panels weight [tons]
	installation = installation_ref/Vss_ref*0.25*pi*D_receiver**2*H_receiver # Internal installation weight [tons]
	fluid = fluid_ref*Q_rec_out/Q_rec_out_ref*delta_h_salt/delta_h_na        # Sodium weight method 1 [tons]
	W_receiver = scaffholding + panels + installation + fluid                # Receiver weight [tons]
	return W_receiver

def neutral_axis_angle(load,rhot,rm_tow,t_tow,alpha):
	alpha = radians(alpha)                                                   # 
	Es = 29000                                                               # Young's modulus for structural steel [ksi]
	fy = 36                                                                  # Isotropic yield strength for structural steel [ksi]
	Ke = Es/fy

	beta = 0
	gamma = 0
	n1 = 0
	rho_t = rhot
	fp_c = 4                                                                 # [ksi]
	omega_t = rho_t*fy/fp_c

	if fp_c > 4000:
		beta1 = max(0.85 - 0.05*(fp_c - 4000)/1000, 0.65)
	else:
		beta1 = 0.85

	em = min(0.003, 0.07*(1 - cos(alpha))/(1 + cos(alpha)))

	tau = acos(1 - beta1*(1 - cos(alpha)))
	psi = acos(max(-1,cos(alpha) - (1 - cos(alpha))/em*(fy/Es)))
	mu = acos(min(1,cos(alpha) + (1 - cos(alpha))/em*(fy/Es)))

	Q1 = (sin(psi) - sin(mu) - (psi - mu)*cos(alpha))/(1 - cos(alpha))

	lamda = tau - n1*beta
	lambda1 = mu + psi - np.pi

	if alpha > radians(35):
		Q = 0.89
	elif alpha > radians(25):
		Q = (0.993 - 0.00258*alpha) + (-3.27 + 0.0862*alpha)*(t_tow/rm_tow)
	elif alpha > radians(17):
		Q = (-1.345 + 0.2018*alpha + 0.004434*alpha**2) + (15.83 - 1.676*alpha + 0.03994*alpha**2)*(t_tow/rm_tow)
	elif alpha > radians(10):
		Q = (-0.488 + 0.076*alpha) + (9.758 - 0.640*alpha)*(t_tow/rm_tow)
	elif alpha > radians(5):
		Q = (-0.154 + 0.01773*alpha + 0.00249*alpha**2) + (16.42 - 1.980*alpha + 0.0674*alpha**2)*(t_tow/rm_tow)
	else:
		Q = (-0.523 + 0.181*alpha - 0.0154*alpha**2) + (41.3 - 13.2*alpha + 1.32*alpha**2)*(t_tow/rm_tow)

	stigma = 4*cos(alpha)*(sin(alpha) + sin(psi) - sin(mu))
	K = sin(psi) + sin(mu) + (np.pi - psi - mu)*cos(alpha)
	R_bar = sin(tau) - (tau - n1*beta1)*cos(alpha) - (0.5*n1)*(sin(gamma + beta) - sin(gamma - beta))

	Q2 = ((psi - mu)*(1 + 2*(cos(alpha))**2) + 0.5*(4*sin(2*alpha) + sin(2*psi) - sin(2*mu)) - stigma)/(1 - cos(alpha))

	K1 = 1.7*Q*lamda + 2*em*Ke*omega_t*Q1 + 2*omega_t*lambda1
	K2 = 1.7*Q*R_bar + em*Ke*omega_t*Q2 + 2*omega_t*K
	K3 = cos(alpha) + K2/K1

	Pu = K1*rm_tow*t_tow*fp_c*12**2
	res = abs(Pu - load)
	return res

def bending_resistance(rhot,rm_tow,t_tow,alpha):
	alpha = radians(alpha)
	phi = 0.7                                                       # ACI 307-08 strength reduction factor, section 5
	Es = 29000                                                      # Modulus of elasticity of reinforcement, in ksi
	fy = 36                                                         # Specified yield strength of reinforcing steel bars, in ksi
	Ke = Es/fy                                                      # Yield strain of reinforcing steel bars, in/in

	beta = 0                                                        # One-half of the central angle subtended by an opening on concrete shell cross section
	gamma = 0                                                       # One-half of the central angle subtended by the centerlines of two openings on concrete shell cross section
	n1 = 0                                                          # Number of openings entirely in compression zone
	rho_t = rhot                                                    # Ratio of total area of vertical reinforcement to total area of concrete shell cross section
	fp_c = 4                                                        # Specified compressive strength of concrete, in ksi
	omega_t = rho_t*fy/fp_c                                         # Ratio of compressive force on concrete and tension force on steel bars

	if fp_c > 4000:
		beta1 = max(0.85 - 0.05*(fp_c - 4000)/1000, 0.65)
	else:
		beta1 = 0.85

	em = min(0.003, 0.07*(1 - cos(alpha))/(1 + cos(alpha)))

	tau = acos(1 - beta1*(1 - cos(alpha)))
	psi = acos(max(-1,cos(alpha) - (1 - cos(alpha))/em*(fy/Es)))
	mu = acos(min(1,cos(alpha) + (1 - cos(alpha))/em*(fy/Es)))

	Q1 = (sin(psi) - sin(mu) - (psi - mu)*cos(alpha))/(1 - cos(alpha))

	lamda = tau - n1*beta
	lambda1 = mu + psi - np.pi

	if alpha > radians(35):
		Q = 0.89
	elif alpha > radians(25):
		Q = (0.993 - 0.00258*alpha) + (-3.27 + 0.0862*alpha)*(t_tow/rm_tow)
	elif alpha > radians(17):
		Q = (-1.345 + 0.2018*alpha + 0.004434*alpha**2) + (15.83 - 1.676*alpha + 0.03994*alpha**2)*(t_tow/rm_tow)
	elif alpha > radians(10):
		Q = (-0.488 + 0.076*alpha) + (9.758 - 0.640*alpha)*(t_tow/rm_tow)
	elif alpha > radians(5):
		Q = (-0.154 + 0.01773*alpha + 0.00249*alpha**2) + (16.42 - 1.980*alpha + 0.0674*alpha**2)*(t_tow/rm_tow)
	else:
		Q = (-0.523 + 0.181*alpha - 0.0154*alpha**2) + (41.3 - 13.2*alpha + 1.32*alpha**2)*(t_tow/rm_tow)

	stigma = 4*cos(alpha)*(sin(alpha) + sin(psi) - sin(mu))
	K = sin(psi) + sin(mu) + (np.pi - psi - mu)*cos(alpha)
	R_bar = sin(tau) - (tau - n1*beta1)*cos(alpha) - (0.5*n1)*(sin(gamma + beta) - sin(gamma - beta))

	Q2 = ((psi - mu)*(1 + 2*(cos(alpha))**2) + 0.5*(4*sin(2*alpha) + sin(2*psi) - sin(2*mu)) - stigma)/(1 - cos(alpha))

	K1 = 1.7*Q*lamda + 2*em*Ke*omega_t*Q1 + 2*omega_t*lambda1
	K2 = 1.7*Q*R_bar + em*Ke*omega_t*Q2 + 2*omega_t*K
	K3 = cos(alpha) + K2/K1

	Pu = K1*rm_tow*t_tow*fp_c*12**2
	Mn = phi*Pu*rm_tow*K3
	return Mn

def tower(tower_height, section_height, receiver_weight, thickness_bottom, thickness_top, diameter_bottom, diameter_top,verbose):
	"""
	This function designs the geometry of a concrete reinforced tower with fixed inner diameter, variable outside diameter and variables thickness
	
	tower_height:                    Tower height in meters
	section_height:                  Section height in meters
	thickness_bottom:                Tower thickness at bottom
	thickness_top                    Tower thickness at top
	"""

	# ************************** COMPUTING OUTER DIAMETERS AT THE TOP AND BOTTOM OF THE TOWER ***************************************
	do_b = diameter_bottom                                          # tower outside diameter at bottom in meters
	do_t = diameter_top                                             # tower outside diameter at top in meters

	# ************************** UNIT CONVERSION FROM ENGLISH TO SI SYSTEMS *********************************************************
	t_b = thickness_bottom*1000./25.4/12                            # tower thickness at the bottom, in feets
	t_t = thickness_top*1000./25.4/12                               # tower thickness at the top, in feets
	do_b = do_b*1000./25.4/12                                       # tower outer diameter at the bottom in feets
	do_t = do_t*1000./25.4/12                                       # tower outer diameter at the top in feets
	di_b = do_b - 2*t_b                                             # tower inner diameter at the bottom in feets
	di_t = do_t - 2*t_t                                             # tower inner diameter at the top in feets
	w_rec = receiver_weight*9.81*0.22481                            # Receiver weight, in kips
	h = tower_height*1000./25.4/12                                  # Tower height in feet
	h_sec = section_height*1000./25.4/12                            # Section height in feet

	# ************************** UNIT CONVERSION FROM ENGLISH TO SI SYSTEMS *********************************************************
	n = int(h/h_sec)                                                # Number of sections
	if h - n*h_sec > 0:
		n += 1
	h_sec = float(h)/n
	hbx = np.linspace(h,0,n+1)[1:n+1]                               # height to the bottom of each section, in feets
	hx = hbx + h_sec*0.5                                            # height to the middle of each section, in feets
	do_t_x = np.linspace(do_t,do_b,n+1)[0:n]                        # tower section outer diameter at top, in feets
	do_b_x = np.linspace(do_t,do_b,n+1)[1:n+1]                      # tower section outer diameter at bottom, in feets
	di_t_x = np.linspace(di_t,di_b,n+1)[0:n]                        # tower section inner diameter at top, in feets
	di_b_x = np.linspace(di_t,di_b,n+1)[1:n+1]                      # tower section inner diameter at bottom, in feets
	rm_tow = 0.25*(0.5*(do_t_x + do_b_x) + 0.5*(di_t_x + di_b_x))   # tower mean radius in feets
	t_tow = 0.5*(0.5*(do_t_x + do_b_x) - 0.5*(di_t_x + di_b_x))     # tower mean thickness in feets

	# ************************** COMPUTING THE WEIGHT OF STRUCTURE SECTIONS **********************************************************
	v_out_x = 0.25*pi*h_sec*1./3.*(do_t_x**2 + do_t_x*do_b_x + do_b_x**2)
	v_in_x = 0.25*pi*h_sec*1./3.*(di_t_x**2 + di_t_x*di_b_x + di_b_x**2)
	v_sec_x = v_out_x - v_in_x
	w_sec_x = v_sec_x*0.086*12**3/1000.
	# ************************** ADDITION OF RECEIVER WEIGHT AT THE TOP **************************************************************
	w_sec_x[0] += w_rec

	# ************************** COMPUTING WEIGHT OF THE STRUCTURE *******************************************************************
	w_structure = np.sum(w_sec_x, axis=0)

	# ************************** INITIALISATION OF SEISMIC ANALYSIS ******************************************************************
	E = 4200                                                        # Reinforced concrete's Young modulus
	Ss = 1.307                                                      # Mapped maximum considered earthquake (MCE), 5 percent damped, spectral response acceleration parameter at a period of 0.2s
	S1 = 0.467                                                      # Mapped maximum considered earthquake (MCE), 5 percent damped, spectral response acceleration parameter at a period of 1.0s

	# ************************** DETERMING SHORT-PERIOD AND LONG PERIOD SITE COEFFICIENTS **********************************************
	site_class = 'D'
	if site_class == 'D':
		x = [0.25,0.5,0.75,1,1.25]
		y = [1.6,1.4,1.2,1.1,1.00]
		f = interpolate.interp1d(x, y, fill_value=(y[0],y[-1]), bounds_error=False)
		Fa = f(Ss)                                                  # Short-period site coefficient (at 0.2 s-period)
		x = [0.1,0.2,0.3,0.4,0.5]
		y = [2.4,2.0,1.8,1.6,1.5]
		f = interpolate.interp1d(x, y, fill_value=(y[0],y[-1]), bounds_error=False)
		Fv = f(S1)                                                  # Long-period site coefficient (at 1.0 s-period)

	# ************************** DETERMING SHEAR FORCE *********************************************************************************
	SMS = Ss*Fa                                                     #
	SM1 = S1*Fv                                                     #
	SDS = 2./3.*Ss                                                  #
	SD1 = 2./3.*SM1                                                 #
	Ct = 0.02                                                       #
	x = 0.75                                                        #
	Ta = Ct*h**x                                                    #
	T = 1.8*h**2/(3*do_b - do_t)/np.sqrt(E*1000.)                   #
	I= 1.15                                                         # Importance factor
	R = 3.                                                          #
	Cu = 1.4                                                        #
	Tmax = Cu*Ta                                                    #
	Cs = min(SDS/(R/I),SDS/(Ta*R/I))                                #
	V = Cs*w_structure                                              #
	x = [0.5,2.5]
	y = [1.0,2.0]
	f = interpolate.interp1d(x, y, fill_value=(y[0],y[-1]), bounds_error=False)
	k = f(Ta)                                                       #
	J = max(min(0.6/(T**(1./3.)),1),0.45)                           #

	# ************************** DETERMING HORIZONTAL FORCE AND BENDIGN MOMENT DUE TO MCE **************************************************
	W_h = w_sec_x*hx**k
	S_W_h = np.sum(W_h, axis = 0)
	Cvx = W_h/S_W_h
	Fx = Cvx*V
	Jx = J + (1-J)*(hbx/h)**3
	Mx_seismic = np.zeros(n)
	for i in range(n):
		suma = 0
		for j in range(i+1):
			suma = suma + Fx[j]*(hx[j] - hbx[i])
			Mx_seismic[i] = Jx[i]*suma

	# ************************** INITIALISATION OF WIND ANALYSIS *****************************************************************************
	V = 85                                                          # miles per hour (mph)
	Vr = I**0.5*V
	Kd = 0.95
	T1 = 5*h**2/do_b*sqrt(0.086*12/32.2/4.2/1000000*(t_t/t_b)**0.3) #First period (Natural frequency)
	V33 = 1.47*Vr*(33/33)**0.154*0.65
	Gwp = 0.3+11*(T1*V33)**0.47/(h+16)**0.86
	z_cr = 5./6.*h
	V_z_cr = 1.47*Vr*(z_cr/33)**0.154*0.65
	V_z_cr_05 = 0.5*V_z_cr
	V_z_cr_13 = 1.3*V_z_cr
	F1A = 0.333 + 0.206*log(h/do_t)
	St = 0.25*F1A
	f = 1./T1
	V_cr = f*do_t/St

	# ************************** DETERMING MEAN ALONG-WIND LOADS ******************************************************************************
	Vx = 1.47*Vr*(hx/33)**0.154*0.65
	Cdr = np.zeros(n)
	for i in range(n):
		if hx[i] < (h - 1.5*do_t):
			Cdr[i] = 0.65
		else:
			Cdr[i] = 1.0
	px_bar = 0.0013*Vx**2
	wx_bar = Cdr*0.5*(do_t_x + do_b_x)*px_bar
	Mx_bar = wx_bar*h_sec/1000*hx
	Mwb_bar = np.sum(Mx_bar[:], axis=0)

	# ************************** DETERMING FLUCTUATING ALONG-WIND LOADS *************************************************************************
	wx_prime = 3*hx*Gwp*Mwb_bar*1000/(h**3)
	Mx_prime = wx_prime*h_sec/1000*hx

	# ************************** DETERMING ALONG-WIND LOADS *************************************************************************************
	Mx_wind = Mx_bar + Mx_prime
	Mx_wind = np.cumsum(Mx_wind)
	Fx_wind = (wx_prime + wx_bar)*h_sec/1000

	# ************************** CIRCUMFERENTIAL BENDING *************************************************************************************
	Gr_x = np.zeros(n)
	for i in range(n):
		if hx[i] > 1:
			Gr_x[i] = 4 - 0.8*log10(hx[i])
		else:
			Gr_x[i] = 4
	Mi_x = 0.31*px_bar*Gr_x*rm_tow/1000.
	Mo_x = 0.27*px_bar*Gr_x*rm_tow/1000.

	# ************************** FUNCTION OUTPUTS ***********************************************************************************************
	D = np.cumsum(w_sec_x)
	W = np.cumsum(Fx_wind)
	E = np.cumsum(Fx)

	# ************************** CALCULATING DESIGN LOADS ****************************************************************************************
	Pu = max_load(D, W, E)
	n = np.size(Pu)

	# ************************** OBTAINING BENDIGN RESISTANCE ************************************************************************************
	min_sf = 1.99
	Pu_tol = 1e-5
	res_min = 1e6
	sf_f_min = 0
	ii = 0
	rhot_f = 0.25/100.                                                # ACI 307-08, Section 4.4.1 states The circumferential reinforcement in each face shall be not less than 0.25% of the concrete area.

	while res_min > Pu_tol or sf_f_min < min_sf or rhot_f < 2.5/100:
		res = []
		sf = []
		Mn = []
		x0 = [45]
		for i in range(n):
			dbl = partial(neutral_axis_angle,Pu[i],rhot_f,rm_tow[i],t_tow[i])
			z = optimize.fsolve(dbl,45)
			alpha = z[0]
			res.append(neutral_axis_angle(Pu[i], rhot_f, rm_tow[i], t_tow[i], alpha))
			Mn.append(bending_resistance(rhot_f,rm_tow[i],t_tow[i],alpha))
			sf.append(bending_resistance(rhot_f,rm_tow[i],t_tow[i],alpha) / (Mx_seismic[i]+Mx_wind[i]))
		sf_f_min = np.min(sf)
		res_min = np.max(res)
		rhot_f += 0.1/100.                                            # Sweeping reinforcement ratio

	# ************************** DESIGN FOR CIRCUMFERENTIAL BENDING ******************************************************************************
	phi = 0.7                                                         # ACI 307-08 strength reduction factor, section 5
	bw = 12.                                                          # 1 feet of height section
	d_bar = 0.5                                                       # Diameter of N4 US reinforcing steel bars, in inches (https://www.tpub.com/steelworker2/76.htm)
	cov = 2.                                                          # Covering of steel, in inches
	fp_c = 4                                                          # Specified compressive strength of concrete [ksi]
	fy = 36.                                                          # Isotropic yield strength for structural steel [ksi]
	rhot_c = 0                                                        # ACI 307-08, Section 4.4.2 states The circumferential reinforcement in each face shall be not less than 0.1% of the concrete area at the section.
	if fp_c > 4000:
		beta1 = max(0.85 - 0.05*(fp_c - 4000)/1000, 0.65)
	else:
		beta1 = 0.85
	d = 12*t_tow - cov - 0.5*d_bar                                    # Width of section, in inches
	rho_b = 0.85*fp_c*beta1/fy*(87./(87.+fy))                         # ACI 318-11, Section B.8.4.2, Equation B-1

	sf_c_min = 0
	while sf_c_min < min_sf and rhot_c <= 0.5*rho_b:                  # ACI 318-11, Section B.8.4.2
		rhot_c += 0.1/100
		Mu = phi*rhot_c*bw*d*fy*(d - rhot_c*bw*d*fy/(1.7*fp_c*bw))/12 # Circumferential bending resistance
		sf_c_min = np.min(Mu/(Mi_x + Mo_x))                           # Safety factor for circumferential reinforcement

	# ************************** DESIGN FOR CIRCUMFERENTIAL BENDING ******************************************************************************
	V_concrete = 1.1*np.sum(v_sec_x)*(1 - rhot_c - rhot_f)/27.         # Cubic feet to cubic yards
	M_steel_c = np.sum(v_sec_x)*rhot_c*0.284*12**3/2000.               # Mass of circumferential reinforcing steel, in short tons
	M_steel_v = np.sum(v_sec_x)*rhot_f*0.284*12**3/2000.               # Mass of vertical reinforcing steel, in short tons
	M_steel = (M_steel_v + M_steel_c)

	# ************************** CALCULATING COST ************************************************************************************************
	C = [550.8, 584.6, 591.1]                                          # CEPCI index for years 2010, 2012 and 2020, respectively
	C_steel = M_steel*2364.3025971411                                  # Cost of reinforcing steel (using 2010 basis)
	C_concrete = V_concrete*420.9930175246                             # Cost of concrete
	C_tower_fix = 19588901*exp(0.0113*tower_height)/exp(0.0113*268.0)  # Cost of embedded metals, foundation and sitework
	C_tower = (C_tower_fix + C_concrete + C_steel)*C[1]/C[0]           # Cost of tower

	# ************************** MINIMUM SAFETY FACTOR *******************************************************************************************
	sf_min = min(sf_c_min,sf_f_min)
	if sf_min < min_sf and res_min > Pu_tol:
		C_tower = 1e20
	# ************************** PRINTING OUTPUTS ************************************************************************************************
	if verbose:
		print '%sThickness bottom (m):%s                        %s'%(fg(11),fg(13),thickness_bottom)
		print '%sThickness top (m):%s                           %s'%(fg(11),fg(13),thickness_top)
		print '%sMinimum vertical reinforcement ratio:%s        %s'%(fg(11),fg(13),rhot_f)
		print '%sMinimum circumferential reinforcement ratio:%s %s'%(fg(11),fg(13),rhot_c)
		print '%sPu iteration residual:%s                       %s'%(fg(11),fg(13),res_min)
		print '%sMinimum vertical safety factor:%s              %4.2f'%(fg(11),fg(13),sf_f_min)
		print '%sMinimum circumferential safety factor:%s       %4.2f'%(fg(11),fg(13),sf_c_min)
		print '%sConcrete volume [CY]:%s                        %4.2f'%(fg(11),fg(13),V_concrete)
		print '%sSteel mass [tons]:%s                           %4.2f'%(fg(11),fg(13),M_steel)
		print '%sCost of tower [$]:%s                           %4.2f\n'%(fg(11),fg(13),C_tower)

	return C_tower,sf_min,res_min
