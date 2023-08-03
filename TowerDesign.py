#!/bin/python3
import numpy as np
from scipy import interpolate
from scipy import optimize
from functools import partial

class ConcreteTower:
    """
    This class contains methods to design a concrete tower for concentrated solar power (CSP) systems
    by combining the standards ACI 307-08 from the American Concrete Institute (ACI) and ASCE 7-16
    from the American Society of Civil Engineers (ASCE).

    The tower geometry is assumed to be a hollow truncated cone with a fixed inner diameter and variable thickness.

    To use this class, create an instance and call the `calculate_cost` method with the required inputs.

    Example:
    >>> params = {
    ...    'tower_height': 268.0,
    ...     'section_height': 5.0,
    ...     'thickness_bottom': 0.566,
    ...     'thickness_top': 0.447,
    ...     'diameter_bottom': 30.4,
    ...     'diameter_top': 20.2,
    ...     'D_receiver': 20.0,
    ...     'H_receiver': 25.0,
    ...     'Q_rec_out': 936,
    ...     'T_in': 290.0,
    ...     'T_out': 566.0,
    ...     'IsSodium': False,
    ...     'verbose': True,
    ...     'load_type': 3,
    ...     'min_sf': 1.0
    ... }
    >>> tower = ConcreteTower()
    >>> cost, safety_factor, iteration_residual = tower.calculate_cost()
    >>> print('{0:.0f}'.format(cost))
    33529190  # Sample output for the estimated cost of the tower.

    Methods:
    - calculate_cost: Calculates the cost of a concrete reinforced tower with a user-defined geometry.

    Attributes:
    - params (dict): A dictionary containing the following parameters:
        - tower_height (float): Height of the tower in meters.
        - section_height (float): Height of each tower section in meters.
        - thickness_bottom (float): Thickness of the tower at the bottom in meters.
        - thickness_top (float): Thickness of the tower at the top in meters.
        - diameter_bottom (float): Outer diameter of the tower at the bottom in meters.
        - diameter_top (float): Outer diameter of the tower at the top in meters.
        - D_receiver (float): Diameter of the receiver in meters.
        - H_receiver (float): Height of the receiver in meters.
        - Q_rec_out (float): Heat output of the receiver in watts.
        - T_in (float): Inlet temperature of the receiver in Kelvin.
        - T_out (float): Outlet temperature of the receiver in Kelvin.
        - IsSodium (bool): Set to True if sodium is used in the receiver, otherwise False.
        - verbose (bool): If True, the method will print detailed output.
        - load_type (int): Type of load to consider. 1 for wind only, 2 for seismic only, and 3 for wind and seismic.
        - min_sf (float): Minimum safety factor for optimization purposes.

    """

    def __init__(self):
        """
        Initialize the ConcreteTower instance with default parameters.
        """
        self.params = {
            'tower_height': 268.0,
            'section_height': 5.0,
            'thickness_bottom': 0.566,
            'thickness_top': 0.447,
            'diameter_bottom': 30.4,
            'diameter_top': 20.2,
            'D_receiver': 20.0,
            'H_receiver': 25.0,
            'Q_rec_out': 936,
            'T_in': 290.0,
            'T_out': 566.0,
            'IsSodium': False,
            'verbose': True,
            'load_type': 3,
            'min_sf': 1.0
        }
        self.params['receiver_weight'] = self.scaling_w_receiver()

    def calculate_cost(self):
        """
        Calculate the cost of a concrete reinforced tower with a user-defined geometry.

        Returns:
        - cost (float): The estimated cost of the concrete reinforced tower.
        - safety_factor (float): The minimum safety factor obtained in the design process.
        - iteration_residual (float): The iteration residual from the design process.
        """

        # ************************** COMPUTING OUTER DIAMETERS AT THE TOP AND BOTTOM OF THE TOWER ***************************************
        do_b = self.params['diameter_bottom']                                                   # tower outside diameter at bottom in meters
        do_t = self.params['diameter_top']                                                      # tower outside diameter at top in meters

        # ************************** UNIT CONVERSION FROM SI TO ENGLISH SYSTEM **********************************************************
        t_b = self.params['thickness_bottom']*1000./25.4/12                                     # tower thickness at the bottom, in feets
        t_t = self.params['thickness_top']*1000./25.4/12                                        # tower thickness at the top, in feets
        do_b = do_b*1000./25.4/12                                                               # tower outer diameter at the bottom in feets
        do_t = do_t*1000./25.4/12                                                               # tower outer diameter at the top in feets
        di_b = do_b - 2*t_b                                                                     # tower inner diameter at the bottom in feets
        di_t = do_t - 2*t_t                                                                     # tower inner diameter at the top in feets
        w_rec = self.params['receiver_weight']*9.81*0.22481                                     # Receiver weight, in kips
        h = self.params['tower_height']*1000./25.4/12                                           # Tower height in feet
        h_sec = self.params['section_height']*1000./25.4/12                                     # Section height in feet

        # ************************** UNIT CONVERSION FROM ENGLISH TO SI SYSTEMS *********************************************************
        n = int(h/h_sec)                                                         # Number of sections
        if h%h_sec > 1e-6:
            n += 1
        h_sec = float(h)/n
        hbx = np.linspace(h,0,n+1)[1:n+1]                                        # height to the bottom of each section, in feets
        hx = hbx + h_sec*0.5                                                     # height to the middle of each section, in feets
        do_t_x = np.linspace(do_t,do_b,n+1)[0:n]                                 # tower section outer diameter at top, in feets
        do_b_x = np.linspace(do_t,do_b,n+1)[1:n+1]                               # tower section outer diameter at bottom, in feets
        di_t_x = np.linspace(di_t,di_b,n+1)[0:n]                                 # tower section inner diameter at top, in feets
        di_b_x = np.linspace(di_t,di_b,n+1)[1:n+1]                               # tower section inner diameter at bottom, in feets
        rm_tow = 0.25*(0.5*(do_t_x + do_b_x) + 0.5*(di_t_x + di_b_x))            # tower mean radius in feets
        t_tow = 0.5*(0.5*(do_t_x + do_b_x) - 0.5*(di_t_x + di_b_x))              # tower mean thickness in feets

        # ************************** COMPUTING THE WEIGHT OF STRUCTURE SECTIONS *********************************************************
        v_out_x = 0.25*np.pi*h_sec*1./3.*(do_t_x**2 + do_t_x*do_b_x + do_b_x**2)    # volume of the outer truncated conical section (or cylinder if the outer diameter is constant)
        v_in_x = 0.25*np.pi*h_sec*1./3.*(di_t_x**2 + di_t_x*di_b_x + di_b_x**2)     # volume of the inner truncated conical section (or cylinder if the inner diameter is constant)
        v_sec_x = v_out_x - v_in_x                                                  # volume of tower section
        w_sec_x = v_sec_x*0.086*12**3/1000.                                         # weight of tower section
        w_sec_x[0] += w_rec                                                         # adding receiver to the top

        # ************************** COMPUTING WEIGHT OF THE STRUCTURE ******************************************************************
        w_structure = np.sum(w_sec_x, axis=0)                                    # cummulative weight of tower

        # ************************** INITIALISATION OF SEISMIC ANALYSIS *****************************************************************
        E = 4200                                                                 # Reinforced concrete's Young modulus
        Ss = 1.307                                                               # Mapped maximum considered earthquake (MCE), 5 percent damped, spectral response acceleration parameter at a period of 0.2s
        S1 = 0.467                                                               # Mapped maximum considered earthquake (MCE), 5 percent damped, spectral response acceleration parameter at a period of 1.0s

        # ************************** DETERMINING SHORT-PERIOD AND LONG PERIOD SITE COEFFICIENTS *****************************************
        site_class = 'D'                                                         # Site class based on the site soil properties (Sec. 11.4.3, ASCE 7-16)
        if site_class == 'D':
            x = [0.25,0.5,0.75,1,1.25]
            y = [1.6,1.4,1.2,1.1,1.00]
            f = interpolate.interp1d(x, y, fill_value=(y[0],y[-1]), bounds_error=False)
            Fa = f(Ss)                                                           # Short-period site coefficient (at 0.2 s-period)
            x = [0.1,0.2,0.3,0.4,0.5]
            y = [2.4,2.0,1.8,1.6,1.5]
            f = interpolate.interp1d(x, y, fill_value=(y[0],y[-1]), bounds_error=False)
            Fv = f(S1)                                                           # Long-period site coefficient (at 1.0 s-period)

        # ************************** DETERMINING SHEAR FORCE ****************************************************************************
        SMS = Ss*Fa                                                              # The MCER spectral response acceleration parameters for short periods, Eq. 11.4.1 ASCE 7-16 (Dagget, USA)
        SM1 = S1*Fv                                                              # The MCER spectral response acceleration parameters at a period of 1 s, Eq. 11.4.2 ASCE 7-16 (Dagget, USA)
        SDS = 2./3.*Ss                                                           # Design, 5% damped, spectral response acceleration parameter at short periods, Eq. 11.4.3 ASCE 7-16
        SD1 = 2./3.*SM1                                                          # Design, 5% damped, spectral response acceleration parameter at a period of 1 s, Eq. 11.4.4 ASCE 7-16
        T = 5*h**2/do_b*np.sqrt(0.086*12/32.2/4.2/1000000)*(t_t/t_b)**0.3        # Fundamental period ACI 307-08
        I= 1                                                                     # Importance factor ASCE 7-16, Section 1.5.1
        R = 1.5                                                                  # Response modification factor, Section 4.3.2 ACI 307-08
        Cu = 1.4                                                                 # Coefficient for Upper Limit on Calculated Period, Table 12.8.1 ASCE 7-16
        T = min(T, Cu*T)                                                         # Upper Limit on Calculated Period, ASCE 7-16
        Cs = min(SDS/(R/I),SD1/(T*R/I))                                          # The seismic response coefficient, Eq. 12.8.2-12.8.3 ASCE 7-16
        V = Cs*w_structure                                                       # Seismic base shear, Eq. 12.8.1 ASCE 7-16
        x = [0.5,2.5]
        y = [1.0,2.0]
        f = interpolate.interp1d(x, y, fill_value=(y[0],y[-1]), bounds_error=False)
        k = f(T)                                                                 # Exponent related to the structure period, Section 12.8.3 ASCE 7-16
        J = max(min(0.6/(T**(1./3.)),1),0.45)                                    # Numerical coefficient for based moment, ACI 307-79

        # ************************** DETERMINING HORIZONTAL FORCE AND BENDIGN MOMENT DUE TO MCE *****************************************
        W_h = w_sec_x*hx**k                                                      # Portion of the total effective seismic moment of the structure, ASCE 7-16
        S_W_h = np.sum(W_h, axis = 0)                                            # Total effective seismic moment of the structure, ASCE 7-16
        Cvx = W_h/S_W_h                                                          # Vertical distribution factor, Eq. 12.18.12 ASCE 7-16
        Fx = Cvx*V                                                               # Lateral seismic force inducted at any level (x), Eq. 12.18.11 ASCE 7-16
        Jx = J + (1-J)*(hbx/h)**3                                                # Numerical coefficient for based moment at any level (x)
        Mx_seismic = np.zeros(n)
        for i in range(n):
            suma = 0
            for j in range(i+1):
                suma = suma + Fx[j]*(hx[j] - hbx[i])
                Mx_seismic[i] = Jx[i]*suma                                       # Moment at any level (x), ACI 307-79

        # ************************** INITIALISATION OF WIND ANALYSIS ********************************************************************
        V = 97                                                               # Basic wind speed, in miles per hour (mph), Figure 26.5-1A ASCE 7-16 (Dagget, USA)
        Vr = I**0.5*V                                                            # Reference design wind speed, in miles per hour (mph)
        V33 = 1.47*Vr*(33/33)**0.154*0.65                                        # Mean hourly wind speed at height of 33 ft, in ft/s
        z_cr = 5./6.*h                                                           # Height corresponding to Vcr
        V_z_cr = 1.47*Vr*(z_cr/33)**0.154*0.65                                   # Critical wind speed for across-wind loads
        V_z_cr_05 = 0.5*V_z_cr                                                   # Critical wind speed for across-wind loads over a ranfe of 0.5*V_z_cr
        V_z_cr_13 = 1.3*V_z_cr                                                   # Critical wind speed for across-wind loads over a ranfe of 1.3*V_z_cr
        F1A = 0.333 + 0.206*np.log(h/do_t)                                       # Strouhal number parameter
        St = 0.25*F1A                                                            # Strouhal number
        f = 1./T                                                                 # Frequency
        V_cr = f*do_t/St                                                         # Mean hourly design wind speed at (5/6)h

        # ************************** DETERMINING MEAN ALONG-WIND LOADS ******************************************************************
        Vx = 1.47*Vr*(hx/33)**0.154*0.65                                         # Mean hourly wind speed at any level, Eq. 4-1 ACI 307-08
        V_rec = 1.47*Vr*((h+20*3.28)/33)**0.154*0.65
        Kd = 0.95                                                                # Geometric parameter for circular shapes
        Cdr = np.zeros(n)
        for i in range(n):
            if hx[i] < (h - 1.5*do_t):
                Cdr[i] = 0.65                                                    # Drag coefficient for along-wind load, Eq. 4-3 ACI 307-08
            else:
                Cdr[i] = 1.0                                                     # Drag coefficient for along-wind load, Eq. 4-4 ACI 307-08
        px_bar = 0.00119*Kd*Vx**2                                                # Pressure due to mean hourly design wind speed at any level (x), Eq. 4-4 ACI 307-08
        px_bar_rec = 0.00119*Kd*V_rec**2
        wx_bar = Cdr*0.5*(do_t_x + do_b_x)*px_bar                                # Mean along-wind load per unit length at any level, Eq. 4-2 ACI 307-08
        wx_bar_rec = 17*3.28*px_bar_rec
        Mx_bar = wx_bar*h_sec/1000*hx                                            # Mean along-wind bending moment at any level
        Mx_bar[0] += wx_bar_rec*40*3.28/1000*(h + 20*3.28)
        Mwb_bar = np.sum(Mx_bar[:], axis=0)                                      # Mean along-wind bending moment at the base

        # ************************** DETERMINING FLUCTUATING ALONG-WIND LOADS ***********************************************************
        Gwp = 0.3+11*(T*V33)**0.47/(h+16)**0.86                                  # Gust factor for along-wind fluctuating load, Eq. 4-7 ACI 307-08
        wx_prime = 3*hx*Gwp*Mwb_bar*1000/(h**3)                                  # Fluctuating along-wind load per unit length at any level, Eq. 4-6 ACI 307-08
        wx_prime_rec = 3*(h + 20*3.28)*Gwp*Mwb_bar*1000/(h**3)
        Mx_prime = wx_prime*h_sec/1000*hx                                        # Fluctuating along-wind bending moment at any level
        Mx_prime[0] += wx_prime_rec*40*3.28/1000*(h + 20*3.28)

        # ************************** DETERMINING ALONG-WIND LOADS ***********************************************************************
        Mx_wind = Mx_bar + Mx_prime                                              # Mean and fluctuating along-wind bending moment at any level
        Mx_wind = np.cumsum(Mx_wind)                                             # Cummulative mean and fluctuating along-wind bending moment at any level
        Fx_wind = (wx_prime + wx_bar)*h_sec/1000                                 # Mean and fluctuating along-wind shear force at any level
        Fx_wind[0] += (wx_prime_rec + wx_bar_rec)*40*3.28/1000

        # ************************** CIRCUMFERENTIAL BENDING ****************************************************************************
        Gr_x = np.zeros(n)
        for i in range(n):
            if hx[i] > 1:
                Gr_x[i] = 4 - 0.8*np.log10(hx[i])                                # Gust factor for radial wind pressure at any level, Eq. 4-2 ACI 307-08
            else:
                Gr_x[i] = 4                                                      # Gust factor for radial wind pressure at any level, Eq. 4-2 ACI 307-08
        Mi_x = 0.31*px_bar*Gr_x*rm_tow/1000.                                     # Maximum circumferential bending moment due to radial wind pressure, at height z, tension on inside, Eq. 4-29 ACI 307-08
        Mo_x = 0.27*px_bar*Gr_x*rm_tow/1000.                                     # Maximum circumferential bending moment due to radial wind pressure, at height z, tension on outside, Eq. 4-30 ACI 307-08

        # ************************** FUNCTION OUTPUTS ***********************************************************************************
        D = np.cumsum(w_sec_x)                                                   # Cummulative dead load, in Kips
        W = np.cumsum(Fx_wind)                                                   # Cummulative wind shear load, in Kips
        E = np.cumsum(Fx)                                                        # Cummulative earthquake shear load, in Kips

        # ************************** CALCULATING DESIGN LOADS ****************************************************************************************
        Pu = self.max_load(D, W, E, self.params['load_type'])                                        # Factored vertical load, in Kips
        n = np.size(Pu)
        if self.params['load_type'] == 1:
            type_load = 'Wind only'
            Mu = Mx_wind
        elif self.params['load_type'] == 2:
            Mu = Mx_seismic
            type_load = 'Seismic only'
        elif self.params['load_type'] == 3:
            Mu = Mx_seismic + Mx_wind
            type_load = 'Wind and seismic'
        else:
            raise(ValueError("load_type is outside the available options. load_type should be 1 for wind only, 2 for seismic only, and 3 for wind and seismic."))

        # ************************** OBTAINING BENDIGN RESISTANCE ************************************************************************************
        Pu_tol = 1e-5                                                            # Minimum tolerance to obtain the circumferential reinforcement
        res_min = 1e6                                                            # Place holder for residual
        sf_f_min = 0                                                             # Place holder for calculated safety factor
        rhot_f = 0.25/100.                                                       # ACI 307-08, Section 4.4.1, the circumferential reinforcement in each face shall be not less than 0.25% of the concrete area.

        while res_min > Pu_tol or sf_f_min < self.params['min_sf']:
            res = []
            sf = []
            Mn = []
            for i in range(n):
                dbl = partial(self.neutral_axis_angle,Pu[i],rhot_f,rm_tow[i],t_tow[i])
                z = optimize.fsolve(dbl,45)
                alpha = z[0]
                res.append(self.neutral_axis_angle(Pu[i], rhot_f, rm_tow[i], t_tow[i], alpha))
                Mn.append(self.bending_resistance(rhot_f,rm_tow[i],t_tow[i],alpha))
                sf.append(self.bending_resistance(rhot_f,rm_tow[i],t_tow[i],alpha) / (Mu[i]))
            sf_f_min = np.min(sf)
            res_min = np.max(res)
            if sf_f_min > self.params['min_sf'] and res_min < Pu_tol:
                pass
            else:
                rhot_f += 0.1/100.                                               # Reinforcement ratio step (0.1%)

        # ************************** DESIGN FOR CIRCUMFERENTIAL BENDING ******************************************************************************
        phi = 0.7                                                                # ACI 307-08 strength reduction factor, section 5
        bw = 12.                                                                 # 1 feet of height section
        d_bar = 0.5                                                              # Diameter of N4 US reinforcing steel bars, in inches (https://www.tpub.com/steelworker2/76.htm)
        cov = 2.                                                                 # Covering of steel, in inches
        fp_c = 4                                                                 # Specified compressive strength of concrete [ksi]
        fy = 36.                                                                 # Isotropic yield strength for structural steel [ksi]
        rhot_c = 0                                                               # ACI 307-08, Section 4.4.2 states The circumferential reinforcement in each face shall be not less than 0.1% of the concrete area at the section.
        if fp_c > 4000:
            beta1 = max(0.85 - 0.05*(fp_c - 4000)/1000, 0.65)
        else:
            beta1 = 0.85
        d = 12*t_tow - cov - 0.5*d_bar                                           # Width of section, in inches
        rho_b = 0.85*fp_c*beta1/fy*(87./(87.+fy))                                # ACI 318-11, Section B.8.4.2, Equation B-1

        sf_c_min = 0
        while sf_c_min < self.params['min_sf'] and rhot_c <= 0.5*rho_b:                         # ACI 318-11, Section B.8.4.2
            rhot_c += 0.1/100
            Mu = phi*rhot_c*bw*d*fy*(d - rhot_c*bw*d*fy/(1.7*fp_c*bw))/12        # Circumferential bending resistance
            sf_c_min = np.min(Mu/(Mi_x + Mo_x))                                  # Safety factor for circumferential reinforcement

        # ************************** DESIGN FOR CIRCUMFERENTIAL BENDING ******************************************************************************
        V_concrete = np.sum(v_sec_x)/27.                                         # Cubic feet to cubic yards
        M_steel_c = np.sum(v_sec_x)*rhot_c*0.284*12**3/2000.                     # Mass of circumferential reinforcing steel, in short tons
        M_steel_v = np.sum(v_sec_x)*rhot_f*0.284*12**3/2000.                     # Mass of vertical reinforcing steel, in short tons
        M_steel = (M_steel_v + M_steel_c)

        # ************************** CALCULATING COST ************************************************************************************************
        C = [550.8, 584.6, 591.1]                                                               # CEPCI index for years 2010, 2012 and 2020, respectively
        C_steel = M_steel*2364.3025971411                                                       # Cost of reinforcing steel (using 2010 basis)
        C_concrete = V_concrete*420.9930175246                                                  # Cost of concrete
        C_tower_fix = 19588901*np.exp(0.0113*self.params['tower_height'])/np.exp(0.0113*268.0)  # Cost of embedded metals, foundation and sitework
        C_tower = (C_tower_fix + C_concrete + C_steel)*C[2]/C[0]                                # Cost of tower, scaled to 2020 values

        # ************************** MINIMUM SAFETY FACTOR *******************************************************************************************
        sf_min = min(sf_c_min,sf_f_min)
        if sf_min < self.params['min_sf'] and res_min > Pu_tol:
            C_tower = 1e20                                                       # Cost penalty for optimisation purposes
        # ************************** PRINTING OUTPUTS ************************************************************************************************
        if self.params['verbose']:
            print('---------------------------------------------------')
            print('Type of load:{}                                '.format(type_load))
            print('---------------------------------------------------')
            print('Thickness bottom (m):                        {}'.format(self.params['thickness_bottom']))
            print('Thickness top (m):                           {}'.format(self.params['thickness_top']))
            print('Minimum vertical reinforcement ratio:        {0:.1e}'.format(rhot_f))
            print('Minimum circumferential reinforcement ratio: {0:.1e}'.format(rhot_c))
            print('Pu iteration residual:                       {0:.1e}'.format(res_min))
            print('Minimum vertical safety factor:              {0:.1e}'.format(sf_f_min))
            print('Minimum circumferential safety factor:       {0:.1e}'.format(sf_c_min))
            print('---------------------------------------------------')
            print('Concrete volume [CY]:                        {0:.1f}'.format(V_concrete))
            print('Steel mass [tons]:                           {0:.1f}'.format(M_steel))
            print('---------------------------------------------------')
            print('Concrete volume [m3]:                        {0:.1f}'.format(V_concrete*27/(1000./25.4/12)**3))
            print('Steel mass [metric tons]:                    {0:.1f}'.format(M_steel*0.907185))
            print('---------------------------------------------------')
            print('Cost of tower [$]:                           {0:.0f}'.format(C_tower))
            print('---------------------------------------------------')
        return C_tower,sf_min,res_min


    def max_load(self,D,W,E,load_type):
        """
        Determines the maximum factored load for design purposes based on ACI 307-08 standards.

        This method calculates the maximum factored load for design purposes of a concrete tower by considering different load combinations
        based on the type of load specified. The load combinations are derived from the ACI 307-08 standards, which include the factored 
        loads for dead load, wind load, and earthquake load.

        Parameters:
        - D (numpy.array): Array containing the cumulative dead loads at different sections of the tower, in Kips.
        - W (numpy.array): Array containing the cumulative wind shear loads at different sections of the tower, in Kips.
        - E (numpy.array): Array containing the cumulative earthquake shear loads at different sections of the tower, in Kips.
        - load_type (int): Type of load to consider. 1 for wind only, 2 for seismic only, and 3 for wind and seismic.

        Returns:
        - Pu (numpy.array): Array containing the maximum factored loads at different sections of the tower, in Kips.
        """

        n = np.size(D)
        Uv = np.zeros((n,7))
        Pu = np.zeros(n)

        if self.params['load_type'] == 1:
            for i in range(n):
                Uv[i,0] = 1.4*D[i]                                                   # Eq. 5-1, ACI 307-08
                Uv[i,1] = 0.9*D[i] + 1.6*W[i]                                        # Eq. 5-2, ACI 307-08
                Uv[i,2] = 1.2*D[i] + 1.6*W[i]                                        # Eq. 5-2a, ACI 307-08
                Uv[i,3] = 0.9*D[i] + 1.4*W[i]                                        # Eq. 5-3, ACI 307-08
                Uv[i,4] = 1.2*D[i] + 1.4*W[i]                                        # Eq. 5-3a, ACI 307-08
                Pu[i] = np.max(Uv[i,:])
        elif self.params['load_type'] == 2:
            for i in range(n):
                Uv[i,0] = 1.4*D[i]                                                   # Eq. 5-1, ACI 307-08
                Uv[i,5] = 0.9*D[i] + E[i]                                            # Eq. 5-4, ACI 307-08
                Uv[i,6] = 1.2*D[i] + E[i]                                            # Eq. 5-4a, ACI 307-08
                Pu[i] = np.max(Uv[i,:])
        elif self.params['load_type'] == 3:
            for i in range(n):
                Uv[i,0] = 1.4*D[i]                                                   # Eq. 5-1, ACI 307-08
                Uv[i,1] = 0.9*D[i] + 1.6*W[i]                                        # Eq. 5-2, ACI 307-08
                Uv[i,2] = 1.2*D[i] + 1.6*W[i]                                        # Eq. 5-2a, ACI 307-08
                Uv[i,3] = 0.9*D[i] + 1.4*W[i]                                        # Eq. 5-3, ACI 307-08
                Uv[i,4] = 1.2*D[i] + 1.4*W[i]                                        # Eq. 5-3a, ACI 307-08
                Uv[i,5] = 0.9*D[i] + E[i]                                            # Eq. 5-4, ACI 307-08
                Uv[i,6] = 1.2*D[i] + E[i]                                            # Eq. 5-4a, ACI 307-08
                Pu[i] = np.max(Uv[i,:])
        else:
            raise(ValueError("load_type is outside the available options. load_type should be 1 for wind only, 2 for seismic only, and 3 for wind and seismic."))
        return Pu

    def salt(self,T):
        """
        Determines the salt density and specific enthalpy for receiver weight calculation.

        This method calculates the density and specific enthalpy of salt at a given temperature for the purpose of calculating
        the receiver weight in a solar energy system. The calculations are based on empirical correlations.

        Parameters:
        - T (float): Temperature of the salt in degrees Celsius.

        Returns:
        - density (float): Density of salt at the given temperature in kilograms per cubic meter (kg/m3).
        - enthalpy (float): Specific enthalpy of salt at the given temperature in kilojoules per kilogram (kJ/kg).
        """

        Temp = T + 273.15
        density = -0.5786666667*Temp + 2124.1516888889                           # Salt density [kg/m3]
        enthalpy = 1/1000.*(1396.0182 * Temp + 0.086 * Temp**2)                  # Salt specific enthalpy [kJ/kg]
        return density,enthalpy

    def sodium(self,T):
        """
        Determines the sodium density and specific enthalpy for receiver weight calculation.

        This method calculates the density and specific enthalpy of sodium at a given temperature for the purpose of calculating
        the receiver weight in a solar energy system. The calculations are based on empirical correlations.

        Parameters:
        - T (float): Temperature of the sodium in degrees Celsius.

        Returns:
        - density (float): Density of sodium at the given temperature in kilograms per cubic meter (kg/m³).
        - enthalpy (float): Specific enthalpy of sodium at the given temperature in kilojoules per kilogram (kJ/kg).
        """

        Temp = T + 273.15
        density = 219 + 275.32 * (1-Temp/2503.7) + 511.58*sqrt(1-Temp/2503.7)    # Sodium density [kg/m3]
        enthalpy = -365.77 + 1.6582 * Temp - 4.2395e-4 * Temp**2 + 1.4847e-7 * Temp**3 + 2992.6 /Temp - 104.90817873321107
        return density,enthalpy

    def scaling_w_receiver(self):
        """
        Calculates the weight of the receiver based on its capacity, geometry, heat transfer fluid (HTF), and operating temperatures.

        The function is designed for two heat transfer fluids (HTFs): liquid sodium and nitrate salt (KNO3-NaNO3).
        The following parameters are required:
        - D_receiver (float): Diameter of the receiver in meters.
        - H_receiver (float): Height of the receiver in meters.
        - Q_rec_out (float): Tower thickness at the bottom (assumed to be the thermal output at the receiver outlet).
        - T_in (float): HTF temperature at the receiver inlet in degrees Celsius.
        - T_out (float): HTF temperature at the receiver outlet in degrees Celsius.
        - IsSodium (bool): HTF choice: True for liquid sodium, False for nitrate salt.

        Returns:
        - W_receiver (float): Weight of the receiver in metric tons (tonnes).
        """

        rho_salt_hot,h_salt_hot = self.salt(566.)                                                              # Nitrate salt density and enthalpy at outlet
        rho_salt_cold,h_salt_cold = self.salt(290.)                                                            # Nitrate salt density and enthalpy at inlet
        if self.params['IsSodium']:
            rho_htf_hot,h_htf_hot = self.sodium(self.params['T_out'])                                          # Liquid sodium density and enthalpy at outlet
            rho_htf_cold,h_htf_cold = self.sodium(self.params['T_in'])                                         # Liquid sodium density and enthalpy at inlet
        else:
            rho_htf_hot,h_htf_hot = self.salt(self.params['T_out'])                                            # Nitrate salt density and enthalpy at outlet
            rho_htf_cold,h_htf_cold = self.salt(self.params['T_in'])                                           # Nitrate salt density and enthalpy at inlet
        rho_htf = 0.5*(rho_htf_cold + rho_htf_hot)                                                             # Average sodium density
        rho_salt = 0.5*(rho_salt_cold + rho_salt_hot)                                                          # Average salt density
        delta_h_salt = h_salt_hot - h_salt_cold                                                                # Fluid enthalpy gain
        delta_h_htf = h_htf_hot - h_htf_cold                                                                   # Nitrate salt enthalpy gain
        Q_rec_out_ref = 700.                                                                                   # CMI Receiver thermal output at design [MWth]
        scaffholding_ref = 1000.                                                                               # CMI Steel scaffholding weight [metric tons]
        panels_ref = 100.                                                                                      # CMI panels weight [metric tons]
        installation_ref = 650.                                                                                # CMI internal installation weight [metric tons]
        fluid_ref = 500.                                                                                       # CMI nitrate salt weight [metric tons]
        Vss_ref = 0.25*np.pi*17.**2*18.                                                                        # CMI receiver volume [m3]
        Apa_ref = np.pi*17.*18.                                                                                # CMI panel area [m2]
        scaffholding = scaffholding_ref/Vss_ref*0.25*np.pi*self.params['D_receiver']**2*self.params['H_receiver'] # Steel scaffholding weight [metric tons]
        panels = panels_ref/Apa_ref*np.pi*self.params['D_receiver']*self.params['H_receiver']                     # Panels weight [metric tons]
        installation = installation_ref/Vss_ref*0.25*np.pi*self.params['D_receiver']**2*self.params['H_receiver'] # Internal installation weight [metric tons]
        fluid = fluid_ref*self.params['Q_rec_out']/Q_rec_out_ref*delta_h_salt/delta_h_htf                         # Fluid weight method 1 [metric tons]
        W_receiver = scaffholding + panels + installation + fluid                                                 # Receiver weight [metric tons]
        return W_receiver

    def neutral_axis_angle(self,load,rhot,rm_tow,t_tow,alpha):
        """
        Computes the neutral axis angle for a tower's cross-section subjected to bending moment.

        This method estimates the neutral axis angle for a tower's concrete shell cross-section under a specific bending moment load.
        The method involves various empirical correlations and calculations to determine the neutral axis angle.

        Parameters:
        - load (float): The bending moment load applied to the tower's cross-section.
        - rhot (float): Ratio of total area of vertical reinforcement to the total area of the concrete shell cross-section.
        - rm_tow (float): Outer radius of the tower at the bottom (in meters).
        - t_tow (float): Tower thickness at the bottom (in meters).
        - alpha (float): Angle of the applied bending moment load in degrees.

        Returns:
        - res (float): The absolute difference between the estimated moment strength and the provided bending moment load.
        """

        # ************************** COMPUTING OUTER DIAMETERS AT THE TOP AND BOTTOM OF THE TOWER ***************************************
        alpha = np.radians(alpha)                                                # Conversion to radians
        Es = 29000                                                               # Young's modulus for structural steel [ksi]
        fy = 36                                                                  # Isotropic yield strength for structural steel [ksi]
        Ke = Es/fy                                                               # Yield strain of reinforcing steel bars, in/in

        # ************************** DEFINITIONS OF OPENINGS ****************************************************************************
        beta = 0                                                                 # One-half of the central angle subtended by an opening on concrete shell cross section
        gamma = 0                                                                # One-half of the central angle subtended by the centerlines of two openings on concrete shell cross section
        n1 = 0                                                                   # Number of openings entirely in compression zone
        rho_t = rhot                                                             # Ratio of total area of vertical reinforcement to total area of concrete shell cross section
        fp_c = 4                                                                 # Specified compressive strength of concrete, in ksi
        omega_t = rho_t*fy/fp_c                                                  # Ratio of compressive force on concrete and tension force on steel bars

        # ************************** ESTIMATION COMPRESSIVE STRENGTH OF CONCRETE ********************************************************
        if fp_c > 4000:
            beta1 = max(0.85 - 0.05*(fp_c - 4000)/1000, 0.65)
        else:
            beta1 = 0.85

        # ************************** MAXIMUM CONCRETE COMPRESSIVE STRAIN ****************************************************************
        em = min(0.003, 0.07*(1 - np.cos(alpha))/(1 + np.cos(alpha)))

        # ************************** ANGLES FROM THE STRAIN DIAGRAM *********************************************************************
        tau = np.arccos(1 - beta1*(1 - np.cos(alpha)))
        psi = np.arccos(max(-1,np.cos(alpha) - (1 - np.cos(alpha))/em*(fy/Es)))
        mu = np.arccos(min(1,np.cos(alpha) + (1 - np.cos(alpha))/em*(fy/Es)))
        Q1 = (np.sin(psi) - np.sin(mu) - (psi - mu)*np.cos(alpha))/(1 - np.cos(alpha))
        lamda = tau - n1*beta
        lambda1 = mu + psi - np.pi

        # ************************** ANGLES FROM THE STRESS DIAGRAM *********************************************************************
        if alpha > np.radians(35):
            Q = 0.89
        elif alpha > np.radians(25):
            Q = (0.993 - 0.00258*alpha) + (-3.27 + 0.0862*alpha)*(t_tow/rm_tow)
        elif alpha > np.radians(17):
            Q = (-1.345 + 0.2018*alpha + 0.004434*alpha**2) + (15.83 - 1.676*alpha + 0.03994*alpha**2)*(t_tow/rm_tow)
        elif alpha > np.radians(10):
            Q = (-0.488 + 0.076*alpha) + (9.758 - 0.640*alpha)*(t_tow/rm_tow)
        elif alpha > np.radians(5):
            Q = (-0.154 + 0.01773*alpha + 0.00249*alpha**2) + (16.42 - 1.980*alpha + 0.0674*alpha**2)*(t_tow/rm_tow)
        else:
            Q = (-0.523 + 0.181*alpha - 0.0154*alpha**2) + (41.3 - 13.2*alpha + 1.32*alpha**2)*(t_tow/rm_tow)

        # ************************** ESTIMATING NOMINAL MOMENT STRENGTH *****************************************************************
        stigma = 4*np.cos(alpha)*(np.sin(alpha) + np.sin(psi) - np.sin(mu))
        K = np.sin(psi) + np.sin(mu) + (np.pi - psi - mu)*np.cos(alpha)
        R_bar = np.sin(tau) - (tau - n1*beta1)*np.cos(alpha) - (0.5*n1)*(np.sin(gamma + beta) - np.sin(gamma - beta))
        Q2 = ((psi - mu)*(1 + 2*(np.cos(alpha))**2) + 0.5*(4*np.sin(2*alpha) + np.sin(2*psi) - np.sin(2*mu)) - stigma)/(1 - np.cos(alpha))
        K1 = 1.7*Q*lamda + 2*em*Ke*omega_t*Q1 + 2*omega_t*lambda1
        K2 = 1.7*Q*R_bar + em*Ke*omega_t*Q2 + 2*omega_t*K
        K3 = np.cos(alpha) + K2/K1
        Pu = K1*rm_tow*t_tow*fp_c*12**2
        res = abs(Pu - load)
        return res

    def bending_resistance(self,rhot,rm_tow,t_tow,alpha):
        """
        Computes the bending resistance of a tower's concrete shell cross-section.

        This method estimates the bending resistance of a tower's concrete shell cross-section subjected to bending moment.
        The method involves various empirical correlations and calculations to determine the bending resistance.

        Parameters:
        - rhot (float): Ratio of total area of vertical reinforcement to the total area of the concrete shell cross-section.
        - rm_tow (float): Outer radius of the tower at the bottom (in meters).
        - t_tow (float): Tower thickness at the bottom (in meters).
        - alpha (float): Angle of the applied bending moment load in degrees.

        Returns:
        - Mn (float): The bending resistance of the tower's concrete shell cross-section in ft-kips.
        """

        # ************************** COMPUTING OUTER DIAMETERS AT THE TOP AND BOTTOM OF THE TOWER ***************************************
        alpha = np.radians(alpha)                                                # Conversion to radians
        phi = 0.7                                                                # ACI 307-08 strength reduction factor, section 5
        Es = 29000                                                               # Modulus of elasticity of reinforcement, in ksi
        fy = 36                                                                  # Specified yield strength of reinforcing steel bars, in ksi
        Ke = Es/fy                                                               # Yield strain of reinforcing steel bars, in/in

        # ************************** DEFINITIONS OF OPENINGS ****************************************************************************
        beta = 0                                                                 # One-half of the central angle subtended by an opening on concrete shell cross section
        gamma = 0                                                                # One-half of the central angle subtended by the centerlines of two openings on concrete shell cross section
        n1 = 0                                                                   # Number of openings entirely in compression zone
        rho_t = rhot                                                             # Ratio of total area of vertical reinforcement to total area of concrete shell cross section
        fp_c = 4                                                                 # Specified compressive strength of concrete, in ksi
        omega_t = rho_t*fy/fp_c                                                  # Ratio of compressive force on concrete and tension force on steel bars

        # ************************** ESTIMATION COMPRESSIVE STRENGTH OF CONCRETE ********************************************************
        if fp_c > 4000:
            beta1 = max(0.85 - 0.05*(fp_c - 4000)/1000, 0.65)
        else:
            beta1 = 0.85

        # ************************** MAXIMUM CONCRETE COMPRESSIVE STRAIN ****************************************************************
        em = min(0.003, 0.07*(1 - np.cos(alpha))/(1 + np.cos(alpha)))

        # ************************** ANGLES FROM THE STRAIN DIAGRAM *********************************************************************
        tau = np.arccos(1 - beta1*(1 - np.cos(alpha)))
        psi = np.arccos(max(-1,np.cos(alpha) - (1 - np.cos(alpha))/em*(fy/Es)))
        mu = np.arccos(min(1,np.cos(alpha) + (1 - np.cos(alpha))/em*(fy/Es)))
        Q1 = (np.sin(psi) - np.sin(mu) - (psi - mu)*np.cos(alpha))/(1 - np.cos(alpha))
        lamda = tau - n1*beta
        lambda1 = mu + psi - np.pi

        # ************************** ANGLES FROM THE STRESS DIAGRAM *********************************************************************
        if alpha > np.radians(35):
            Q = 0.89
        elif alpha > np.radians(25):
            Q = (0.993 - 0.00258*alpha) + (-3.27 + 0.0862*alpha)*(t_tow/rm_tow)
        elif alpha > np.radians(17):
            Q = (-1.345 + 0.2018*alpha + 0.004434*alpha**2) + (15.83 - 1.676*alpha + 0.03994*alpha**2)*(t_tow/rm_tow)
        elif alpha > np.radians(10):
            Q = (-0.488 + 0.076*alpha) + (9.758 - 0.640*alpha)*(t_tow/rm_tow)
        elif alpha > np.radians(5):
            Q = (-0.154 + 0.01773*alpha + 0.00249*alpha**2) + (16.42 - 1.980*alpha + 0.0674*alpha**2)*(t_tow/rm_tow)
        else:
            Q = (-0.523 + 0.181*alpha - 0.0154*alpha**2) + (41.3 - 13.2*alpha + 1.32*alpha**2)*(t_tow/rm_tow)

        # ************************** ESTIMATING NOMINAL MOMENT STRENGTH *****************************************************************
        stigma = 4*np.cos(alpha)*(np.sin(alpha) + np.sin(psi) - np.sin(mu))
        K = np.sin(psi) + np.sin(mu) + (np.pi - psi - mu)*np.cos(alpha)
        R_bar = np.sin(tau) - (tau - n1*beta1)*np.cos(alpha) - (0.5*n1)*(np.sin(gamma + beta) - np.sin(gamma - beta))
        Q2 = ((psi - mu)*(1 + 2*(np.cos(alpha))**2) + 0.5*(4*np.sin(2*alpha) + np.sin(2*psi) - np.sin(2*mu)) - stigma)/(1 - np.cos(alpha))
        K1 = 1.7*Q*lamda + 2*em*Ke*omega_t*Q1 + 2*omega_t*lambda1
        K2 = 1.7*Q*R_bar + em*Ke*omega_t*Q2 + 2*omega_t*K
        K3 = np.cos(alpha) + K2/K1
        Pu = K1*rm_tow*t_tow*fp_c*12**2
        Mn = phi*Pu*rm_tow*K3
        return Mn

if __name__=='__main__':
    tower = ConcreteTower()
    cost, safety_factor, iteration_residual = tower.calculate_cost()
    print('{0:.0f}'.format(cost))

