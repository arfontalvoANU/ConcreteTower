from tower_design import *

def tower_opt(Q_rec_out,Tower_height,D_receiver,H_receiver,x):

	# ******************************************** DEFINITION ********************************************
	# x[0]: Thickness at the top, should be minimum 25.4 cm to accommodate the ring and meridian reinforcement (Burghartz, 2016)
	# x[1]: Offset from thickness at the top. Thickness_bottom = Thickness at the top + x[1]

	Receiver_weight = scaling_w_receiver(D_receiver, H_receiver, Q_rec_out)

	Section_height = 5.                                               # [m]
	Thickness_top = x[0]                                              # [m]
	Thickness_bottom = x[0] + x[1]                                    # [m]
	Diameter_top = 1.2*D_receiver + 2*Thickness_top                   # [m]
	Diameter_bottom = 1.2*D_receiver + 2*Thickness_bottom             # [m]
	verbose = False

	if x[0] < 0.254 or x[1] < 0 :                                     # A minimum wall thickness of 25 cm is required to accommodate the ring and meridian reinforcement
		y = 1e20                                                      # Cost penalty
	else:
		y,sf_min,res_min = tower(Tower_height,Section_height,Receiver_weight,Thickness_bottom,Thickness_top,Diameter_bottom,Diameter_top,verbose)
	return y
