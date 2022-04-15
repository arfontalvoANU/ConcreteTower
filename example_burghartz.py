from colored import fg
from tower_design import *
import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')

# Tower geometry 
D_receiver = 17.                                                      # [m]
H_receiver = 18.                                                      # [m]
Q_rec_out = 700                                                       # [MWth]
T_in = 290.                                                           # [deg C]
T_out = 566.                                                          # [deg C]
IsSodium = False                                                      # Set to False to use nitrate salt
Tower_height = 210.                                                   # [m]
Section_height = 5.                                                   # [m]
Thickness_bottom = 0.600                                              # [m]
Thickness_top = 0.250                                                 # [m]
Diameter_bottom = 23.2                                                # [m]
Diameter_top = 22.5                                                   # [m]
verbose = True                                                        # True to print intermediate output
load_type = 1                                                         #

Receiver_weight = scaling_w_receiver(D_receiver, H_receiver, Q_rec_out, T_in, T_out, IsSodium)

C_tower,sf_min,res_min = tower(Tower_height,Section_height,Receiver_weight,Thickness_bottom,Thickness_top,Diameter_bottom,Diameter_top,verbose,load_type,burghartz = True,min_sf = 1.0)
