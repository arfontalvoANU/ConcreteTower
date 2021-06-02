from colored import fg
from tower_design import *
import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')

# Tower geometry Abengoa report
D_receiver = 20.                                                      # [m]
H_receiver = 25.                                                      # [m]
Q_rec_out = 936                                                       # [MWth]
Tower_height = 268.                                                   # [m]
Section_height = 5.                                                   # [m]
Thickness_bottom = 0.566                                              # [m]
Thickness_top = 0.447                                                 # [m]
Diameter_bottom = 30.4                                                # [m]
Diameter_top = 20.2                                                   # [m]
verbose = True                                                        # True to print intermediate output

Receiver_weight = scaling_w_receiver(D_receiver, H_receiver, Q_rec_out)

C_tower,sf_min,res_min = tower(Tower_height,Section_height,Receiver_weight,Thickness_bottom,Thickness_top,Diameter_bottom,Diameter_top,verbose)
