import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')
import numpy as np
from scipy import optimize
from functools import partial
from tower_optimisation import *
from colored import fg

i = 0

Q = [151.0723496531,153.1658623537,152.8151618348,155.0485142915,152.2393636592,197.6590045083,201.475497697,203.6991111324,203.3278231673,
     204.2842773803,300.4223477035,309.1665884238,313.1824789973,312.5137526733,315.1012653699, 624.0082952208,628.4483169736,630.1481971056]
T = [100,125,150,175,200,100,125,150,175,200,100,125,150,175,200,150,175,200]
D = [10,9,9,8,9,12,11,10,10,9,15,13,12,12,11,16,16,15]
H = [13,13,13,13,13,15,14,14,14,15,19,18,17,17,17,26,24,25]

Tower_height = float(T[i])
D_receiver = float(D[i])
H_receiver = float(H[i])
Q_rec_out  = Q[i]
dbl = partial(tower_opt,Q_rec_out,Tower_height,D_receiver,H_receiver)
x0 = np.zeros(2)
x0[0] = 0.5
x0[1] = 0.25
b1 = (0.2,0.5)
b2 = (0.0,0.5)
bnds = (b1,b2)
res = optimize.minimize(dbl, x0, method='COBYLA', bounds=bnds, tol=1e-6)
print '%s%s,%s,%s'%(fg(11),res.x[0],res.x[1],res.fun)
