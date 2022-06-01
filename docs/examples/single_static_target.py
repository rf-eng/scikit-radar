import numpy as np
import matplotlib.pyplot as plt
import skradar

r = 10.1
B = 1e9
fc = 76.5e9
N = 1024
f_s = 1e6
s_if = skradar.sim_FMCW_if(2*r, B, fc, N, 1/f_s, cplx=True)
range_profile, range_vec = skradar.range_compress_FMCW(s_if, B, zp_fact=4)

plt.figure(1)
plt.clf()
plt.subplot(2, 1, 1)
plt.cla()
plt.plot(range_vec/2, 20*np.log10(np.abs(range_profile)))
plt.subplot(2, 1, 2)
plt.cla()
plt.plot(range_vec/2, np.angle(range_profile))

print(range_vec[np.argmax(np.abs(range_profile))]/2-r)