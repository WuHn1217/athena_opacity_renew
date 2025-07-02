import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

import sys
sys.path.insert(0, '../vis/python')
import athena_read

G = 6.67e-8
M_s = 1.989e33
T_unit = 1.227e7
density_unit = 4.4445e-23
length_unit = 2.99195e11
AU = 1.496e13
mol_weight = 2.3
gamma = 1.6667
h_plank = 6.626e-27
k_boltz = 1.38e-16
c = 3e10
rstar = 1.39268e11
tstar = 5500

ONE_PI_FOUR_POWER = 1.0 / (np.pi ** 4)


# fre = np.array([0.000000e+00, 1.602839e-06, 3.453212e-06, 7.439719e-06, 1.602839e-05,
#                 3.453212e-05, 7.439719e-05, 1.602839e-04, 3.453212e-04, 7.439719e-04,
#                 1.602839e-03, 3.453212e-03, 7.439719e-03, 1.602839e-02, 3.453212e-02,
#                 7.439719e-02, 1.602839e-01])
#
# nu_Hz = fre * k_boltz * T_unit / h_plank
# print(nu_Hz)

def fit_int_planck_func(nu_t):
    """
    Return the integral of normalized Planck function from 0 to nu_t.
    nu_t: dimensionless frequency (h*nu / kT)
    """
    nu_2 = nu_t ** 2
    nu_3 = nu_t * nu_2
    nu_7 = nu_2 * nu_2 * nu_3

    if nu_t < 1.9434:
        integral = (0.051329911273422 * nu_3
                    - 0.019248716727533 * nu_t * nu_3
                    + 0.002566495563671 * nu_2 * nu_3
                    - 3.055351861513195e-5 * nu_7)
    elif nu_t < 5.23754:
        logfre = np.log10(nu_t)
        integral = (-0.6874 * logfre**3
                    - 0.5698 * logfre**2
                    + 2.671 * logfre
                    - 1.476)
        integral = 10 ** integral
    elif nu_t < 50.0:
        exp_term = np.exp(-nu_t)
        integral = 1.0 - 15.0 * ONE_PI_FOUR_POWER * exp_term * (
            6.0 + 6.0 * nu_t + 3.0 * nu_t**2 + nu_t**3)
    else:
        integral = 1.0

    return integral


def int_planck_func(nu_min, nu_max):

    return fit_int_planck_func(nu_max) - fit_int_planck_func(nu_min)

def B(fre, tem):
    index = h_plank * fre / k_boltz / tem
    if index > 700:
        index = 700
    exp_index = np.exp(index) - 1
    B = 2*h_plank*(fre**3)/(c**2) / exp_index
    return B

def planck_mean(tem, fre, kap):
    dfre = np.diff(fre)
    dfre = np.append(dfre, dfre[-1])

    kap_pm_1 = 0
    kap_pm_2 = 0
    for i in range(len(fre)):
        kap_pm_1 += kap[i] * B(fre[i], tem) * dfre[i]
        kap_pm_2 += B(fre[i], tem) * dfre[i]
    kap_pm = kap_pm_1 / kap_pm_2

    return kap_pm

def planck_sum(tem, fre, kap):
    dfre = np.diff(fre)
    dfre = np.append(dfre, dfre[-1])

    kap_pm_1 = 0
    for i in range(len(fre)):
        kap_pm_1 += kap[i] * B(fre[i], tem) * dfre[i]

    return kap_pm_1

def read1(frame_number):
    data = athena_read.athdf('../outputs/inner_disk/fro_17/disk.out1.'+'%05d'%frame_number+'.athdf')
    return data

def read2(frame_number):
    data = athena_read.athdf('../outputs/inner_disk/fro_18/disk.out1.'+'%05d'%frame_number+'.athdf')
    return data

def read3(frame_number):
    data = athena_read.athdf('../outputs/inner_disk/fro_19/disk.out1.'+'%05d'%frame_number+'.athdf')
    return data

def read4(frame_number):
    data = athena_read.athdf('../outputs/inner_disk/fro_21/disk.out1.'+'%05d'%frame_number+'.athdf')
    return data

def get_rho_index(rho):
    if rho < 1e-14:
        return 0
    elif rho > 1.0:
        return 299
    else:
        return int((299.0 / 14.0) * np.log10(rho) + 299)

def get_tem_index(tem):
    if tem < 1.0:
        return 0
    elif tem > 1e7:
        return 299
    else:
        return int((299.0 / 7.0) * np.log10(tem))

# data = read(20)
#
# x1c = data['x1v']
# dis = x1c * length_unit
# x2c = data['x2v']
# rho = data['rho']
# press = data['press']
# tem = press / rho * T_unit
# sigmap = [data[f'Sigma_p_{i}'] for i in range(17)]
# kap_pm_tdisk_fre = sigmap / rho / (density_unit * length_unit)
# # print(1/(density_unit * length_unit))
# rho = rho * density_unit
#
# kap_pm_fre_table0 = np.loadtxt("../outputs/inner_disk/opa_table/sparse/kappa_pm_table.txt")
# fre_table = np.loadtxt("../outputs/inner_disk/opa_table/sparse/fre_table.txt")
# tem_table = np.loadtxt("../outputs/inner_disk/opa_table/sparse/tem_table.txt")
# rho_table = np.loadtxt("../outputs/inner_disk/opa_table/sparse/rho_table.txt")
# i = kap_pm_fre_table0[:, 0].astype(int)
# j = kap_pm_fre_table0[:, 1].astype(int)
# k = kap_pm_fre_table0[:, 2].astype(int)
# v = kap_pm_fre_table0[:, 3]
# kap_pm_fre_table = np.zeros((300, 300, 16))
# kap_pm_fre_table[i, j, k] = v
#
# n_2 = 30
#
# Td_approx = np.zeros(len(x1c))
# for n_1 in range(0, len(x1c)):
#     print(n_1)
#
#     # tem_cal0 = np.arange(tem[0, n_2, n_1]-2000, tem[0, n_2, n_1]+2000, 5)
#     tem_cal0 = np.arange(0, tem[0, n_2, n_1] + 2000, 5)
#     absorption = np.zeros(len(tem_cal0))
#     emission = np.zeros(len(tem_cal0))
#
#     for j in range(len(tem_cal0)):
#
#         rho_index = get_rho_index(rho[0, n_2, n_1])
#         tem_index = get_tem_index(tem_cal0[j])
#         k00_pm = np.log10(kap_pm_fre_table[rho_index, tem_index, :])
#         k01_pm = np.log10(kap_pm_fre_table[rho_index, tem_index + 1, :])
#         k10_pm = np.log10(kap_pm_fre_table[rho_index + 1, tem_index, :])
#         k11_pm = np.log10(kap_pm_fre_table[rho_index + 1, tem_index + 1, :])
#         # k00_pm = kap_pm_fre_table[rho_index, tem_index, :]
#         # k01_pm = kap_pm_fre_table[rho_index, tem_index + 1, :]
#         # k10_pm = kap_pm_fre_table[rho_index + 1, tem_index, :]
#         # k11_pm = kap_pm_fre_table[rho_index + 1, tem_index + 1, :]
#
#         if rho[0, n_2, n_1] < 1e-14:
#             r = 0.0
#         else:
#             r = (np.log(rho[0, n_2, n_1]) - np.log(rho_table[rho_index])) / \
#                 (np.log(rho_table[rho_index + 1]) - np.log(rho_table[rho_index]))
#         if tem_cal0[j] < 1.0:
#             t = 0.0
#         else:
#             t = (np.log(tem_cal0[j]) - np.log(tem_table[tem_index])) / \
#                 (np.log(tem_table[tem_index + 1]) - np.log(tem_table[tem_index]))
#         kappa_pm = (1 - t) * (1 - r) * k00_pm + t * (1 - r) * k01_pm + (1 - t) * r * k10_pm + t * r * k11_pm
#         kappa_pm = pow(10, kappa_pm)
#
#         absorption[j] = pow((rstar/2/dis[n_1]), 2) * planck_sum(tstar, fre_table, kappa_pm)
#         emission[j] = planck_sum(tem_cal0[j], fre_table, kappa_pm)
#         # emission[j] = planck_sum(T_unit, fre_table, kappa_pm)
#
#     diff = absorption - emission
#     threshold = 10
#     indices = np.where(diff < threshold)[0]
#     if len(indices) == 0:
#         print(n_1)
#         i_min = np.argmin(diff)
#     else:
#         i_min = indices[0]
#     # i_min = np.argmax(diff)
#     Td_approx[n_1] = tem_cal0[i_min]
#
#     # print(Td_approx[n_1])
#     # plt.plot(tem_cal0, diff)
#     # plt.plot(tem_cal0, np.zeros(len(tem_cal0)))
#     # plt.show()
#
# tem_sim = tem[0, n_2, :]
# print(tem_sim[2])
# print(get_tem_index(tem_sim[2]))
#
# rho_sim = rho[0, n_2, :]
# print(rho_sim[2])
# print(get_rho_index(rho_sim[2]))
#
# print(kap_pm_tdisk_fre[:, 0, n_2, 2])
#
# plt.plot(x1c*length_unit/AU, Td_approx, label='tem_cal')
# plt.plot(x1c*length_unit/AU, tem_sim, label='tem_sim')
# plt.plot(x1c*length_unit/AU, tstar*pow((rstar/2/dis), 0.5), label='tem_inver_squa')
# # plt.xscale('log')
# # plt.yscale('log')
# plt.legend()
# # plt.savefig("/Users/wuhening/files/radiation_hydro/outburst/figures/disk_tem.png", bbox_inches='tight', dpi=600)
# plt.show()


data1 = read1(0)
x1c = data1['x1v']
dis = x1c * length_unit
rho = data1['rho']
press = data1['press']
tem1 = press / rho * T_unit
tem1 = tem1[0, 24, :]

data2 = read1(2)
x1c = data2['x1v']
dis = x1c * length_unit
rho = data2['rho']
press = data2['press']
tem2 = press / rho * T_unit
tem2 = tem2[0, 24, :]

data3 = read1(4)
x1c = data3['x1v']
dis = x1c * length_unit
rho = data3['rho']
press = data3['press']
tem3 = press / rho * T_unit
tem3 = tem3[0, 24, :]

data4 = read1(6)
x1c = data4['x1v']
dis = x1c * length_unit
rho = data4['rho']
press = data4['press']
tem4 = press / rho * T_unit
tem4 = tem4[0, 24, :]

data5 = read1(8)
x1c = data5['x1v']
dis = x1c * length_unit
rho = data5['rho']
press = data5['press']
tem5 = press / rho * T_unit
tem5 = tem5[0, 24, :]

# data6 = read1(10)
# x1c = data6['x1v']
# dis = x1c * length_unit
# rho = data6['rho']
# press = data6['press']
# tem6 = press / rho * T_unit
# tem6 = tem6[0, 24, :]

plt.plot(x1c*length_unit/AU, tem1, color=plt.get_cmap('Purples')(0.8), label='initial')
plt.plot(x1c*length_unit/AU, tem2, color=plt.get_cmap('Blues')(0.2))
plt.plot(x1c*length_unit/AU, tem3, color=plt.get_cmap('Blues')(0.4))
plt.plot(x1c*length_unit/AU, tem4, color=plt.get_cmap('Blues')(0.6), label='ite=100')
plt.plot(x1c*length_unit/AU, tem5, color=plt.get_cmap('Blues')(0.8))
# plt.plot(x1c*length_unit/AU, tem6, color=plt.get_cmap('Blues')(1.0))


data1 = read2(0)
x1c = data1['x1v']
dis = x1c * length_unit
rho = data1['rho']
press = data1['press']
tem1 = press / rho * T_unit
tem1 = tem1[0, 24, :]

data2 = read2(2)
x1c = data2['x1v']
dis = x1c * length_unit
rho = data2['rho']
press = data2['press']
tem2 = press / rho * T_unit
tem2 = tem2[0, 24, :]

data3 = read2(4)
x1c = data3['x1v']
dis = x1c * length_unit
rho = data3['rho']
press = data3['press']
tem3 = press / rho * T_unit
tem3 = tem3[0, 24, :]

data4 = read2(6)
x1c = data4['x1v']
dis = x1c * length_unit
rho = data4['rho']
press = data4['press']
tem4 = press / rho * T_unit
tem4 = tem4[0, 24, :]

data5 = read2(8)
x1c = data5['x1v']
dis = x1c * length_unit
rho = data5['rho']
press = data5['press']
tem5 = press / rho * T_unit
tem5 = tem5[0, 24, :]

# data6 = read2(10)
# x1c = data6['x1v']
# dis = x1c * length_unit
# rho = data6['rho']
# press = data6['press']
# tem6 = press / rho * T_unit
# tem6 = tem6[0, 24, :]

#plt.plot(x1c*length_unit/AU, tem1, color=plt.get_cmap('Greens')(0.2), label='time=0.0')
plt.plot(x1c*length_unit/AU, tem2, color=plt.get_cmap('Greens')(0.2))
plt.plot(x1c*length_unit/AU, tem3, color=plt.get_cmap('Greens')(0.4))
plt.plot(x1c*length_unit/AU, tem4, color=plt.get_cmap('Greens')(0.6), label='ite=50')
plt.plot(x1c*length_unit/AU, tem5, color=plt.get_cmap('Greens')(0.8))
# plt.plot(x1c*length_unit/AU, tem6, color=plt.get_cmap('Greens')(1.0))

data1 = read3(0)
x1c = data1['x1v']
dis = x1c * length_unit
rho = data1['rho']
press = data1['press']
tem1 = press / rho * T_unit
tem1 = tem1[0, 24, :]

data2 = read3(1)
x1c = data2['x1v']
dis = x1c * length_unit
rho = data2['rho']
press = data2['press']
tem2 = press / rho * T_unit
tem2 = tem2[0, 24, :]

data3 = read3(2)
x1c = data3['x1v']
dis = x1c * length_unit
rho = data3['rho']
press = data3['press']
tem3 = press / rho * T_unit
tem3 = tem3[0, 24, :]

data4 = read3(3)
x1c = data4['x1v']
dis = x1c * length_unit
rho = data4['rho']
press = data4['press']
tem4 = press / rho * T_unit
tem4 = tem4[0, 24, :]

data5 = read3(4)
x1c = data5['x1v']
dis = x1c * length_unit
rho = data5['rho']
press = data5['press']
tem5 = press / rho * T_unit
tem5 = tem5[0, 24, :]

# data6 = read3(5)
# x1c = data6['x1v']
# dis = x1c * length_unit
# rho = data6['rho']
# press = data6['press']
# tem6 = press / rho * T_unit
# tem6 = tem6[0, 24, :]

#plt.plot(x1c*length_unit/AU, tem1, color=plt.get_cmap('Reds')(0.2), label='time=0.0')
plt.plot(x1c*length_unit/AU, tem2, color=plt.get_cmap('Reds')(0.2))
plt.plot(x1c*length_unit/AU, tem3, color=plt.get_cmap('Reds')(0.4))
plt.plot(x1c*length_unit/AU, tem4, color=plt.get_cmap('Reds')(0.6), label='ite=25')
plt.plot(x1c*length_unit/AU, tem5, color=plt.get_cmap('Reds')(0.8))
# plt.plot(x1c*length_unit/AU, tem6, color=plt.get_cmap('Reds')(1.0))

# data1 = read4(0)
# x1c = data1['x1v']
# dis = x1c * length_unit
# rho = data1['rho']
# press = data1['press']
# tem1 = press / rho * T_unit
# tem1 = tem1[0, 24, :]
#
# data2 = read4(3)
# x1c = data2['x1v']
# dis = x1c * length_unit
# rho = data2['rho']
# press = data2['press']
# tem2 = press / rho * T_unit
# tem2 = tem2[0, 24, :]
#
# data3 = read4(6)
# x1c = data3['x1v']
# dis = x1c * length_unit
# rho = data3['rho']
# press = data3['press']
# tem3 = press / rho * T_unit
# tem3 = tem3[0, 24, :]
#
# data4 = read4(9)
# x1c = data4['x1v']
# dis = x1c * length_unit
# rho = data4['rho']
# press = data4['press']
# tem4 = press / rho * T_unit
# tem4 = tem4[0, 24, :]
#
# data5 = read4(12)
# x1c = data5['x1v']
# dis = x1c * length_unit
# rho = data5['rho']
# press = data5['press']
# tem5 = press / rho * T_unit
# tem5 = tem5[0, 24, :]

# plt.plot(x1c*length_unit/AU, tem1, color=plt.get_cmap('Reds')(0.2), label='time=0.0')
# plt.plot(x1c*length_unit/AU, tem2, color=plt.get_cmap('Reds')(0.4), label='time=0.0')
# plt.plot(x1c*length_unit/AU, tem3, color=plt.get_cmap('Reds')(0.6), label='time=0.0')
# plt.plot(x1c*length_unit/AU, tem4, color=plt.get_cmap('Reds')(0.8), label='time=0.0')
# plt.plot(x1c*length_unit/AU, tem5, color=plt.get_cmap('Reds')(1.0), label='time=0.0')


plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig("/Users/wuhening/files/radiation_hydro/outburst/figures/iteration.png", bbox_inches='tight', dpi=600)
plt.show()
