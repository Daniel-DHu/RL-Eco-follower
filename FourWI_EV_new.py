#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
4wEV for RL-based following plan
                                 by HuDong  2023.7.1
"""

import sys
import math
import time
import pickle
import numpy as np
import scipy.io as scio
from scipy.interpolate import interp1d,interp2d
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

class EV(object):
    '''
    this is a 4wEV with HVAC
    '''
    def __init__(self, time_c = 1):
        self.R_wheel = 0.33          # m, wheel radius
        self.mass = 1874      # kg, vehicle mass
        self.C_roll = 0.01          # Rolling resistance coefficient
        self.rho = 1.2              # kg/m^3, Density of air
        self.A_fr = 2.6             # m^2, Frontal area
        self.C_d = 0.23             # Drag coefficient
        self.gg = 9.81              # m/s^2, Acceleration due to gravity
        # Wheel speed (rad/s), Wheel acceleration (rad/s^2), Wheel torque (Nm)
        # self.G_f = 3.2
        self.G_f = 1
        self.RPM_2_rads = 2*math.pi/60

        self.V_air = 3.88           # m^3, Interior air volume
        self.n_p = 2                # number of passengers
        self.k = 1.706              # 对流换热系数
        self.A_k = 15.6             # 换热面积 m^3

        # Motor
        # motor speed list (rad/s)
        Mot_spd_list = np.arange(-1351, 1351, 10) * (2 * math.pi) / 60
        # motor torque list (Nm)
        Mot_trq_list = np.arange(-801, 801, 10)

        #  motor maximum torque (indexed by max speed list, 31 elements)
        Mot_t_maxlist = np.array([280, 320, 390, 490, 680,790, 800, 800, 800,
                                  800,800, 790, 680, 490, 390, 320, 280])
        #  motor minimum torque (indexed by max speed list)
        Mot_t_minlist = - Mot_t_maxlist
        #  motor maximum torque corresponding speed  (31 elements)
        Mot_w_maxlist = [-1350*self.RPM_2_rads,- 1200*self.RPM_2_rads, - 1000*self.RPM_2_rads,
                         - 800*self.RPM_2_rads, - 600*self.RPM_2_rads, -530*self.RPM_2_rads
            ,-400*self.RPM_2_rads,- 200*self.RPM_2_rads, 0,
            200*self.RPM_2_rads, 400*self.RPM_2_rads, 530*self.RPM_2_rads, 600*self.RPM_2_rads,
                         800*self.RPM_2_rads,1000*self.RPM_2_rads, 1200*self.RPM_2_rads,1350*self.RPM_2_rads]
        self.interp1d_mot_w2tmin = interp1d(Mot_w_maxlist, Mot_t_minlist)
        self.interp1d_mot_w2tmax = interp1d(Mot_w_maxlist, Mot_t_maxlist)
        # motor efficiency map
        data_path1 = 'MotoringEfficiency.mat'
        data1 = scio.loadmat(data_path1)
        Mot_eta_quarter_1 = data1['MotoringEfficiency']
        Mot_eta_map_1 = np.concatenate(([np.flipud(Mot_eta_quarter_1[1:, :]), Mot_eta_quarter_1]), axis=0)
        #=======Braking Efficiency============
        data_path1 = 'BrakingEfficiency.mat'
        data1 = scio.loadmat(data_path1)
        Mot_eta_quarter_2 = data1['BrakingEfficiency']
        Mot_eta_map_2 = np.concatenate(([np.flipud(Mot_eta_quarter_2[1:, :]), Mot_eta_quarter_2]), axis=0)
        # =======Total Efficiency============
        Mot_eta_map = np.concatenate(([Mot_eta_map_2, Mot_eta_map_1[:, 1:]]), axis=1) * 0.01
        self.interp2d_mot_eff = interp2d(Mot_trq_list, Mot_spd_list, Mot_eta_map)
        # plt.contour(Mot_trq_list, Mot_spd_list, Mot_eta_map,80)
        # plt.show()

        # ===========================================================================================================
        # Battery
        SOC_list = np.arange(0, 1.01, 0.1)
        # power of the battery
        self.Q_batt = 225 * 3600                #(A·s)
        #  open circuit voltage (indexed by state-of-charge list)
        V_oc = np.array([11.7,11.85,11.96,12.11,12.26,12.37,12.48,12.59,12.67,12.78,12.89]) * 24 # V
        rdis = np.array([0.0407, 0.037, 0.0338, 0.0269, 0.0193, 0.0151, 0.0131, 0.0123, 0.0117, 0.0118, 0.0122]) * 24  # ohm
        rchg = np.array([0.0316, 0.0298, 0.0295, 0.0287, 0.028, 0.0269, 0.0231, 0.025, 0.0261, 0.0288, 0.0472]) * 24   # ohm
        # Battery voltage
        self.interp1d_soc2voc = interp1d(SOC_list, V_oc, kind = 'linear', fill_value = 'extrapolate')
        self.interp1d_soc2rdis = interp1d(SOC_list, rdis, kind = 'linear', fill_value = 'extrapolate')
        self.interp1d_soc2rchg = interp1d(SOC_list, rchg, kind = 'linear', fill_value = 'extrapolate')
        
        self.time_s = time_c
        
    def limit_T(self,t,t_min,t_max):
        if t<t_min:
            return t_min
        else:
            return t_max if t>t_max else t
        
    def np_condition(self,c):
        return np.asarray(c,dtype=np.float64)

    def run(self, para_dict):
        '''
        para_dict:
        speed,acc,SOC,k
        '''
        W_axle = para_dict['speed']/self.R_wheel
        dwv = para_dict['acc']/self.R_wheel
        SOC = para_dict['SOC']

        if para_dict['speed'] > 0:
            T_fact = 1
        else:
            T_fact = 0
            
        T_axle = self.R_wheel * (self.mass * dwv * self.R_wheel + self.mass * self.gg * self.C_roll *T_fact+\
                          0.5 * self.rho * self.A_fr * self.C_d * para_dict['speed']**2)

        T_axle_F = T_axle * para_dict['k'] 
        T_axle_R = T_axle * (1 - para_dict['k'])
        #====Braking===========================
        if T_axle < 0:
            T_axle_F = T_axle * para_dict['k2']
            T_axle_R = T_axle * (1 - para_dict['k2'])
        
        T_axle_FR = T_axle_FL = T_axle_F/2
        T_axle_RR = T_axle_RL = T_axle_R/2
        
        P_axle_FR = T_axle_FR * W_axle
        P_axle_FL = T_axle_FL * W_axle
        P_axle_RR = T_axle_RR * W_axle
        P_axle_RL = T_axle_RL * W_axle
        
        W_mot = W_axle * self.G_f
        
        mot_tmin = self.interp1d_mot_w2tmin(abs(W_mot))
        mot_tmax = self.interp1d_mot_w2tmax(abs(W_mot))

        V_batt = self.interp1d_soc2voc(SOC)

        T_mot_FR = T_axle_FR / self.G_f
        T_mot_FL = T_axle_FL / self.G_f
        T_mot_RR = T_axle_RR / self.G_f
        T_mot_RL = T_axle_RL / self.G_f

        # =========================== MOTOR
        T_mot_FR = self.limit_T(T_mot_FR, mot_tmin, mot_tmax)
        T_mot_FL = self.limit_T(T_mot_FL, mot_tmin, mot_tmax)
        T_mot_RR = self.limit_T(T_mot_RR, mot_tmin, mot_tmax)
        T_mot_RL = self.limit_T(T_mot_RL, mot_tmin, mot_tmax)
        
        Mot_eff_FR = 1 if W_mot == 0 or T_mot_FR < -800 or T_mot_FR > 800 or W_mot > (1350 * self.RPM_2_rads) else \
            self.interp2d_mot_eff(W_mot, T_mot_FR)[0]
        Mot_eff_FL = 1 if W_mot == 0 or T_mot_FL < -800 or T_mot_FL > 800 or W_mot > (1350 * self.RPM_2_rads) else \
            self.interp2d_mot_eff(W_mot, T_mot_FL)[0]
        Mot_eff_RR = 1 if W_mot == 0 or T_mot_RR < -800 or T_mot_RR > 800 or W_mot > (1350 * self.RPM_2_rads) else \
            self.interp2d_mot_eff(W_mot, T_mot_RR)[0]
        Mot_eff_RL = 1 if W_mot == 0 or T_mot_RL < -800 or T_mot_RL > 800 or W_mot > (1350 * self.RPM_2_rads) else \
            self.interp2d_mot_eff(W_mot, T_mot_RL)[0]

        if Mot_eff_FR < 0.6:
            Mot_eff_FR = 0.6
        if Mot_eff_FL < 0.6:
            Mot_eff_FL = 0.6
        if Mot_eff_RR < 0.6:
            Mot_eff_RR = 0.6
        if Mot_eff_RL < 0.6:
            Mot_eff_RL = 0.6

        P_mot_FL = W_mot * T_mot_FL * Mot_eff_FL if T_mot_FL * W_mot <= 0 else W_mot * T_mot_FL / Mot_eff_FL
        P_mot_FR = W_mot * T_mot_FR * Mot_eff_FR if T_mot_FR * W_mot <= 0 else W_mot * T_mot_FR / Mot_eff_FR
        P_mot_RR = W_mot * T_mot_RR * Mot_eff_RR if T_mot_RR * W_mot <= 0 else W_mot * T_mot_RR / Mot_eff_RR
        P_mot_RL = W_mot * T_mot_RL * Mot_eff_RL if T_mot_RL * W_mot <= 0 else W_mot * T_mot_RL / Mot_eff_RL

        # ===================================================BATTERY
        P_batt = P_mot_FL + P_mot_FR + P_mot_RR + P_mot_RL
        
        # =================================================
        r = self.interp1d_soc2rdis(SOC) if P_batt > 0 else self.interp1d_soc2rchg(SOC)
        if V_batt ** 2 - 4 * r * P_batt < 0:
            P_mot_FL = P_mot_FL + P_batt - V_batt ** 2 / (4 * r)
            P_mot_FR = P_mot_FR + P_batt - V_batt ** 2 / (4 * r)
            P_mot_RR = P_mot_RR + P_batt - V_batt ** 2 / (4 * r)
            P_mot_RL = P_mot_RL + P_batt - V_batt ** 2 / (4 * r)
            
            T_mot_FL = P_mot_FL / W_mot
            T_mot_FR = P_mot_FR / W_mot
            T_mot_RR = P_mot_RR / W_mot
            T_mot_RL = P_mot_RL / W_mot
            
        # =================================================
        if T_axle < 0:
            T_mot_F = T_axle / self.G_f * para_dict['k2']  if T_axle/self.G_f * para_dict['k2'] > 2 * mot_tmin else 2 * mot_tmin
            T_mot_R = T_axle / self.G_f * (1 - para_dict['k2']) if T_axle / self.G_f * (1 - para_dict['k2']) > 2 * mot_tmin else 2 * mot_tmin
            
            # T_axle = T_axle - T_mot_F - T_mot_R
            W_mot = W_axle * self.G_f

            T_mot_FL = T_mot_FR = T_mot_F / 2
            T_mot_RL = T_mot_RR = T_mot_R / 2

            P_mot_FL = W_mot * T_mot_FL * Mot_eff_FL if T_mot_FL * W_mot <=0 else W_mot * T_mot_FL / Mot_eff_FL
            P_mot_FR = W_mot * T_mot_FR * Mot_eff_FR if T_mot_FR * W_mot <=0 else W_mot * T_mot_FR / Mot_eff_FR
            P_mot_RL = W_mot * T_mot_RL * Mot_eff_RL if T_mot_RL * W_mot <=0 else W_mot * T_mot_RL / Mot_eff_RL
            P_mot_RR = W_mot * T_mot_RR * Mot_eff_RR if T_mot_RR * W_mot <=0 else W_mot * T_mot_RR / Mot_eff_RR
            
            P_batt = P_mot_FL + P_mot_FR + P_mot_RL + P_mot_RR

        e_batt = 1 if P_batt>0 else 0.98
        Imax = 460

    # =================================================
    # Battery current
        if V_batt**2 - 4*r*P_batt +1e-10>0:
            I_batt = e_batt * ( V_batt - np.sqrt(V_batt**2 - 4*r*P_batt+1e-10))/(2*r)
        else:
            I_batt = e_batt *  V_batt /(2*r)
        # New battery state of charge
        SOC_temp = - I_batt / (self.Q_batt)*self.time_s + SOC
        P_batt = (np.conj(P_batt)+P_batt)/2
        I_batt = (np.conj(I_batt)+I_batt)/2

        INB = 1 if V_batt**2-4*r*P_batt+1e-10 < 0 or abs(I_batt) > Imax else 0
#        if para_dict['speed'] > 0 and para_dict['acc'] > 0:
        SOC = (np.conj(SOC_temp)+SOC_temp)/2
        if np.isscalar(SOC):
            if SOC>1:
                SOC = 1.0
        else:
            SOC[np.where(SOC>1)] = 1

        P_axle = T_axle * W_axle
        price_elec = P_batt/0.8/1000/3600*0.97
        cost = price_elec

        out = {}


        out['T_axle'] = T_axle         # 需求扭矩
        out['W_axle'] = W_axle          # 轮速
        out['P_axle'] = P_axle         # 轮功率
        # 
        out['W_mot'] = W_mot
        out['T_mot'] = T_mot_FR
#        out['T_mot_R'] = T_mot_RR
#         out['P_mot_F'] = P_mot_FR
#         out['P_mot_R'] = P_mot_RR
        # 
        # 
        out['I_batt'] = I_batt
        out['V_batt'] = V_batt
        out['P_batt'] = P_batt

        # # out['eff'] = eff
        out['Mot_eta'] = Mot_eff_FR
        # 
        out['price_elec'] = price_elec
        
        return SOC, cost, INB, out
            
            
if __name__ == '__main__':
   para = {}
   para['speed'] = 10
   para['acc'] = 0.8
   para['SOC'] = 0.992
   para['k'] = 0.5
   para['k2'] = 0.5
   print(para)
   EV = EV()
   SOC_new, cost, INB, out = EV.run(para)

   print("out = ", out)
   print("SOC_new = ", SOC_new)
   print ("cost = ",cost)