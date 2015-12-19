from scipy.special import beta
P_B_D = beta(6+1, 5+1) / beta(3+1, 5+1)
print P_B_D
print P_B_D/(1-P_B_D)