#!/usr/bin/env python3

GROW = {
'NO3' :0.22000, 'NH4' :0.00700, 'P2O3':0.00500, 'K2O' :0.24000,
'B'   :0.00030, 'Cu'  :0.00005, 'Fe'  :0.00150, 'Mn'  :0.00055,
'Mo'  :0.00005, 'Zn'  :0.00033, 'Ca'  :0.22000, 'Mg'  :0.05000,
'SO4' :0.20000}

FLWR = {
'NO3' :0.18000, 'NH4' :0.01500, 'P2O3':0.00500, 'K2O' :0.20000,
'B'   :0.00030, 'Cu'  :0.00005, 'Fe'  :0.00150, 'Mn'  :0.00055,
'Mo'  :0.00005, 'Zn'  :0.00033, 'Ca'  :0.20000, 'Mg'  :0.04500,
'SO4' :0.18000}

Ca = {'NO3':14.50, 'NH4':1.000, 'Ca':19.00}

Mg = {'Mg':20.00, 'SO4':80.00}

NPK_4_18_38 = { # Greenway Biotech Tomato Fertilizer: https://www.greenwaybiotech.com/products/tomato-fertilizer
'NO3' :3.500, 'NH4' :0.500, 'P2O3':18.00, 'K2O' :38.00, 'B'   :0.100,
'Cu'  :0.050, 'Fe'  :0.500, 'Mn'  :0.200, 'Mo'  :0.001, 'Zn'  :0.300}

NPK = NPK_4_18_38

g_npk = round((GROW['K2O'])/(NPK['K2O']/100), 2)
g_mg  = round((GROW['Mg'])/(Mg['Mg']/100), 2)
g_ca  = round((GROW['Ca'])/(Ca['Ca']/100), 2)

f_npk = round((FLWR['K2O'])/(NPK['K2O']/100), 2)
f_mg  = round((FLWR['Mg'])/(Mg['Mg']/100), 2)
f_ca  = round((FLWR['Ca'])/(Ca['Ca']/100), 2)

print('Fertiliser requirements for pepper plants:')
print(f'Grow:   NPK {g_npk}g Mg {g_mg}g Ca {g_ca}g / 1L')
print(f'Flower: NPK {f_npk}g Mg {f_mg}g Ca {f_ca}g / 1L')
print()

sample = 5
final = 100
multiple = final/sample

print('Concentrated solution (Grow):')
print(f'NPK:  {g_npk}g  in {sample}ml therefore {g_npk*multiple}g in {final}ml')
print(f'Mg:   {g_mg}g  in {sample}ml therefore {g_mg*multiple}g  in {final}ml')
print(f'Ca:   {g_ca}g  in {sample}ml therefore {g_ca*multiple}g  in {final}ml')
print()
print('Concentrated solution (Flower):')
print(f'NPK:  {f_npk}g  in {sample}ml therefore {f_npk*multiple}g in {final}ml')
print(f'Mg:   {f_mg}g  in {sample}ml therefore {f_mg*multiple}g  in {final}ml')
print(f'Ca:   {f_ca}g  in {sample}ml therefore {f_ca*multiple}g  in {final}ml')
