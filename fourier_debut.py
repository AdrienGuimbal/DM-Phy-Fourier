#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 14:44:14 2022

@author: Adrien Licari-Guillaume
@author 2: Adrien Guimbal
"""

import numpy as np
import matplotlib.pyplot as plt

π = np.pi
τ = 2 * π
sqrt2 = np.sqrt(2)
cos = np.cos

# %%
#   Synthèse spectrale carré

def synthèse(liste_f   : np.array,
             liste_amp : np.array,
             liste_phi : np.array,
             n_périodes = 5) -> (np.array, np.array):
    """
    Génère un signal à partir de la liste des fréquences et de la donnée
    du spectre.
    Renvoie deux tableaux : les temps et les valeurs du signal
    """
    # on part du principe que la liste des fréquences est dans triée dans l'ordre croissant
    t_max = n_périodes/liste_f[1]
    nb_points = int(t_max * liste_f[-1] * 50) 
    
    t = np.linspace(0, t_max, nb_points)
    s = lambda t : sum(A * cos(τ*f*t + phi) for f, A, phi in zip(liste_f, liste_amp, liste_phi))
    # s(t) = Σ A·cos(2πf·t + φ)
    s = np.vectorize(s) # pour appliquer la fontion a un tableau np
    
    return t, s(t)

# %%
#   Signal carré

f_carré, A_carré, phi_carré = np.loadtxt("spectre_carre.dat", skiprows=1, unpack=True)

for i, nb in enumerate((2, 5, 20, 51)) :
    plt.figure(i)
    plt.title("Signal carré %i harmonniques" % nb)
    t_carré, s_carré = synthèse(f_carré[:nb], A_carré[:nb], phi_carré[:nb])
    plt.plot(t_carré, s_carré)
plt.show()

del i, f_carré, A_carré, phi_carré, t_carré, s_carré, nb

# %%
#   Températures

plt.title("Température Marseille")

t, T = np.loadtxt("temperatures_marseille.dat", skiprows=1, unpack=True)
f_temp, A_temp, phi_temp = np.loadtxt("spectre_temperatures.dat", skiprows=1, unpack=True)
t_spectre, T_spectre = synthèse(f_temp, A_temp, phi_temp, 1)

plt.plot(t, T, label="relevé")
plt.plot(t_spectre, T_spectre, label="spectre")
plt.legend(loc="lower right")

plt.show()

plt.title("Spectre température")
plt.bar(f_temp, A_temp, width=0.0032)
plt.show()

del t, T, f_temp, A_temp, phi_temp, t_spectre, T_spectre

# %%
#   Signal Mystère

f_x, A_x, phi_x = np.loadtxt("spectre_x.dat", skiprows=1, unpack=True)
f_y, A_y, phi_y = np.loadtxt("spectre_y.dat", skiprows=1, unpack=True)

t_x, x = synthèse(f_x, A_x, phi_x, 2)
t_y, y = synthèse(f_y, A_y, phi_y, 2)

plt.title("Spectre Mystère")
plt.plot(t_x, x, label="x")
plt.plot(t_y, y, label="y")
plt.legend(loc="lower right")
plt.show()

plt.title("Spectre Mystère y(x)")
plt.plot(x, y)
plt.show()

del f_x, f_y, A_x, A_y, phi_x, phi_y, t_x, t_y, x, y

# %%
#   Passe-bas
def passe_bas_1(liste_f   : np.array,
                liste_A   : np.array,
                liste_phi : np.array,
                f_coupure : float) -> (np.array, np.array, np.array):
    """
    Renvoie le spectre filtré par le fonction de transfert
        H = (1 + 𝒋f/fc)⁻¹
    (passe-bas d'ordre 1')
    """
        
    G = lambda f : 1/np.sqrt(1 + (f/f_coupure)**2) # Gain
    dec = lambda f : -np.arctan(f/f_coupure) # decalage de phase
    G, dec = np.vectorize(G), np.vectorize(dec)
    
    # f_sortie = liste_f
    A_sortie = liste_A * G(liste_f)
    phi_sortie = liste_phi + dec(liste_f)
    
    return liste_f, A_sortie, phi_sortie

def passe_bas_2(liste_f   : np.array,
                liste_A   : np.array,
                liste_phi : np.array,
                f_coupure : float) -> (np.array, np.array, np.array):
    """
    Renvoie le spectre filtré par le fonction de transfert
        H = (1 -(f/fc)² + 𝒋√2 f/fc)⁻¹
    (passe-bas d'ordre 2')
    """
    
    G = lambda f : 1/np.sqrt(1 + (f/f_coupure)**4) # Gain
    dec = lambda f : π/2 - np.arctan((f/f_coupure - f_coupure/np.float64(f))/sqrt2) # decalage de phase
    G, dec = np.vectorize(G), np.vectorize(dec)
    
    # f_sortie = liste_f
    A_sortie = liste_A * G(liste_f)
    phi_sortie = liste_phi + dec(liste_f)
    
    return liste_f, A_sortie, phi_sortie

# %%
#   Passe-bas sur carré Q9

f_carré, A_carré, phi_carré = np.loadtxt("spectre_carre.dat", skiprows=1, unpack=True)

f_s1, A_s1, phi_s1 = passe_bas_1(f_carré, A_carré, phi_carré, 150)
f_s2, A_s2, phi_s2 = passe_bas_2(f_carré, A_carré, phi_carré, 150)

t0, s0 = synthèse(f_carré, A_carré, phi_carré, 4)
t1, s1 = synthèse(f_s1, A_s1, phi_s1, 4)
t2, s2 = synthèse(f_s2, A_s2, phi_s2, 4)

plt.title("Signal carré avant/après passe-bas 150Hz")
plt.plot(t0, s0, label="sans filtre", color="0.8")
plt.plot(t1, s1, label="filtre 1")
plt.plot(t2, s2, label="filtre 2")

plt.legend(loc="lower right")
plt.show()

del f_s1, A_s1, phi_s1, f_s2, A_s2, phi_s2
del t0, s0, t1, s1, t2, s2

# %%
#   Passe-bas sur carré Q10

try :
    f_carré, A_carré, phi_carré    
except NameError:
    f_carré, A_carré, phi_carré = np.loadtxt("spectre_carre.dat", skiprows=1, unpack=True)

f_s1, A_s1, phi_s1 = passe_bas_1(f_carré, A_carré, phi_carré, 10)
f_s2, A_s2, phi_s2 = passe_bas_2(f_carré, A_carré, phi_carré, 10)

t0, s0 = synthèse(f_carré, A_carré, phi_carré, 4)
t1, s1 = synthèse(f_s1, A_s1, phi_s1, 4)
t2, s2 = synthèse(f_s2, A_s2, phi_s2, 4)

plt.title("Signal carré avant/après passe-bas 10Hz")
plt.plot(t0, s0, label="sans filtre", color="0.8")
plt.plot(t1, s1, label="filtre 1")
plt.plot(t2, s2, label="filtre 2")

plt.legend(loc="lower right")
plt.show()

del f_s1, A_s1, phi_s1, f_s2, A_s2, phi_s2
del t0, s0, t1, s1, t2, s2

# %%
#   Passe-bas sur carré Q11

try :
    f_carré, A_carré, phi_carré    
except NameError:
    f_carré, A_carré, phi_carré = np.loadtxt("spectre_carre.dat", skiprows=1, unpack=True)

f_s2, A_s2, phi_s2 = passe_bas_2(f_carré, A_carré, phi_carré, 3e-7)

t0, s0 = synthèse(f_carré, A_carré, phi_carré, 4)
t2, s2 = synthèse(f_s2, A_s2, phi_s2, 4)

plt.title("Signal carré avant/après passe-bas 3·10⁻³ h⁻¹")
plt.plot(t0, s0, label="sans filtre", color="0.8")
plt.plot(t2, s2, label="filtre 2", color="orange")

plt.legend(loc="lower right")
plt.show()

del f_s2, A_s2, phi_s2
del t0, s0, t2, s2

# %%
#   Passe-bande

def passe_bande(liste_f   : np.array,
                liste_A   : np.array,
                liste_phi : np.array,
                f_coupure : float) -> (np.array, np.array, np.array):

    z = lambda f : (f/f_coupure - f_coupure/np.float64(f)) # fonction pour alléger les calculs
    
    G = lambda f : 1/np.sqrt(1 + 100 * z(f)**4) # Gain
    dec = lambda f : - np.arctan(10 * z(f)) # decalage de phase
    G, dec = np.vectorize(G), np.vectorize(dec)
    
    # f_sortie = liste_f
    A_sortie = liste_A * G(liste_f)
    phi_sortie = liste_phi + dec(liste_f)
    
    return liste_f, A_sortie, phi_sortie

# %%
#   Passe-bande sur carré Q11

try :
    f_carré, A_carré, phi_carré    
except NameError:
    f_carré, A_carré, phi_carré = np.loadtxt("spectre_carre.dat", skiprows=1, unpack=True)

f_filtré, A_filtré, phi_filtré = passe_bande(f_carré, A_carré, phi_carré, 372)

t_carré, s_carré = synthèse(f_carré, A_carré, phi_carré, 4)
t_filtré, s_filtré = synthèse(f_filtré, A_filtré, phi_filtré, 4)

plt.title("Signal carré avant/après passe-bande ")
plt.plot(t_carré, s_carré, label="sans filtre", color="0.8")
plt.plot(t_filtré, s_filtré, label="après passe-bande", color="orange")

plt.legend(loc="lower right")
plt.show()

#%%
del f_filtré, A_filtré, phi_filtré, t_carré, s_carré, t_filtré, s_filtré










