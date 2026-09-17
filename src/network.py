#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 8 10:28:17 2026

@author: melindakrouri
"""
# Reseau de neurones - AlphaZero
# d'apres les slides du cours (AlphaGo Zero, Silver 2017)
#
# le reseau prend un etat s en entree et sort :
#   P : probabilites sur chaque action (politique)
#   V : valeur de l'etat (entre -1 et +1)
# soit : (P, V) = f_theta(s)

import torch

# dimensions
LIGNES     = 6
COLONNES   = 7
taille_entree = LIGNES * COLONNES  # 42 cases aplaties

# un seul reseau comme dans learning.py du cours
# les 7 premieres sorties = politique P
# la derniere sortie      = valeur V
net = torch.nn.Sequential(
    torch.nn.Linear(taille_entree, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 8)   # 7 actions + 1 valeur
)

# initialisation des poids comme dans le cours
for p in net.parameters():
    torch.nn.init.uniform_(p, -0.1, 0.1)


def reseau(s):
    # (P, V) = f_theta(s)  d'apres le slide 60 du cours
    sortie = net(s)
    P = torch.softmax(sortie[:7], dim=-1)  # politique : somme a 1
    V = torch.tanh(sortie[7])              # valeur : entre -1 et +1
    return P, V


def etat_vers_tenseur(grille):
    # on aplatit la grille 6x7 en un vecteur de 42 valeurs
    # comme state_tensor() dans learning.py du cours
    return torch.FloatTensor(grille.flatten())


# ---------------------------------------------------------
# test rapide
# ---------------------------------------------------------
if __name__ == '__main__':

    from jeu import Puissance4

    jeu = Puissance4()
    jeu.reinitialiser()

    s = etat_vers_tenseur(jeu.etat_pour_reseau())
    P, V = reseau(s)

    print('etat (aplati) :', s)
    print('P (probabilites) :', P)
    print('V (valeur)       :', V.item())
    print('somme de P       :', P.sum().item())  # doit valoir 1.0