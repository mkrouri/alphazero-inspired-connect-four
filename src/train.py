#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 14:05:10 2026

@author: melindakrouri
"""


import torch
import numpy
import random

from game import Puissance4
from network import reseau, net, etat_vers_tenseur
from mcts import mcts_recherche

# hyperparametres
nb_episodes = 200    # nombre de parties de self-play
lr          = 0.001  # learning rate comme dans les TP
c           = 0.001  # coefficient regularisation L2

optimizer = torch.optim.Adam(net.parameters(), lr=lr)

# ------------------------------
# boucle d'entrainement principale
# ------------------------------------------
jeu = Puissance4()

print('debut de l entrainement...')
print(f'nb_episodes = {nb_episodes}')
print()

for episode in range(nb_episodes):

    # --------------
    # 1. SELF-PLAY : on joue une partie complete
    #    et on stocke (etat, politique, joueur) a chaque coup
    # --------------------------
    jeu.reinitialiser()
    historique = []

    while not jeu.est_termine():
        # le MCTS donne la politique pour le coup courant
        politique = mcts_recherche(jeu)

        # on sauvegarde l'etat et la politique
        etat = jeu.etat_pour_reseau().copy()
        historique.append((etat, politique, jeu.joueur_courant))

        # on choisit l'action proportionnellement aux visites
        # pi_a(s) proportionnel a N(s,a)  (slide 60)
        action = numpy.random.choice(7, p=politique)
        jeu.jouer(action)

    # resultat final de la partie : z = +1, -1 ou 0
    z = jeu.resultat()

    # -----------------------------------
    # 2. CALCUL DE LA LOSS:
    # loss = (z - V)^2 - pi^T * log(P) + c * ||theta||^2
    # -------------------------------------------------------
    optimizer.zero_grad()
    loss_totale = torch.tensor(0.0)

    for etat, politique, joueur in historique:
        s  = etat_vers_tenseur(etat)
        P, V = reseau(s)

        pi        = torch.FloatTensor(politique)
        z_joueur  = torch.tensor(float(z * joueur))

        # erreur sur la valeur
        perte_valeur    = (z_joueur - V) ** 2
        # erreur sur la politique (entropie croisee)
        perte_politique = -torch.dot(pi, torch.log(P + 1e-8))

        loss_totale = loss_totale + perte_valeur + perte_politique

    loss_totale = loss_totale / len(historique)

    # regularisation L2 : c * ||theta||^2
    for param in net.parameters():
        loss_totale = loss_totale + c * torch.sum(param ** 2)

    # ------------------
    # 3. MISE A JOUR DU RESEAU
    # -------------------------------------
    loss_totale.backward()
    optimizer.step()

    if episode % 10 == 0:
        print(f'episode {episode:4d}  |  loss = {loss_totale.item():.4f}  |  coups joues = {len(historique)}')

print()
print('entrainement termine !')

# on sauvegarde le reseau entrain
torch.save(net.state_dict(), 'reseau_entraine.pth')
print('reseau sauvegarde dans reseau_entraine.pth')

# ------------------------------------
# test : reseau entraine vs joueur aleatoire
# ------------------------------
print()
print('=== test : reseau entraine vs joueur aleatoire ===')

nbVictoires = 0
nbMatchs    = 20

for i in range(nbMatchs):
    jeu.reinitialiser()
    endOfEpisode = False

    while not endOfEpisode:
        if jeu.joueur_courant == 1:
            # le reseau joue : on prend le meilleur coup
            politique = mcts_recherche(jeu)
            action    = numpy.argmax(politique)
        else:
            # adversaire aletoire
            action = random.choice(jeu.coups_legaux())

        jeu.jouer(action)
        endOfEpisode = jeu.est_termine()

    if jeu.resultat() == 1:
        nbVictoires += 1

print(f'victoires du reseau : {nbVictoires} / {nbMatchs}')
