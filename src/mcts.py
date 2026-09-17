#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 23:39:12 2026

@author: melindakrouri
"""

# MCTS - Monte Carlo Tree Search
# d'apres les slides du cours (AlphaGo Zero, Silver 2017)
#
# au lieu de class Noeud, on utilise des dictionnaires
# comme les tableaux Q[etat, action] dans les TPs du cours :
#   N[(cle, action)] = nombre de visites
#   W[(cle, action)] = somme des valeurs V remontees
#   P[(cle, action)] = probabilite a priori donnee par le reseau

import numpy
import copy
from game import Puissance4
from network import reseau, etat_vers_tenseur

# parametre d'exploration (comme epsilon dans le cours)
c_puct = 1.0

# nombre de simulations par coup
nb_simulations = 100


def cle(jeu):
    # identifiant unique d'un etat : la grille aplatie en bytes
    # comme l'indice "state" dans les TPs
    return jeu.grille.tobytes()


def mcts_recherche(jeu_initial):

    # statistiques : on repart de zero a chaque coup
    N = {}   # nombre de visites
    W = {}   # somme des valeurs V
    P = {}   # probabilites a priori du reseau

    for _ in range(nb_simulations):
        jeu_sim = copy.deepcopy(jeu_initial)
        chemin  = []   # on memorise les (cle, action) pour le backup

        # --------------------------------------------------
        # 1. SELECTION : argmax(Q + U) comme le slide 60
        # --------------------------------------------------
        while not jeu_sim.est_termine():
            coups = jeu_sim.coups_legaux()
            c     = cle(jeu_sim)

            # si cet etat n'est pas encore dans nos stats : on s'arrete
            if (c, coups[0]) not in N:
                break

            # on calcule Q + U pour chaque coup legal
            N_total = sum(N.get((c, a), 0) for a in coups)
            meilleur_score   = -float('inf')
            meilleure_action = coups[0]

            for action in coups:
                # Q(s,a) = W / N  (slide 60)
                q = W.get((c, action), 0.0) / (N.get((c, action), 0) + 1e-8)
                # U(s,a) proportionnel a P(s,a) / (1 + N(s,a))
                u = c_puct * P.get((c, action), 1.0 / len(coups)) * numpy.sqrt(N_total) / (1 + N.get((c, action), 0))

                if q + u > meilleur_score:
                    meilleur_score   = q + u
                    meilleure_action = action

            chemin.append((c, meilleure_action))
            jeu_sim.jouer(meilleure_action)

        # -------------------------------
        # 2. EXPANSION + 3. EVALUATION
        # -------------------------------
        if jeu_sim.est_termine():
            # la partie est finie : on utilise le vrai resultat
            valeur = jeu_sim.resultat()
        else:
            # on appelle le reseau : (P, V) = f_theta(s)
            s               = etat_vers_tenseur(jeu_sim.etat_pour_reseau())
            probas, valeur_tenseur = reseau(s)
            valeur          = valeur_tenseur.item()

            # on initialise les stats pour ce nouvel etat
            c     = cle(jeu_sim)
            coups = jeu_sim.coups_legaux()
            for action in coups:
                P[(c, action)] = probas[action].item()
                N[(c, action)] = 0
                W[(c, action)] = 0.0

        # --------------------------------------------------
        # 4. BACKUP : on remonte V dans tous les noeuds
        # --------------------------------------------------
        for c, action in reversed(chemin):
            N[(c, action)] = N.get((c, action), 0) + 1
            W[(c, action)] = W.get((c, action), 0.0) + valeur
            valeur = -valeur   # on change de signe car les joueurs alternent

    # politique finale proportionnelle aux visites : pi_a = N(s,a) / sum N
    politique = numpy.zeros(7)
    c         = cle(jeu_initial)
    for action in jeu_initial.coups_legaux():
        politique[action] = N.get((c, action), 0)

    if politique.sum() > 0:
        politique = politique / politique.sum()

    return politique


# ----------
# test
# ----------
if __name__ == '__main__':

    jeu = Puissance4()
    jeu.reinitialiser()

    print('lancement de', nb_simulations, 'simulations MCTS...')
    politique = mcts_recherche(jeu)

    print('politique MCTS :')
    for action in range(7):
        print(f'  colonne {action} : {politique[action]:.3f}')

    print('meilleure action :', numpy.argmax(politique))
