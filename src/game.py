# Puissance 4 - logique du jeu
# Projet AlphaZero

import numpy

LIGNES   = 6
COLONNES = 7
ALIGNEMENT = 4   # nombre de pions a aliger pour gagner


class Puissance4:

    def __init__(self):
        # grille : 0 = vide, 1 = joueur courant, -1 = adversaire
        self.grille = numpy.zeros((LIGNES, COLONNES), dtype=int)
        self.joueur_courant = 1   # 1 ou -1, on alterne

    def reinitialiser(self):
        # remet la grille a zero et le joueur courant a 1
        # equivalent a env.reset() dans les TPs du cours
        self.grille = numpy.zeros((LIGNES, COLONNES), dtype=int)
        self.joueur_courant = 1
        return self.grille.copy()

    def coups_legaux(self):
        # une colonne est jouable si la case tout en hau est encore vide
        return [c for c in range(COLONNES) if self.grille[0][c] == 0]

    def jouer(self, colonne):
        # on pose le pion dans la case la plus basse de la colonne choisie
        for ligne in range(LIGNES - 1, -1, -1):
            if self.grille[ligne][colonne] == 0:
                self.grille[ligne][colonne] = self.joueur_courant
                break

        victoire = self.verifier_victoire(self.joueur_courant)

        # on passe au joueur suivant
        self.joueur_courant = -self.joueur_courant

        return victoire

    def verifier_victoire(self, joueur):
        # horizontal
        for ligne in range(LIGNES):
            for col in range(COLONNES - 3):
                if all(self.grille[ligne][col + i] == joueur for i in range(ALIGNEMENT)):
                    return True

        # vertical
        for ligne in range(LIGNES - 3):
            for col in range(COLONNES):
                if all(self.grille[ligne + i][col] == joueur for i in range(ALIGNEMENT)):
                    return True

        # diagonale montant /
        for ligne in range(3, LIGNES):
            for col in range(COLONNES - 3):
                if all(self.grille[ligne - i][col + i] == joueur for i in range(ALIGNEMENT)):
                    return True

        # diagonale descendante \
        for ligne in range(LIGNES - 3):
            for col in range(COLONNES - 3):
                if all(self.grille[ligne + i][col + i] == joueur for i in range(ALIGNEMENT)):
                    return True

        return False

    def est_termine(self):
        # la partie est finie si quelqu'un a gagne ou si la grille est pleine
        if self.verifier_victoire(1) or self.verifier_victoire(-1):
            return True
        if len(self.coups_legaux()) == 0:
            return True
        return False

    def resultat(self):
        # retourne +1 si joueur 1 a gagne, -1 si joueur 2, 0 si match nul
        if self.verifier_victoire(1):
            return 1
        if self.verifier_victoire(-1):
            return -1
        return 0

    def etat_pour_reseau(self):
        # on retourne la grille vue du joueur courant
        # le reseau voit toujours ses pion en +1 et ceux de l'adversaire en -1
        return self.grille * self.joueur_courant

    def afficher(self):
        symboles = {0: '.', 1: 'X', -1: 'O'}
        print()
        for ligne in self.grille:
            print(' '.join(symboles[c] for c in ligne))
        print(' '.join(str(i) for i in range(COLONNES)))
        print()


# ---------------------------------------------------------
# test rapide
# -------------------------------
if __name__ == '__main__':

    jeu = Puissance4()
    jeu.reinitialiser()

    import random

    nbEpisodes = 3

    for i in range(nbEpisodes):
        print(f'=== partie {i} ===')
        jeu.reinitialiser()
        endOfEpisode = False

        while not endOfEpisode:
            coups = jeu.coups_legaux()
            action = random.choice(coups)
            victoire = jeu.jouer(action)
            endOfEpisode = jeu.est_termine()

        jeu.afficher()
        print('resultat :', jeu.resultat())