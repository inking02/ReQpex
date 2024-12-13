# QLink : Calcul Analogue Quantique

Bienvenue dans le dépôt GitHub de **QLink** ! Nous sommes une équipe passionnée par l'informatique quantique et explorons différents algorithmes quantiques analogues à travers ce projet. L'objectif est d'aider l'organisation Récupex à améliorer leur gestion de bacs de récuperation en utilisant différents approches quantiques. Pour ce projet, des concepts tel que **la théorie des graphes**, l'utilisation des **ordinateurs quantiques à atomes neutres** notamment les appareils développés par **Pasqal**, **l'ensemble indépendant maximal**, l'algorithme quantique adiabatique **( QAA )** et l'algorithme d'optimisation approximative quantique **( QAOA )** ont été utilisés. 

## Structure du Projet
- **`Recupex_problem.pdf`** : Le rapport contenant notre la définition du problème de Récupex et nos solutions ainsi que les résultats que nous avons obtenus.
- **`QMIS_code`** : Dossier contenant les deux méthodes quantiques utilisées pour trouver l'ensemble indépendant maximal ( QAA et QAOA ).
- **`datasets`** : Dossier contenant les ensembles de données utilisés pour la création du graphe.
- **`figures`** : Dossier qui, après avoir roulé les algotihmes contiendra les figures et cartes résultantes.
- **`utils`** : Dossier contenant les fonctions utilitaires au projet.
- **`requirements.txt`** : Fichier texte avec les commandes d'installation des différents modules requis pour rouler le code.
- **`main.py`** : Fichier permettant de résoudre le problème de Récupex. Plus de détails sont disponibles dans le rapport.
- **`tutorial.ipynb`** : Tutoriel d'utilisation des classes du projet et des fonctions permettant de résoudre le problème de Récupex.

## Utilisation
Pour bien comprendre l'implementation des différentes méthodes proposées dans ce projet, vous pouvez commencer par consulter le jupyter_notebook **'tutorial.ipynb'** dans le dossier ... Vous pouvez ensuite importer les librairies utiles qui existent dans **'requirement.txt'**, puis consulter **'main.py'**.

Installez les librairies nécessaires avec la ligne de code suivante dans l'environnement de votre choix:

```
pip install -r requirements.txt
```

Selon l'environnement, *pip* devra peut-être remplacé par *pip3*. L'installation de celles-ci sont nécessaire pur facilement rouler le projet.
**Attention**
La version de python utilisée doit être de 3.11 pour utiliser la libraire Pulser.
