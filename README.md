Analyse de la Performance des Étudiants avec PCA

Ce projet utilise l'Analyse en Composantes Principales (PCA) pour explorer et réduire la dimensionnalité d'un jeu de données sur la performance des étudiants en langue portugaise. L'objectif est de comprendre les facteurs sous-jacents qui influencent les caractéristiques des étudiants et d'évaluer l'impact de la réduction de dimensionnalité sur la prédiction de leur réussite scolaire.
Table des Matières

    Contexte du Projet

    Jeu de Données

    Méthodologie

        Prétraitement des Données

        Application de la PCA

        Visualisation des Composantes Principales

        Validation du Modèle

    Résultats Clés

    Comment Exécuter le Projet

    Dépendances

Contexte du Projet

L'analyse des facteurs influençant la performance scolaire est un domaine clé dans l'éducation. Le jeu de données utilisé ici contient de nombreuses caractéristiques socio-économiques, comportementales et éducatives des étudiants. La grande dimensionnalité et les corrélations potentielles entre ces variables peuvent rendre l'analyse complexe. La PCA est employée pour simplifier cette complexité, identifier les dimensions latentes les plus significatives et évaluer si cette simplification peut bénéficier à des tâches d'apprentissage automatique, comme la prédiction de la réussite des étudiants.
Jeu de Données

Le projet utilise le jeu de données student-por.csv (Student Performance Data Set) disponible sur l'UCI Machine Learning Repository.

Description des attributs clés :

    Caractéristiques Socio-démographiques : school, sex, age, address, famsize, Pstatus, Medu (éducation de la mère), Fedu (éducation du père), Mjob (métier de la mère), Fjob (métier du père), guardian.

    Caractéristiques Comportementales/Relationnelles : traveltime, studytime, failures, schoolsup, famsup, paid, activities, nursery, higher, internet, romantic, famrel, freetime, goout, Dalc (consommation d'alcool en semaine), Walc (consommation d'alcool le week-end), health, absences.

    Notes (Cibles Potentielles) : G1 (note du premier semestre), G2 (note du deuxième semestre), G3 (note finale). Pour la validation, G3 est binarisée en "Réussite" (>=10) ou "Échec" (<10).

Méthodologie

Le projet suit un pipeline standard de science des données :
Prétraitement des Données

    Chargement : Le fichier student-por.csv est chargé dans un DataFrame Pandas.

    Encodage des Variables Catégorielles :

        Les variables binaires (yes/no, F/M, U/R, etc.) sont mappées en 0 et 1.

        Les variables nominales avec plus de deux catégories (Mjob, Fjob, reason, guardian) sont encodées en utilisant l'One-Hot Encoding (pd.get_dummies).

    Définition des Caractéristiques et de la Cible : Les colonnes G1, G2, G3 sont exclues des caractéristiques d'entrée pour la PCA afin de se concentrer sur les facteurs socio-démographiques et comportementaux. La note G3 est utilisée pour créer une variable cible binaire pass_fail (1 si G3 >= 10, 0 sinon).

    Standardisation : Toutes les caractéristiques numériques (y compris les variables encodées) sont standardisées à l'aide de StandardScaler pour avoir une moyenne de 0 et un écart-type de 1. Cette étape est cruciale pour la PCA.

Application de la PCA

    Analyse de la Variance Expliquée : Une PCA est initialement appliquée sur toutes les dimensions pour analyser la proportion de variance expliquée par chaque composante principale et la variance cumulée. Un "scree plot" (non affiché par défaut dans le code fourni, mais peut être décommenté) est utilisé pour déterminer le nombre optimal de composantes (k) à conserver.

    Réduction de Dimensionnalité : La PCA est ensuite réappliquée avec le nombre choisi de composantes (k=20 dans ce projet) pour projeter les données standardisées dans un espace de dimension inférieure.

    Interprétation des Composantes Principales (Loadings) : Les poids (loadings) de chaque caractéristique originale sur les premières composantes principales sont analysés pour comprendre ce que chaque composante représente conceptuellement (par exemple, "facteur de soutien éducatif", "facteur de comportement social").

Visualisation des Composantes Principales

Les données sont projetées sur les deux premières composantes principales (k=2 pour la visualisation) pour créer des nuages de points. Ces nuages de points sont colorés par la variable cible (pass_fail), l'école (school) et le sexe (sex) pour observer d'éventuels regroupements ou séparations visuelles.
Validation du Modèle

Pour évaluer l'efficacité de la PCA comme étape de prétraitement, un modèle de classification RandomForestClassifier est entraîné et évalué dans deux scénarios :

    Sans PCA : Le modèle est entraîné sur les caractéristiques originales standardisées (X_scaled).

    Avec PCA : Le modèle est entraîné sur les données réduites par PCA (X_reduced, avec k=20).

La performance des deux modèles est comparée à l'aide de :

    Validation croisée (5-fold) : Pour une estimation robuste de l'accuracy moyenne.

    Rapport de classification : Incluant la précision, le rappel et le F1-score pour chaque classe.

    Matrice de confusion : Pour visualiser les vrais positifs/négatifs et les faux positifs/négatifs.

Résultats Clés

Les résultats de l'exécution du script incluront :

    Les dimensions des données après encodage One-Hot et après réduction par PCA.

    Les loadings des caractéristiques sur les composantes principales, aidant à leur interprétation.

    Les visualisations 2D des étudiants dans l'espace des deux premières composantes principales, colorées par leur statut de réussite, leur école et leur sexe.

    Une comparaison détaillée des métriques de performance (accuracy, précision, rappel, F1-score) et des matrices de confusion pour les modèles entraînés avec et sans PCA.

Cette comparaison permettra de déterminer si la PCA a permis de maintenir, voire d'améliorer, la performance du modèle tout en réduisant considérablement la complexité des données d'entrée.
Comment Exécuter le Projet

    Téléchargez le jeu de données : Assurez-vous d'avoir le fichier student-por.csv dans le même répertoire que le script Python.

    Clonez ce dépôt (ou copiez le code) :

    git clone https://github.com/Oussama098/Student-Performance-analyse.git

    Installez les dépendances :

    pip install pandas scikit-learn matplotlib numpy seaborn

    Exécutez le script Python :

    python StudentPerformanceAnalyse.py

    

Dépendances

Ce projet nécessite les bibliothèques Python suivantes :

    pandas

    scikit-learn

    matplotlib

    numpy

    seaborn
