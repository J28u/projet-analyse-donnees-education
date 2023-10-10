# OC/DS Projet 2 : Analysez des données de systèmes éducatifs
Formation OpenClassrooms - Parcours data scientist - Projet Professionnalisant (Septembre-Octobre 2022)

## Secteur : 
Éducation 

## Technologies utilisées : 
  * Jupyter Notebook
  * Python (pandas, numpy, matplotlib, seaborn, missingno)

 ## Livrables :
 * notebook.ipynb : notebook jupyter comportant les analyses pré-exploratoires réalisées
 * presentation.pdf : support de présentation pour la soutenance

## Le contexte : 
Le client, une startup fictive nommée «Academy», propose des cours en ligne niveau lycée et plus. Il a pour projet de développer ses activités à l’internationale et aimerait identifier les pays dans lesquels opérer en priorité. 

Notre manager nous a confié plusieurs fichiers contenant des milliers d’indicateurs en lien avec l’éducation sur des centaines de pays, récoltés par un organisme de la banque mondiale. 

## La mission : 
Mener une analyse pré-exploratoire pour déterminer si le jeu de données a une qualité suffisante pour informer le projet d’expansion de l’entreprise. Si oui, proposer une liste de pays correspondants aux critères du client.

## Méthodologie suivie : 
1. Nettoyage des données :
  * comprendre les variables à disposition et sélectionner les plus pertinentes
  * analyser la qualité des données sélectionnées et écarter les pays, indicateurs, années avec trop de données manquantes
  * imputation par interpolation linéaire (car valeurs régulièrement espacées d’un an).

2. Analyse des données :
  * visualiser la répartition des données à l'aide de boxplots
  * établir un score d’attractivité par pays (méthode du rang centile ou percentile)

## Compétences acquises :  
* Mettre en place un environnement Python
* Effectuer une représentation graphique à l’aide d’une librairie Python adaptée
* Maîtriser les opérations fondamentales du langage Python pour la Data Science
* Manipuler des données avec des librairies Python spécialisées
* Utiliser un notebook Jupyter pour faciliter la rédaction du code et la collaboration

## Data source : 
 https://datacatalog.worldbank.org/dataset/education-statistics

