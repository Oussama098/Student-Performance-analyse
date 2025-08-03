import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
# importaion de ficher CSV student-por.csv contient les performances des étudiants avec ses grades en Portugais
df = pd.read_csv('student-por.csv', sep=';')
# print(df.head())
# print(df.info())

# pretraitement des données
# print(df.isnull().sum())

# encodage des variables catégorielles
binary_cols = ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})

df['school'] = df['school'].map({'GP': 0, 'MS': 1})
df['sex'] = df['sex'].map({'F': 0, 'M': 1})
df['address'] = df['address'].map({'U': 0, 'R': 1})
df['famsize'] = df['famsize'].map({'LE3': 0, 'GT3': 1})
df['Pstatus'] = df['Pstatus'].map({'T': 0, 'A': 1})

# encodage des variables nominales
nominal_cols = ['Mjob', 'Fjob', 'reason', 'guardian']
df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

# print(df.head())
# print(df.info())

grades_cols = ['G1', 'G2', 'G3']

features = df.drop(columns=['G1', 'G2', 'G3'])



# standardisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)
X_scaled_df = pd.DataFrame(X_scaled, columns=features.columns)

df['pass_fail'] = (df['G3'] >= 10).astype(int)
y = df['pass_fail']

# print(X_scaled_df.describe())

# implementation de la PCA
pca = PCA()
# calcul les composantes principales
X_pca = pca.fit_transform(X_scaled)

# analyse de la variance expliquée
explained_variance = pca.explained_variance_ratio_
cumsum = np.cumsum(explained_variance)
# plot de la variance expliquée
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--', label='Variance expliquée individuelle')
# plt.plot(range(1, len(explained_variance) + 1), cumsum, marker='o', linestyle='-', label='Variance expliquée cumulée')
# plt.title('Cumulative Explained Variance by PCA Components')
# plt.xlabel('Number of Components')
# plt.ylabel('Cumulative Explained Variance')
# plt.grid(True)
# plt.legend()
# plt.show()

# Reduction de la dimensionnalité
k = 20
pca_final = PCA(n_components=k)
X_reduced = pca_final.fit_transform(X_scaled)
X_reduced_df = pd.DataFrame(X_reduced, columns=[f'PC{i+1}' for i in range(k)])

# affichage des composantes principales
print("Shape of reduced data:", X_reduced_df.shape)
print(X_reduced_df.head())

# Visualisation des 2 premières composantes principales
g3_grades = df['G3']

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_reduced_df['PC1'], X_reduced_df['PC2'], c=g3_grades, cmap='viridis', alpha=0.8)
plt.title('Étudiants projetés sur les 2 premières Composantes Principales (colorés par G3)')
plt.xlabel('Première Composante Principale (PC1)')
plt.ylabel('Deuxième Composante Principale (PC2)')
plt.colorbar(scatter, label='Note Finale (G3)')
plt.grid(True)
plt.show()

# Vous pouvez aussi essayer de colorer par 'school', 'sex', etc.
# plt.figure(figsize=(10, 8))
# scatter = plt.scatter(X_reduced_df['PC1'], X_reduced_df['PC2'], c=df['school'], cmap='coolwarm', alpha=0.8) # 'school' est encodé en 0/1
# plt.title('Étudiants projetés sur les 2 premières Composantes Principales (colorés par École)')
# plt.xlabel('Première Composante Principale (PC1)')
# plt.ylabel('Deuxième Composante Principale (PC2)')
# plt.colorbar(scatter, ticks=[0, 1], label='École (0: GP, 1: MS)')
# plt.grid(True)
# plt.show()





# --- Étape 5 : Validation (Intégration d'un Algorithme Supervisé et Métriques) ---
print("\n--- Étape de Validation : Comparaison des modèles avec et sans PCA ---")

# --- Modèle 1 : Sans PCA (avec toutes les caractéristiques standardisées) ---
print("\n### Performance du modèle de Classification (RandomForest) SANS PCA ###")
model_no_pca = RandomForestClassifier(random_state=42)

# Validation croisée sur les données complètes standardisées
# Utilisation de X_scaled (vos caractéristiques originales standardisées)
scores_no_pca = cross_val_score(model_no_pca, X_scaled, y, cv=5, scoring='accuracy')
print(f"Accuracy moyenne (Cross-Validation 5-fold sans PCA) : {scores_no_pca.mean():.4f} (+/- {scores_no_pca.std()*2:.4f})")

# Pour avoir le rapport de classification et la matrice de confusion, entraînement sur un split train/test
X_train_no_pca, X_test_no_pca, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
model_no_pca.fit(X_train_no_pca, y_train)
y_pred_no_pca = model_no_pca.predict(X_test_no_pca)

print("\nRapport de Classification (SANS PCA) :")
print(classification_report(y_test, y_pred_no_pca, target_names=['Échec (0)', 'Réussite (1)']))

print("\nMatrice de Confusion (SANS PCA) :")
cm_no_pca = confusion_matrix(y_test, y_pred_no_pca)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_no_pca, annot=True, fmt='d', cmap='Blues', xticklabels=['Échec', 'Réussite'], yticklabels=['Échec', 'Réussite'])
plt.xlabel('Prédiction')
plt.ylabel('Vraie Valeur')
plt.title('Matrice de Confusion (SANS PCA)')
plt.show()
print("-" * 50)

# --- Modèle 2 : Avec PCA (avec les K=20 composantes principales) ---
print("\n### Performance du modèle de Classification (RandomForest) AVEC PCA (k=20) ###")
# Utilisation de X_reduced (vos données réduites par PCA)
model_with_pca = RandomForestClassifier(random_state=42)

# Validation croisée sur les données réduites par PCA
scores_with_pca = cross_val_score(model_with_pca, X_reduced, y, cv=5, scoring='accuracy')
print(f"Accuracy moyenne (Cross-Validation 5-fold AVEC PCA) : {scores_with_pca.mean():.4f} (+/- {scores_with_pca.std()*2:.4f})")

# Pour avoir le rapport de classification et la matrice de confusion
X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_reduced, y, test_size=0.3, random_state=42, stratify=y)
model_with_pca.fit(X_train_pca, y_train)
y_pred_with_pca = model_with_pca.predict(X_test_pca)

print("\nRapport de Classification (AVEC PCA) :")
print(classification_report(y_test, y_pred_with_pca, target_names=['Échec (0)', 'Réussite (1)']))

print("\nMatrice de Confusion (AVEC PCA) :")
cm_with_pca = confusion_matrix(y_test, y_pred_with_pca)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_with_pca, annot=True, fmt='d', cmap='Blues', xticklabels=['Échec', 'Réussite'], yticklabels=['Échec', 'Réussite'])
plt.xlabel('Prédiction')
plt.ylabel('Vraie Valeur')
plt.title('Matrice de Confusion (AVEC PCA)')
plt.show()
print("-" * 50)
