# CampusAdvisor 🏫💬

CampusAdvisor est un bot conversationnel conçu pour aider les nouveaux bacheliers à s'orienter dans le choix de leurs études supérieures. Le bot s'appuie sur une technique de *Retrieval-Augmented Generation (RAG)*, exploitant une base de données locale des universités pour fournir des réponses personnalisées. Grâce à son interface développée avec **Streamlit**, les utilisateurs peuvent interagir facilement avec le bot et explorer les options académiques.


> **Tester l'application en ligne** : [CampusAdvisor](https://campusadvisor.streamlit.app/)

---

## 🎯 Objectif du projet

Le projet vise à offrir une assistance interactive et contextuelle pour guider les nouveaux bacheliers dans leurs choix académiques. CampusAdvisor se base sur des données locales d'universités, permettant d'apporter des recommandations pertinentes adaptées aux besoins et profils des utilisateurs.

## 🚀 Fonctionnalités principales

- **Orientation personnalisée** : En fonction des préférences et des résultats scolaires des utilisateurs.
- **Recommandations d'universités et de filières** : Possibilité de découvrir les universités et programmes disponibles.
- **Actualisation des données** : Ajout de nouvelles données via un dossier `data`, suivi d'un *rebuild* du modèle pour inclure ces nouvelles informations dans le contexte des réponses.
- **Interface intuitive** : Utilisation de **Streamlit** pour offrir une expérience utilisateur fluide.

---

## 📂 Structure du projet

- `bot.py` : Le script principal qui lance l'application Streamlit.
- `README.md` : Ce document de présentation du projet.
- `assets/` : Contient les captures d'écran et autres ressources liées à l'interface.
- `data/` : Ce dossier contient les données sur les universités et filières. Il est possible d'ajouter de nouvelles données ici pour enrichir le bot.

---

## 📦 Installation

### Prérequis

- Python 3.8+
- Streamlit

### Étapes d'installation

## 📦 Installation

### Prérequis

- Python 3.8+
- Streamlit

### Étapes d'installation

1. **Clonez le dépôt** :
    ```bash
    git clone https://github.com/TitanSage02/CampusAdvisor.git
    cd CampusAdvisor
    ```

2. **Installez les dépendances** :
    ```bash
    pip install -r requirements.txt
    ```

3. **Générez un token d'accès** pour **Google Gemini** :
   
   Rendez-vous sur [Gemini API - Clé API](https://ai.google.dev/gemini-api/docs/api-key) pour créer votre clé API.

4. **Créez un fichier `.env`** dans le répertoire racine du projet et ajoutez-y votre clé API comme suit :
   
   Créez un fichier nommé `.env` :
   ```bash
   GOOGLE_API_KEY = votre_cle_api_ici
   ```
---

## 🗄️ Ajout de nouvelles données

CampusAdvisor permet d'ajouter de nouvelles universités, programmes ou informations supplémentaires dans le dossier `data`. Suivez les étapes ci-dessous pour enrichir la base de données et *rebuild* le modèle.

### Étapes pour ajouter des données

1. **Ajoutez vos fichiers CSV ou JSON** dans le dossier `data/`.

2. **Relancez l'application** :
    ```bash
    streamlit run app.py
    ```

Le bot inclura désormais les nouvelles données dans ses réponses.

---

## 📊 Démo de l'interface

[assets/demo.mp4]

---

## 🤖 Utilisation

Une fois l'application lancée, vous pouvez poser des questions au bot telles que :

- *Quelles formations pour devenir un ingénieur au Bénin ?*
- *Quelles sont les meilleures options pour étudier la médecine au Bénin ?*
- *Quels débouchés pour un diplôme en ingénierie informatique ?*

Le bot utilise les données disponibles dans le dossier `data` pour fournir des recommandations pertinentes.

---

## 👥 Contributions

Les contributions sont les bienvenues ! Voici comment participer :

1. **Forkez le dépôt**.
2. **Créez une nouvelle branche** (`git checkout -b feature-nouvelle-fonctionnalité`).
3. **Commitez vos modifications** (`git commit -m 'Ajout d'une nouvelle fonctionnalité'`).
4. **Poussez la branche** (`git push origin feature-nouvelle-fonctionnalité`).
5. **Ouvrez une Pull Request** et décrivez vos modifications.

---

## 📄 Licence

Ce projet est sous licence. Consultez le fichier [LICENSE.md](LICENSE.md) pour plus de détails.

---

## 🙌 Remerciements

Merci à tous les contributeurs et testeurs qui ont participé au développement de CampusAdvisor.
