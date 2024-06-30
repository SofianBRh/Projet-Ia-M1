# Projet RAG - README

## Introduction

Bienvenue dans le projet de Génération-Augmentée par Récupération (RAG) ! Ce projet a pour objectif d'améliorer les capacités des modèles de langage en les intégrant avec un mécanisme de récupération pour accéder à des documents externes pertinents. En combinant des modèles de langage de pointe comme GPT-3.5 et Mistral avec des systèmes de récupération de documents efficaces, nous visons à produire des réponses plus précises, riches en contexte et pertinentes.

## Schéma LLM:

![llm](https://github.com/SofianBRh/Projet-Ia-M1/assets/95184450/142c2b0b-31fc-44f2-aed5-9a37345dcd05)





## Objectifs du Projet

- **Amélioration des Réponses** : Utilisation de documents externes pour fournir des réponses plus contextuelles.
- **Comparaison des Modèles** : Évaluer les performances de divers modèles d'embedding et stratégies de récupération.
- **Applications Variées** : Tester l'efficacité de RAG dans différents cas d'utilisation.


## Fonctionnement de la Génération-Augmentée par Récupération (RAG)

La Génération-Augmentée par Récupération (RAG) combine la puissance de la génération de texte par l'IA avec la capacité de récupérer des informations pertinentes à partir d'une base de documents. Voici comment cela fonctionne :

1. **Requête Utilisateur** : Un utilisateur pose une question ou donne une instruction.
2. **Embedding de la Requête** : La requête est transformée en un vecteur numérique (embedding) qui capture le sens de la phrase.
3. **Récupération de Documents** : Ce vecteur est utilisé pour rechercher les documents les plus pertinents dans une base de données, grâce à des techniques de similarité sémantique.
4. **Combinaison de Contexte** : Les textes des documents récupérés sont combinés avec la requête initiale pour créer un contexte enrichi.
5. **Génération de Réponse** : Ce contexte est fourni à un modèle de génération de texte (comme Mistral ou GPT-3.5) pour produire une réponse plus complète et informée.


## Utilisation sur Google Colab

Pour faciliter l'utilisation de ce projet, nous avons préparé des notebooks Google Colab où vous pouvez exécuter les scripts sans avoir à cloner le dépôt localement. Suivez simplement les instructions dans les notebooks pour configurer et exécuter les différents modèles.

### Accéder aux Notebooks Google Colab

1. **GPT-3.5 RAG avec Embedding Intégré :**
   
2. **Mistral RAG avec Sentence Transformers :**
   - [Accéder au notebook ](https://colab.research.google.com/drive/1_12pB2iXumTAjBUqRdJyOf91_1492MLx#scrollTo=6jEjGNNZ8RIo)

## Description des Scripts

### Script 1 : GPT-3.5 RAG avec Embedding Intégré

Ce script utilise GPT-3.5 d'OpenAI pour générer des embeddings et des réponses. Les documents sont chargés, encodés et stockés dans ChromaDB pour la récupération.

#### Exemple de Code

```python
import os
import openai
import chromadb
from chromadb.config import Settings

# Configuration de la clé API OpenAI
api_key = "votre_clé_api_openai"
os.environ['OPENAI_API_KEY'] = api_key
openai.api_key = api_key

def get_embedding(text):
    response = openai.Embedding.create(input=[text], model="text-embedding-ada-002")
    return response['data'][0]['embedding']

client = chromadb.Client(Settings())

def rag_provider(prompt, context, collection, max_tokens=8192):
    query_embedding = get_embedding(prompt)
    results = collection.query(query_embeddings=[query_embedding], n_results=5)
    messages = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}]
    for result in results['documents'][0]:
        messages.append({"role": "assistant", "content": result})

    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    output = response['choices'][0]['message']['content']

    prompt_tokens = len(prompt.split())
    completion_tokens = len(output.split())
    total_tokens = prompt_tokens + completion_tokens

    cost = (total_tokens / 1000) * 0.002

    return {
        "output": output,
        "tokenUsage": {"total": total_tokens, "prompt": prompt_tokens, "completion": completion_tokens, "cost": cost},
        "cached": False,
    }

collection_name = "essays_sentence"
prompt = "Talk about Paul Graham's essay 34 and make a resume."
context = {}
collection = client.get_collection(name=collection_name)
response = rag_provider(prompt, context, collection)
print(f"Résultats pour la collection {collection_name} :")
print(response)
```






### Script : Mistral RAG avec Sentence Transformers

Ce script utilise Sentence Transformers pour générer des embeddings et Mistral pour générer des réponses. Les documents sont gérés et interrogés dans ChromaDB.

#### Exemple de Code

```python
import os
import requests
import json
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Endpoint pour les complétions textuelles
MISTRAL_COMPLETION_URL = "YOUR MISTRAL URL"

# Initialisation du modèle Sentence Transformer
model = SentenceTransformer('all-MiniLM-L6-v2')

def mistral_request(prompt, api_url):
    headers = {
        "Content-Type": "application/json",
    }
    data = {
        "prompt": prompt,
        "max_tokens": 150,
        "temperature": 0.1,
        "top_p": 1.0
    }
    response = requests.post(api_url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Request failed with status code {response.status_code}: {response.text}")

def get_embedding(text):
    return model.encode(text).tolist()

def load_documents_from_directory(directory_path, max_tokens_per_doc=1000):
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".md"):
            with open(os.path.join(directory_path, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                tokens = len(text.split())
                if tokens > max_tokens_per_doc:
                    text = ' '.join(text.split()[:max_tokens_per_doc])
                documents.append({"text": text, "metadata": {"source": filename}})
    return documents

# Configuration de ChromaDB
client = chromadb.Client(Settings())

# Création de la collection de phrases
collection_name = "essays_sentence"
if collection_name not in [col.name for col in client.list_collections()]:
    client.create_collection(name=collection_name)
    documents = load_documents_from_directory("./essays")
    for doc in documents:
        embedding = get_embedding(doc['text'])
        client.get_collection(name=collection_name).add(documents=[doc['text']], embeddings=[embedding], metadatas=[doc['metadata']], ids=[doc['metadata']['source']])

collection = client.get_collection(name=collection_name)

def rag_provider(prompt, context, collection, max_tokens=8192):
    query_embedding = get_embedding(prompt)
    results = collection.query(query_embeddings=[query_embedding], n_results=5)
    retrieved_texts = " ".join([doc[0] for doc in results['documents']])
    combined_prompt = f"{prompt}\n\nContext from documents:\n{retrieved_texts}"

    headers = {
        'Content-Type': 'application/json',
    }
    data = {
        "prompt": combined_prompt,
        "max_tokens": max_tokens
    }
    response = requests.post(MISTRAL_COMPLETION_URL, headers=headers, data=json.dumps(data))
    output = response.json()['choices'][0]['text']

    prompt_tokens = len(prompt.split())
    completion_tokens = len(output.split())
    total_tokens = prompt_tokens + completion_tokens

    return {
        "output": output,
        "tokenUsage": {
            "total": total_tokens,
            "prompt": prompt_tokens,
            "completion": completion_tokens,
        },
        "cached": False,
    }

# Exemple d'utilisation
prompt = "Talk about Paul Graham's essay 34 and make a resume."
context = {}
response = rag_provider(prompt, context, collection)
print(f"Résultats pour la collection {collection_name} :")
print(response)
```

## Différences entre les Modèles d'Embedding

## GPT-3.5 (text-embedding-ada-002) :
Génère des embeddings de haute qualité directement via l'API OpenAI.
Utilisation intégrée pour une compatibilité fluide avec le modèle de langage GPT-3.5.

## Sentence Transformers (all-MiniLM-L6-v2) :
Fournit des embeddings efficaces et performants.
Approprié pour une utilisation avec Mistral et d'autres tâches de récupération.
