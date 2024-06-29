# Projet RAG - README

## Introduction

Bienvenue dans le projet de Génération-Augmentée par Récupération (RAG) ! Ce projet a pour objectif d'améliorer les capacités des modèles de langage en les intégrant avec un mécanisme de récupération pour accéder à des documents externes pertinents. En combinant des modèles de langage de pointe comme GPT-3.5 et Mistral avec des systèmes de récupération de documents efficaces, nous visons à produire des réponses plus précises, riches en contexte et pertinentes.

## Objectifs du Projet

- **Amélioration des Réponses** : Utilisation de documents externes pour fournir des réponses plus contextuelles.
- **Comparaison des Modèles** : Évaluer les performances de divers modèles d'embedding et stratégies de récupération.
- **Applications Variées** : Tester l'efficacité de RAG dans différents cas d'utilisation.

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
