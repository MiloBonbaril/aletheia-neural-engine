# Rapport sur le Modèle et l'Agent (Model.md)

Ce document détaille la structure et le fonctionnement interne de l'agent intelligent du projet **Aletheia Neural Engine**. Il explique l'architecture des réseaux de neurones (Acteur et Critique), leur rôle respectif dans l'apprentissage par renforcement, et fournit une analyse sur la flexibilité du système permettant de faire évoluer ou de remplacer ces composants.

---

## 1. L'Agent : `PPOAgent`

L'agent implémenté est un agent de type **Acteur-Critique** régi par l'algorithme de gradient de politique **PPO** (Proximal Policy Optimization). L'agent a la double responsabilité d'interagir avec l'environnement pour collecter des expériences et de mettre à jour ses réseaux de neurones pour maximiser les récompenses à long terme.

Il gère trois entités principales :
1. **L'Acteur (`ActorNetwork`)** : Le réseau responsable de la prise de décision (la politique $\pi_\theta$).
2. **Le Critique (`CriticNetwork`)** : Le réseau responsable de l'évaluation de la qualité des états rencontrés (la fonction de valeur $V_\phi$).
3. **L'Optimiseur conjoint** : Un unique optimiseur `Adam` qui regroupe les paramètres des deux réseaux et applique les gradients avec un taux d'apprentissage $\alpha$ décroissant et stabilisé par un epsilon numérique $\epsilon_{Adam} = 10^{-5}$ :
```python
self.optimizer = optim.Adam(
    list(self.actor.parameters()) + list(self.critic.parameters()),
    lr=config.lr,
    eps=1e-5
)
```

---

## 2. Architecture des Réseaux de Neurones

Les réseaux sont codés dans `networks.py` sous forme de modules PyTorch héritant de `nn.Module`. Ils sont conçus pour traiter des espaces d'observations et d'actions continus.

```mermaid
graph TD
    subgraph Réseau Acteur (Politique)
        O[Observation de l'état : obs_dim] --> H1[Linear 256 + Tanh]
        H1 --> H2[Linear 256 + Tanh]
        H2 --> ML[Mean Layer: Linear action_dim]
        ML --> T[Tanh Activation]
        T --> AM[Moyenne de l'action: action_mean]
        STD[Paramètre Log_std apprenable] --> ASTD[Ecart-type de l'action: action_std]
        AM & ASTD --> D[Loi Normale: Normal]
        D --> |Echantillonnage stochastique| AS[Action en entraînement]
        AM --> |Mode déterministe| AD[Action en évaluation]
    end

    subgraph Réseau Critique (Valeur)
        OC[Observation de l'état : obs_dim] --> HC1[Linear 256 + Tanh]
        HC1 --> HC2[Linear 256 + Tanh]
        HC2 --> VL[Value Layer: Linear 1]
        VL --> V[Estimation de l'état-valeur: V(s)]
    end
```

### A. L'Acteur (`ActorNetwork`)
L'Acteur mappe un état d'observation à une distribution de probabilité sur les actions possibles. Puisque nous sommes dans un espace d'actions continues (les couples moteurs à appliquer aux articulations), la politique est modélisée par une **distribution normale multi-variée diagonale**.

* **Entrée** : Le vecteur d'observation continu de taille `obs_dim` (ex: 24 pour le bipède).
* **Couches Cachées (MLP)** : Deux couches linéaires entièrement connectées de 256 neurones par défaut (`actor_hidden_dims`), chacune suivie de la fonction d'activation non-linéaire **`Tanh`**.
* **Tête de Moyenne (`mean_layer`)** : Une couche linéaire projetant l'activation vers la dimension des actions (`action_dim`, ex: 4). La sortie est passée dans une fonction **`Tanh`** pour borner strictement la moyenne des actions générées dans l'intervalle $[-1, 1]$, ce qui correspond parfaitement aux contraintes physiques des articulations.
* **Écart-type de l'action (`log_std`)** : Au lieu d'être prédit à partir de l'état (ce qui rendrait l'apprentissage instable), l'écart-type de l'exploration est modélisé comme un **vecteur de paramètres apprenables indépendant de l'état**, initialisé par `init_log_std` = -0.5 ($\sigma \approx 0.6$). Pendant l'entraînement, l'agent ajuste cet écart-type global : s'il commence à converger vers une bonne politique, l'écart-type diminue pour affiner les mouvements ; s'il stagne, il se maintient pour continuer à explorer.
* **Sortie** : L'objet PyTorch `Normal(action_mean, action_std)`. 
  - *En mode entraînement* : L'action est échantillonnée à partir de cette loi normale (`dist.sample()`), ce qui génère un bruit d'exploration nécessaire pour découvrir de nouveaux comportements.
  - *En mode évaluation déterministe* : On ignore le bruit et on retourne directement la moyenne `action_mean`.

### B. Le Critique (`CriticNetwork`)
Le Critique estime la valeur attendue future cumulative d'un état donné, notée $V(s)$. Il sert de base de comparaison pour évaluer si l'action choisie par l'acteur a produit un résultat meilleur ou pire que prévu (l'Avantage).

* **Entrée** : Le vecteur d'observation continu de taille `obs_dim`.
* **Couches Cachées (MLP)** : Deux couches linéaires entièrement connectées de 256 neurones avec activation `Tanh`, de manière identique à l'acteur.
* **Couche de Sortie** : Une unique couche linéaire (`Linear(in_dim, 1)`) sans fonction d'activation. La sortie est un scalaire réel représentant l'estimation de la valeur temporelle de l'état courant.

### C. Initialisation Orthogonale
Toutes les couches linéaires des deux réseaux passent par une fonction d'initialisation orthogonale (`layer_init`) avec des facteurs d'échelle (`std`) spécifiques :
* Couches cachées de l'Acteur et du Critique : `std = sqrt(2)` (optimal pour maintenir la variance des activations à travers les fonctions non-linéaires).
* Couche finale de l'Acteur (`mean_layer`) : `std = 0.01`. Cette valeur extrêmement faible force le réseau à démarrer l'entraînement en produisant des actions très proches de 0 (le robot reste calme et n'effectue pas de mouvements brusques destructeurs dès les premières étapes).
* Couche finale du Critique : `std = 1.0`.

---

## 3. Remplacer ou Faire Évoluer le Modèle

Le framework a été conçu de manière à ce qu'il soit très simple de faire évoluer ou de remplacer l'architecture du réseau ou l'agent lui-même.

### Cas A : Modifier la structure du réseau (Simple)
Si vous souhaitez changer la taille des couches, ajouter des types de couches (comme du Dropout, du Batch Normalization, ou changer de fonction d'activation pour de la LeakyReLU), il suffit de modifier `networks.py`.

* **Changer le nombre/taille des couches** : Vous pouvez le faire directement dans le fichier `config.py` en modifiant les tuples `actor_hidden_dims` et `critic_hidden_dims` (ex: les passer à `(512, 512, 256)`). Les réseaux s'auto-dimensionneront à l'instanciation.
* **Ajouter de la mémoire temporelle (LSTM / GRU)** : Si l'environnement devient partiellement observable (par exemple, si on supprime les capteurs LiDAR ou si on ajoute du brouillard de guerre), l'agent aura besoin de mémoire. Pour cela, il suffit de remplacer le MLP de `networks.py` par une couche récurrente (ex. `nn.LSTM`). L'agent devra alors conserver un état caché (`hidden_state` et `cell_state`) d'une étape à l'autre lors de la collecte du rollout.

### Cas B : Remplacer l'algorithme d'apprentissage (Modéré à Complexe)
Si vous souhaitez abandonner PPO au profit d'un autre algorithme de reinforcement learning continu majeur, comme **SAC** (Soft Actor-Critic) ou **DDPG** (Deep Deterministic Policy Gradient) :

1. **Pourquoi faire cela ?** PPO est un algorithme **on-policy**. Bien que très stable et robuste, il est très inefficace en échantillons (il jette toutes les données collectées après chaque mise à jour). SAC ou DDPG sont **off-policy** : ils stockent les transitions dans un tampon géant et les ré-échantillonnent en boucle, ce qui demande 10 à 100 fois moins d'interactions avec l'environnement pour converger.
2. **Ce qui doit changer dans le code :**
   - **Le Buffer (`buffer.py`)** : Il faut remplacer le `RolloutBuffer` (qui est vidé à chaque mise à jour de politique) par un **`ReplayBuffer`** (tampon de rejeu) de grande taille (ex: $10^6$ transitions) qui conserve l'historique à long terme et permet un échantillonnage purement aléatoire de mini-batchs hors-politique.
   - **L'Agent (`agent.py`)** : Il faut écrire un `SACAgent` à la place de `PPOAgent`. SAC nécessite deux critiques (Double Q-Learning) pour éviter la surestimation des valeurs et un mécanisme de cible mobile (Target Networks) avec mise à jour douce (Polyak averaging, coefficient $\tau \approx 0.005$). De plus, l'acteur dans SAC utilise une gaussienne conditionnelle où la moyenne *et* l'écart-type dépendent de l'état actuel (contrairement à l'écart-type statique de PPO).
   - **La Boucle d'Entraînement (`train.py`)** : Les algorithmes off-policy ne font pas de longs rollouts avant d'entraîner. Typiquement, dans SAC, l'agent prend **une** étape dans l'environnement, l'enregistre dans le `ReplayBuffer`, puis effectue immédiatement **une** étape de gradient (mise à jour des poids) sur un mini-batch aléatoire extrait du buffer.
