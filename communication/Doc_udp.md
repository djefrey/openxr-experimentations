# Documentation de la librairie UDP

## Fonctionnement de la librairie

### 1. Envoi de données (`send`)
- Les données à envoyer sont découpées en blocs, chaque bloc incluant un **header** de 12 octets :
  - 4 octets : ID unique du message.
  - 4 octets : Taille totale du message.
  - 4 octets : ID du bloc.
- Les blocs sont envoyés séquentiellement au destinataire via UDP.
- Un compteur d'ID global garantit qu'un **ID de message unique** est attribué pour chaque envoi.

### 2. Réception de données (`receive`)
- Les paquets reçus sont identifiés par leur **ID de message** et regroupés.
- Lors de la réception du premier paquet d’un nouveau message :
  - La mémoire nécessaire pour le message complet est allouée en fonction de la taille totale spécifiée dans le header.
- Les données du paquet sont copiées à l’**offset** approprié dans le buffer global.
- Un `HashSet` suit les blocs reçus pour éviter les doublons.
- Une fois tous les blocs reçus, les données complètes sont reconstruites et renvoyées à l’application.

### 3. Gestion parallèle de messages
- Plusieurs messages en cours de réception sont stockés dans une `HashMap` indexée par leur ID de message.
- Chaque entrée contient :
  - Le buffer alloué.
  - Les blocs reçus.
  - L'adresse source.
  - La taille totale.
- Une fois un message complété, son entrée est supprimée pour libérer la mémoire.

### 4. Gestion des timeouts
- Les sockets UDP ont un délai d'attente de 5 secondes pour éviter que la réception ne bloque indéfiniment.

---

## Utilisation de la `HashMap`

### Rôle de la `HashMap`
- **Nombre :** 1 seule `HashMap`.
- **Structure :** `HashMap<u32, MessageBuffer>`.
  - La clé est un **ID unique** pour chaque message (`u32`).
  - La valeur est une structure `MessageBuffer` contenant les données nécessaires pour gérer un message en cours de réception.

### Contenu d'un `MessageBuffer`
Un `MessageBuffer` contient les informations suivantes :
1. **`total_size` (u32)** : Taille totale des données à recevoir, déterminée depuis le header du premier paquet.
2. **`buffer` (Vec<u8>)** : Un vecteur pré-alloué pour stocker les fragments du message.
3. **`bytes_received` (usize)** : Le nombre total d'octets reçus pour ce message.
4. **`blocks_received` (HashSet<u32>)** : Un ensemble contenant les IDs des blocs déjà reçus pour éviter les doublons.
5. **`source_addr` (SocketAddr)** : L'adresse source (IP et port) du message.

### Fonctionnement détaillé de la `HashMap`
1. **Ajout d'un nouveau message :**
   - Lorsqu'un paquet est reçu pour un message inconnu (`message_id` non présent dans la `HashMap`), un nouveau `MessageBuffer` est créé et ajouté.
   - La mémoire est allouée immédiatement en fonction de la taille totale indiquée dans le header.
   
2. **Mise à jour d'un message existant :**
   - Si un paquet est reçu pour un `message_id` déjà connu :
     - Le fragment est inséré dans le `buffer` au bon emplacement, calculé via l'ID du bloc.
     - L'ID du bloc est ajouté à `blocks_received` pour éviter de traiter un fragment en double.

3. **Suppression d'un message complet :**
   - Une fois tous les blocs d’un message reçus, le message est reconstruit et renvoyé à l'application.
   - L'entrée correspondante dans la `HashMap` est supprimée, libérant ainsi la mémoire utilisée.

---

## Résumé

### Points clés
- **Transmission fiable :** La librairie gère manuellement le découpage et la reconstruction des données sur UDP.
- **Gestion efficace de la mémoire :** La `HashMap` permet de stocker les données des messages en cours de réception.
- **Traitement parallèle :** La structure gère plusieurs messages simultanément grâce à un système d'ID unique.

### Schéma simplifié
Envoyer :
	• Découpe en blocs -> Ajout header -> Envoi via UDP.

Recevoir :
	• Identifier via message_id -> Allouer mémoire si nouveau message -> Ajouter fragment -> Vérifier complétude -> Renvoyer message complet.