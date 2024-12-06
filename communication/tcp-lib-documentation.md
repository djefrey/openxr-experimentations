# Documentation de la bibliothèque de communication TCP

Cette bibliothèque fournit une interface simple pour la création de serveurs TCP et de clients TCP en Rust. Elle offre des fonctionnalités pour établir des connexions, envoyer et recevoir des données (y compris des types Pod), et gérer plusieurs connexions simultanées côté serveur.

## Table des matières

1. [Installation](#installation)
2. [Utilisation de base](#utilisation-de-base)
   - [Création d'un serveur](#création-dun-serveur)
   - [Création d'un client](#création-dun-client)
3. [API de la bibliothèque](#api-de-la-bibliothèque)
   - [Struct TcpServer](#struct-tcpserver)
   - [Struct TcpConnection](#struct-tcpconnection)
4. [Exemples](#exemples)
   - [Exemple de serveur](#exemple-de-serveur)
   - [Exemple de client](#exemple-de-client)
5. [Utilisation des types Pod](#utilisation-des-types-pod)
6. [Bonnes pratiques](#bonnes-pratiques)

## Installation

Pour utiliser cette bibliothèque dans votre projet Rust, ajoutez les dépendances suivantes à votre fichier `Cargo.toml`:

```toml
[dependencies]
tcp_communication = { path = "src/" }
bytemuck = "1.19"
```

## Utilisation de base

### Création d'un serveur

```rust
use tcp_communication::{TcpServer, TcpConnection};

fn main() -> std::io::Result<()> {
    let server = TcpServer::new("127.0.0.1:7878")?;
    println!("Serveur en écoute sur 127.0.0.1:7878");

    server.run(|connection| {
        // Gérer la connexion ici
    })
}
```

### Création d'un client

```rust
use tcp_communication::TcpConnection;

fn main() -> std::io::Result<()> {
    let mut connection = TcpConnection::connect("127.0.0.1:7878")?;
    println!("Connecté au serveur sur 127.0.0.1:7878");

    // Utiliser la connexion pour envoyer et recevoir des données
}
```

## API de la bibliothèque

### Struct TcpServer

- `new(address: &str) -> io::Result<TcpServer>`: Crée un nouveau serveur TCP lié à l'adresse spécifiée.
- `accept() -> io::Result<TcpConnection>`: Accepte une nouvelle connexion cliente.
- `run<F>(&self, handler: F) -> io::Result<()>`: Exécute le serveur, gérant chaque connexion avec la fonction de gestion fournie.

### Struct TcpConnection

- `connect(address: &str) -> io::Result<TcpConnection>`: Établit une connexion avec un serveur à l'adresse spécifiée.
- `send<T: AsRef<[u8]>>(&mut self, data: T) -> io::Result<()>`: Envoie des données sur la connexion.
- `receive(&mut self, buffer: &mut [u8]) -> io::Result<usize>`: Reçoit des données sur la connexion.
- `send_pod<T: Pod>(&mut self, data: &T) -> io::Result<()>`: Envoie des données de type Pod sur la connexion.
- `receive_pod<T: Pod + Zeroable>(&mut self) -> io::Result<T>`: Reçoit des données de type Pod sur la connexion.
- `close(self) -> io::Result<()>`: Ferme la connexion.

## Exemples

### Exemple de serveur

```rust
use tcp_communication::{TcpServer, TcpConnection};
use std::io;

fn handle_client(mut connection: TcpConnection) -> io::Result<()> {
    let mut buffer = [0; 1024];
    loop {
        let bytes_read = connection.receive(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        let received = String::from_utf8_lossy(&buffer[..bytes_read]);
        println!("Reçu : {}", received);
        let response = format!("Écho : {}", received);
        connection.send(response)?;
    }
    Ok(())
}

fn main() -> io::Result<()> {
    let server = TcpServer::new("127.0.0.1:7878")?;
    println!("Serveur en écoute sur 127.0.0.1:7878");
    server.run(handle_client)
}
```

### Exemple de client

```rust
use tcp_communication::TcpConnection;
use std::io::{self, Write};

fn main() -> io::Result<()> {
    let mut connection = TcpConnection::connect("127.0.0.1:7878")?;
    println!("Connecté au serveur sur 127.0.0.1:7878");

    loop {
        print!("Entrez un message (ou 'quit' pour sortir) : ");
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        let trimmed = input.trim();
        if trimmed == "quit" {
            break;
        }

        connection.send(trimmed)?;

        let mut buffer = [0; 1024];
        let bytes_read = connection.receive(&mut buffer)?;
        println!("Réponse : {}", String::from_utf8_lossy(&buffer[..bytes_read]));
    }

    Ok(())
}
```

## Utilisation des types Pod

Pour utiliser les méthodes `send_pod` et `receive_pod`, vos types doivent implémenter les traits `Pod` et `Zeroable` de `bytemuck`. Voici un exemple :

```rust
use tcp_communication::TcpConnection;
use bytemuck::{Pod, Zeroable};

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct MyData {
    x: i32,
    y: f32,
}

fn main() -> std::io::Result<()> {
    let mut connection = TcpConnection::connect("127.0.0.1:7878")?;
    
    let data = MyData { x: 42, y: 3.14 };
    connection.send_pod(&data)?;

    let received: MyData = connection.receive_pod()?;
    println!("Reçu : x = {}, y = {}", received.x, received.y);

    Ok(())
}
```

## Bonnes pratiques

1. Gérez toujours les erreurs de manière appropriée en utilisant `Result` et `?`.
2. Fermez explicitement les connexions lorsque vous avez terminé de les utiliser.
3. Utilisez des tampons de taille appropriée pour recevoir des données.
4. Considérez l'utilisation de protocoles de niveau supérieur pour structurer vos messages.
5. Implémentez une logique de reconnexion côté client pour une meilleure résilience.
6. Utilisez les types Pod pour des structures de données simples et bien définies qui doivent être transmises efficacement.

Cette bibliothèque fournit une base solide pour la communication TCP, y compris la prise en charge des types Pod pour une transmission efficace de données structurées. Selon vos besoins spécifiques, vous pourriez avoir besoin d'ajouter des fonctionnalités supplémentaires comme le chiffrement, la compression, ou des protocoles personnalisés.