// Importation des modules nécessaires pour la communication réseau TCP et les opérations d'entrée/sortie
use std::net::{TcpListener, TcpStream}; // Importation des structures TcpListener et TcpStream pour la communication TCP
use std::io::{self, Read, Write}; // Importation des fonctionnalités d'entrée/sortie, y compris les traits Read et Write pour la lecture et l'écriture des flux
use std::sync::Arc; // Importation d'Arc pour permettre un partage sûr des ressources entre threads
use std::thread; // Importation du module de gestion des threads
use bytemuck::{Pod, Zeroable}; // Importation des traits Pod et Zeroable pour la manipulation de types de données de valeur de pod

// Structure représentant un serveur TCP
pub struct TcpServer {
    listener: TcpListener, // Le listener TCP qui attend les connexions entrantes
}

// Structure représentant une connexion TCP individuelle
pub struct TcpConnection {
    stream: TcpStream, // Le flux TCP utilisé pour la communication avec un client
}

// Implémentation des méthodes pour la structure TcpServer
impl TcpServer {
    // Crée un nouveau serveur TCP qui écoute sur l'adresse spécifiée
    pub fn new(address: &str) -> io::Result<Self> {
        let listener = TcpListener::bind(address)?; // Lie le listener à l'adresse fournie, retourne une erreur en cas d'échec
        Ok(TcpServer { listener }) // Retourne une instance de TcpServer avec le listener initialisé
    }

    // Accepte une connexion entrante et retourne un TcpConnection
    pub fn accept(&self) -> io::Result<TcpConnection> {
        let (stream, _) = self.listener.accept()?; // Accepte une connexion entrante et récupère le flux
        Ok(TcpConnection { stream }) // Retourne une instance de TcpConnection avec le flux connecté
    }

    // Démarre le serveur et exécute un gestionnaire pour chaque connexion cliente
    // 'F' est un type générique qui représente une fonction ou une fermeture prenant un TcpConnection
    pub fn run<F>(&self, handler: F) -> io::Result<()>
        where
            F: Fn(TcpConnection) + Send + Sync + 'static, // Les contraintes garantissent que la fermeture peut être envoyée entre threads, partagée et a une durée de vie statique
    {
        // Utilisation d'Arc pour permettre de partager le gestionnaire entre plusieurs threads
        // Arc (Atomic Reference Counted) est un pointeur intelligent qui permet de partager une valeur entre plusieurs threads de manière sûre.
        // Contrairement à Rc, qui n'est pas thread safe, Arc garantit la sécurité lors de l'accès concurrent grâce à un comptage de références atomique.
        // Ici, Arc est utilisé pour partager le gestionnaire entre les différents threads créés pour chaque connexion client.
        let handler = Arc::new(handler);

        // Boucle sur chaque connexion entrante
        for stream in self.listener.incoming() {
            match stream {
                Ok(stream) => {
                    // Crée une nouvelle connexion TcpConnection
                    let connection = TcpConnection { stream };
                    // Clone l'Arc pour que le gestionnaire soit accessible par le nouveau thread
                    // Arc::clone() augmente le comptage de références, garantissant que la ressource reste disponible tant qu'il y a des références actives.
                    let handler = Arc::clone(&handler);
                    // Crée un nouveau thread pour gérer la connexion client de manière concurrente
                    thread::spawn(move || {
                        handler(connection); // Appelle le gestionnaire pour traiter la connexion
                    });
                }
                // En cas d'erreur lors de l'acceptation de la connexion, affiche un message d'erreur
                Err(e) => eprintln!("Erreur de connexion : {}", e),
            }
        }
        Ok(()) // Retourne Ok si le serveur se termine correctement (ce qui est peu probable, car le serveur tourne en boucle infinie)
    }
}

// Implémentation des méthodes pour la structure TcpConnection
impl TcpConnection {
    // Crée une nouvelle connexion TCP en se connectant à une adresse spécifiée
    pub fn connect(address: &str) -> io::Result<Self> {
        let stream = TcpStream::connect(address)?; // Tente de se connecter à l'adresse fournie, retourne une erreur en cas d'échec
        Ok(TcpConnection { stream }) // Retourne une instance de TcpConnection avec le flux connecté
    }

    // Envoie des données via la connexion TCP
    pub fn send<T: AsRef<[u8]>>(&mut self, data: T) -> io::Result<()> {
        self.stream.write_all(data.as_ref()) // Écrit toutes les données dans le flux, retourne une erreur en cas d'échec
    }

    // Reçoit des données depuis la connexion TCP
    pub fn receive(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        self.stream.read(buffer) // Lit les données dans le tampon fourni et retourne le nombre d'octets lus
    }

    // Nouvelle méthode pour envoyer des données Pod
    pub fn send_pod<T: Pod>(&mut self, data: &T) -> io::Result<()> {
        let bytes = bytemuck::bytes_of(data);
        self.stream.write_all(bytes)
    }

    // Nouvelle méthode pour recevoir des données Pod
    pub fn receive_pod<T: Pod + Zeroable>(&mut self) -> io::Result<T> {
        let mut buffer = T::zeroed();
        let bytes = bytemuck::bytes_of_mut(&mut buffer);
        self.stream.read_exact(bytes)?;
        Ok(buffer)
    }

    // Ferme proprement la connexion TCP
    pub fn close(self) -> io::Result<()> {
        self.stream.shutdown(std::net::Shutdown::Both) // Ferme la connexion dans les deux directions (lecture et écriture)
    }
}