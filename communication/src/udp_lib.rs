// src/udp_lib.rs

// Importation des modules nécessaires pour la communication UDP
use std::net::{UdpSocket, SocketAddr}; // UdpSocket pour la communication UDP, SocketAddr pour représenter les adresses socket
use std::io::{self, ErrorKind}; // Pour la gestion des erreurs d'entrée/sortie
use std::collections::HashMap; // Pour stocker les fragments reçus
use std::time::Duration; // Pour gérer les délais d'expiration

// Taille maximale d'un paquet UDP (32 KB)
const MAX_UDP_PACKET_SIZE: usize = 4 * 1024;

// Taille du header : 8 octets (4 octets pour la taille totale, 4 octets pour l'ID du bloc)
const HEADER_SIZE: usize = 8;

// Structure représentant une communication UDP
pub struct UdpCommunication {
    socket: UdpSocket,
}

impl UdpCommunication {
    // Crée une nouvelle communication UDP liée à l'adresse spécifiée
    pub fn new(bind_address: &str) -> io::Result<Self> {
        let socket = UdpSocket::bind(bind_address)?;
        socket.set_read_timeout(Some(Duration::from_secs(5)))?; // Définir un délai d'attente de 5 secondes pour la lecture

        Ok(UdpCommunication { socket })
    }

    // Envoie des données vers l'adresse cible en les découpant en blocs de 32 KB avec un header
    pub fn send(&self, data: &[u8], target_address: &str) -> io::Result<()> {
        let target: SocketAddr = target_address.parse().expect("Adresse cible invalide");

        // Calculer le nombre total de blocs
        let total_size = data.len() as u32;
        let mut block_id: u32 = 0;
        let mut offset = 0;

        while offset < data.len() {
            let chunk_size = usize::min(MAX_UDP_PACKET_SIZE - HEADER_SIZE, data.len() - offset);
            let end = offset + chunk_size;

            // Préparer le buffer avec le header
            let mut buffer = Vec::with_capacity(HEADER_SIZE + chunk_size);
            buffer.extend_from_slice(&total_size.to_be_bytes()); // Taille totale des données
            buffer.extend_from_slice(&block_id.to_be_bytes());   // ID du bloc
            buffer.extend_from_slice(&data[offset..end]);        // Données du bloc

            // Envoyer le paquet UDP
            self.socket.send_to(&buffer, target)?;

            // Mettre à jour l'offset et l'ID du bloc
            offset = end;
            block_id += 1;
        }

        Ok(())
    }

    // Reçoit des données depuis n'importe quelle adresse et reconstruit les données complètes
    pub fn receive(&self) -> io::Result<(Vec<u8>, SocketAddr)> {
        // Tampon pour recevoir les paquets UDP (taille maximale du paquet)
        let mut buffer = [0u8; MAX_UDP_PACKET_SIZE];

        // Stocke les fragments reçus, avec l'ID du bloc comme clé
        let mut fragments: HashMap<u32, Vec<u8>> = HashMap::new();

        // Taille totale des données à recevoir (extraite du header du premier paquet)
        let mut total_size: Option<u32> = None;

        // Nombre total d'octets de données reçus
        let mut received_size = 0;

        // Adresse source des paquets (définie lors de la réception du premier paquet)
        let mut source_addr: Option<SocketAddr> = None;

        // Boucle pour recevoir tous les fragments nécessaires
        loop {
            match self.socket.recv_from(&mut buffer) {
                Ok((bytes_read, addr)) => {
                    // Si c'est le premier paquet reçu, enregistrer l'adresse source
                    if source_addr.is_none() {
                        source_addr = Some(addr);
                    }

                    // Vérifier que le paquet provient de la même adresse source
                    if addr != source_addr.unwrap() {
                        continue; // Ignorer les paquets provenant d'autres adresses
                    }

                    // Vérifier que le paquet contient au moins le header
                    if bytes_read >= HEADER_SIZE {
                        // Extraire le header et les données du paquet
                        let header = &buffer[..HEADER_SIZE]; // Les 8 premiers octets
                        let data = &buffer[HEADER_SIZE..bytes_read]; // Le reste des données

                        // Extraire la taille totale des données et l'ID du bloc depuis le header
                        let total_size_bytes = &header[..4]; // Octets 0 à 3
                        let block_id_bytes = &header[4..8];  // Octets 4 à 7

                        // Convertir les octets en nombres entiers non signés de 32 bits (endianness big-endian)
                        let packet_total_size = u32::from_be_bytes(total_size_bytes.try_into().unwrap());
                        let block_id = u32::from_be_bytes(block_id_bytes.try_into().unwrap());

                        // Enregistrer la taille totale des données si elle n'est pas déjà définie
                        if total_size.is_none() {
                            total_size = Some(packet_total_size);
                        }

                        // Stocker le fragment reçu avec son ID dans la HashMap
                        fragments.insert(block_id, data.to_vec());

                        // Incrémenter le nombre total d'octets reçus
                        received_size += data.len();

                        // Vérifier si tous les fragments ont été reçus
                        if received_size as u32 >= total_size.unwrap() {
                            // Reconstruire les données complètes en assemblant les fragments dans l'ordre
                            let mut full_data = Vec::with_capacity(total_size.unwrap() as usize);

                            // Parcourir les IDs de fragments attendus
                            for i in 0..fragments.len() as u32 {
                                if let Some(chunk) = fragments.get(&i) {
                                    // Ajouter le fragment au vecteur des données complètes
                                    full_data.extend_from_slice(chunk);
                                } else {
                                    // Si un fragment manque, retourner une erreur
                                    return Err(io::Error::new(ErrorKind::InvalidData, "Fragment manquant"));
                                }
                            }

                            // Retourner les données complètes et l'adresse source
                            return Ok((full_data, addr));
                        }
                    }
                }
                // Si une erreur de type WouldBlock ou TimedOut se produit, retourner une erreur de délai dépassé
                Err(ref e) if e.kind() == ErrorKind::WouldBlock || e.kind() == ErrorKind::TimedOut => {
                    return Err(io::Error::new(ErrorKind::TimedOut, "Délai de réception dépassé"));
                }
                // Pour les autres erreurs, les propager
                Err(e) => return Err(e),
            }
        }
    }
}
