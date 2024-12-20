// src/udp_lib.rs

use std::net::{UdpSocket, SocketAddr};
use std::io::{self, ErrorKind};
use std::collections::{HashMap, HashSet};
use std::time::Duration;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;

// Taille maximale d'un paquet UDP (32 KB)
const MAX_UDP_PACKET_SIZE: usize = 4 * 1024;

// Taille du header : 12 octets (4 octets pour l'ID du message, 4 octets pour la taille totale, 4 octets pour l'ID du bloc)
const HEADER_SIZE: usize = 12;

// Générateur d'ID unique pour chaque message
static MESSAGE_ID_COUNTER: AtomicU32 = AtomicU32::new(0);

// Structure pour stocker les informations d'un message en cours de réception
struct MessageBuffer {
    total_size: u32,
    buffer: Vec<u8>,
    bytes_received: usize,
    blocks_received: HashSet<u32>,
    source_addr: SocketAddr,
}

// Structure représentant une communication UDP
pub struct UdpCommunication {
    socket: UdpSocket,
    messages: HashMap<u32, MessageBuffer>,
}

impl UdpCommunication {
    // Crée une nouvelle communication UDP liée à l'adresse spécifiée
    pub fn new(bind_address: &str) -> io::Result<Self> {
        let socket = UdpSocket::bind(bind_address)?;
        socket.set_read_timeout(Some(Duration::from_secs(5)))?;
        Ok(UdpCommunication {
            socket,
            messages: HashMap::new(),
        })
    }

    // Envoie des données vers l'adresse cible en les découpant en blocs avec un header
    pub fn send(&self, data: &[u8], target_address: &str) -> io::Result<()> {
        let target: SocketAddr = target_address.parse().expect("Adresse cible invalide");

        // Générer un ID unique pour le message
        let message_id = MESSAGE_ID_COUNTER.fetch_add(1, Ordering::SeqCst);

        let total_size = data.len() as u32;
        let max_chunk_size = MAX_UDP_PACKET_SIZE - HEADER_SIZE;
        let mut block_id: u32 = 0;
        let mut offset = 0;

        while offset < data.len() {
            let chunk_size = usize::min(max_chunk_size, data.len() - offset);
            let end = offset + chunk_size;

            // Préparer le buffer avec le header
            let mut buffer = Vec::with_capacity(HEADER_SIZE + chunk_size);
            buffer.extend_from_slice(&message_id.to_be_bytes());  // ID du message
            buffer.extend_from_slice(&total_size.to_be_bytes());  // Taille totale des données
            buffer.extend_from_slice(&block_id.to_be_bytes());    // ID du bloc
            buffer.extend_from_slice(&data[offset..end]);         // Données du bloc

            // Envoyer le paquet UDP
            self.socket.send_to(&buffer, target)?;

            offset = end;
            block_id += 1;
        }

        Ok(())
    }

    pub fn receive(&mut self) -> io::Result<(Vec<u8>, SocketAddr)> {
        let mut buffer = [0u8; MAX_UDP_PACKET_SIZE];

        loop {
            match self.socket.recv_from(&mut buffer) {
                Ok((bytes_read, addr)) => {
                    if bytes_read >= HEADER_SIZE {
                        let header = &buffer[..HEADER_SIZE];
                        let data = &buffer[HEADER_SIZE..bytes_read];

                        let message_id = u32::from_be_bytes(header[0..4].try_into().unwrap());
                        let total_size = u32::from_be_bytes(header[4..8].try_into().unwrap());
                        let block_id = u32::from_be_bytes(header[8..12].try_into().unwrap());

                        let is_complete;
                        let full_data;
                        let source_addr;

                        // Accéder au buffer de message ou en créer un nouveau
                        {
                            let message_buffer = self.messages.entry(message_id).or_insert_with(|| {
                                let buffer = vec![0u8; total_size as usize];
                                MessageBuffer {
                                    total_size,
                                    buffer,
                                    bytes_received: 0,
                                    blocks_received: HashSet::new(),
                                    source_addr: addr,
                                }
                            });

                            if addr != message_buffer.source_addr {
                                continue;
                            }

                            if message_buffer.blocks_received.contains(&block_id) {
                                continue;
                            }

                            let max_chunk_size = MAX_UDP_PACKET_SIZE - HEADER_SIZE;
                            let offset = (block_id as usize) * max_chunk_size;
                            let end = offset + data.len();

                            if end > message_buffer.buffer.len() {
                                return Err(io::Error::new(ErrorKind::InvalidData, "Dépassement du buffer"));
                            }

                            message_buffer.buffer[offset..end].copy_from_slice(data);
                            message_buffer.bytes_received += data.len();
                            message_buffer.blocks_received.insert(block_id);

                            let total_blocks = ((message_buffer.total_size as usize + max_chunk_size - 1) / max_chunk_size) as u32;

                            // Vérifier si tous les blocs sont reçus
                            is_complete = message_buffer.blocks_received.len() as u32 == total_blocks;
                            if is_complete {
                                full_data = message_buffer.buffer.clone();
                                source_addr = message_buffer.source_addr;
                            } else {
                                continue;
                            }
                        }

                        // Supprimer le message si complet
                        if is_complete {
                            self.messages.remove(&message_id);
                            return Ok((full_data, source_addr));
                        }
                    }
                }
                Err(ref e) if e.kind() == ErrorKind::WouldBlock || e.kind() == ErrorKind::TimedOut => {
                    return Err(io::Error::new(ErrorKind::TimedOut, "Délai de réception dépassé"));
                }
                Err(e) => return Err(e),
            }
        }
    }
}
