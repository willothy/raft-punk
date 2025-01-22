use std::{
    hash::Hash,
    ops::Deref,
    sync::{atomic::AtomicU64, Arc},
    time::Duration,
};

use crossbeam::atomic::AtomicCell;
use serde::{Deserialize, Serialize};
use snafu::{whatever, ResultExt};
use tokio::{sync::RwLock, time::Instant};

#[derive(Debug, snafu::Snafu)]
pub enum Error {
    #[snafu(display("Peer not found: {:?}", peer))]
    PeerNotFound { peer: NodeId },

    #[snafu(whatever, display("{message}"))]
    Whatever {
        message: String,
        #[snafu(source(from(Box<dyn std::error::Error>, Some)))]
        source: Option<Box<dyn std::error::Error>>,
    },
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RaftState {
    Follower,
    Candidate,
    Leader,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry<Command> {
    pub term: Term,
    pub command: Command,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Term(AtomicU64);

impl Clone for Term {
    fn clone(&self) -> Self {
        Term::new(self.0.load(std::sync::atomic::Ordering::SeqCst))
    }
}

impl PartialEq for Term {
    fn eq(&self, other: &Self) -> bool {
        self.0.load(std::sync::atomic::Ordering::SeqCst)
            == other.0.load(std::sync::atomic::Ordering::SeqCst)
    }
}

impl Eq for Term {}

impl PartialOrd for Term {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0
            .load(std::sync::atomic::Ordering::SeqCst)
            .partial_cmp(&other.0.load(std::sync::atomic::Ordering::SeqCst))
    }
}

impl Ord for Term {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .load(std::sync::atomic::Ordering::SeqCst)
            .cmp(&other.0.load(std::sync::atomic::Ordering::SeqCst))
    }
}

impl Hash for Term {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.load(std::sync::atomic::Ordering::SeqCst).hash(state);
    }
}

impl Term {
    pub fn new(term: u64) -> Self {
        Term(AtomicU64::new(term))
    }

    pub fn zero() -> Self {
        Term(AtomicU64::new(0))
    }

    pub fn store(&self, term: Term) {
        self.0
            .store(term.into(), std::sync::atomic::Ordering::SeqCst);
    }

    pub fn increment(&self) {
        self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }

    pub fn into_inner(self) -> AtomicU64 {
        self.0
    }
}

impl Into<u64> for Term {
    fn into(self) -> u64 {
        self.0.load(std::sync::atomic::Ordering::SeqCst)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct NodeId(u64);

impl NodeId {
    pub fn new(id: u64) -> Self {
        NodeId(id)
    }

    pub fn into_inner(self) -> u64 {
        self.0
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestVoteRequest {
    pub term: Term,
    pub candidate_id: NodeId,
    pub last_log_index: u64,
    pub last_log_term: Term,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestVoteResponse {
    pub term: Term,
    pub vote_granted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppendEntriesRequest<Command> {
    pub term: Term,
    pub leader_id: NodeId,
    pub prev_log_index: u64,
    pub prev_log_term: Term,
    pub entries: Vec<LogEntry<Command>>,
    pub leader_commit: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppendEntriesResponse {
    pub term: Term,
    pub success: bool,
    // TODO: conflict handling
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum RaftMessage<Command> {
    RequestVoteRequest(RequestVoteRequest),
    RequestVoteResponse(RequestVoteResponse),
    AppendEntriesRequest(AppendEntriesRequest<Command>),
    AppendEntriesResponse(AppendEntriesResponse),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Peer {
    id: NodeId,
    address: String,
}

pub struct RaftNodeInner<Command> {
    pub id: NodeId,
    pub state: AtomicCell<RaftState>,
    pub current_term: Term,
    pub voted_for: AtomicCell<Option<NodeId>>,
    pub log: RwLock<Vec<LogEntry<Command>>>,

    pub commit_index: AtomicU64,
    pub last_applied: AtomicU64,

    // Leader state
    pub next_index: papaya::HashMap<NodeId, u64>,
    pub match_index: papaya::HashMap<NodeId, u64>,

    pub last_heartbeat: tokio::time::Instant,

    // Cluster info
    pub peers: papaya::HashMap<NodeId, Peer>,

    // Timeouts, etc.
    pub election_timeout: tokio::time::Duration,
    pub heartbeat_interval: tokio::time::Duration,

    pub http: reqwest::Client,
}

#[derive(Clone)]
pub struct RaftNode<Command> {
    inner: Arc<RaftNodeInner<Command>>,
}

impl<Command> Deref for RaftNode<Command> {
    type Target = RaftNodeInner<Command>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<Command: Clone + Serialize + Send + Sync + 'static> RaftNode<Command> {
    pub fn new(id: NodeId, peers: papaya::HashMap<NodeId, Peer>) -> Self {
        Self {
            inner: Arc::new(RaftNodeInner {
                id,
                state: crossbeam::atomic::AtomicCell::new(RaftState::Follower),
                peers,

                current_term: Term::new(0),
                voted_for: AtomicCell::new(None),

                log: RwLock::new(Vec::with_capacity(1024)),

                commit_index: AtomicU64::new(0),
                last_applied: AtomicU64::new(0),

                next_index: papaya::HashMap::new(),
                match_index: papaya::HashMap::new(),

                last_heartbeat: tokio::time::Instant::now(),

                // Set a random range for election_timeout (150–300 ms) which is a typical
                // range used in the Raft paper’s example.
                election_timeout: tokio::time::Duration::from_millis(
                    150 + rand::random::<u64>() % 150,
                ),

                heartbeat_interval: tokio::time::Duration::from_millis(50),

                http: reqwest::Client::new(),
            }),
        }
    }

    /// Sends AppendEntries to each follower to ensure they get up-to-date with the leader’s log.
    pub async fn broadcast_append_entries(&self) {
        if self.state.load() != RaftState::Leader {
            return;
        }

        for &peer_id in self.peers.pin().keys() {
            if peer_id == self.id {
                continue; // Don't send to ourselves
            }

            // Determine which entries the follower is missing
            let next_idx = {
                let guard = self.next_index.guard();
                *self.next_index.get(&peer_id, &guard).unwrap()
            };
            let prev_log_index = next_idx.saturating_sub(1);

            let (prev_log_term, entries_to_send) = {
                let log = self.log.read().await;

                let prev_log_term = if prev_log_index == 0 {
                    Term::zero()
                } else {
                    log[(prev_log_index - 1) as usize].term.clone()
                };

                // Prepare the slice of entries from next_idx onward:
                let entries_to_send = if next_idx <= log.len() as u64 {
                    log[(next_idx - 1) as usize..].to_vec()
                } else {
                    vec![]
                };

                (prev_log_term, entries_to_send)
            };

            let args = AppendEntriesRequest {
                term: self.current_term.clone(),
                leader_id: self.id,
                prev_log_index,
                prev_log_term,
                entries: entries_to_send,
                leader_commit: self.commit_index.load(std::sync::atomic::Ordering::SeqCst),
            };

            tokio::spawn({
                let node_clone = self.clone();
                async move {
                    match node_clone.send_append_entries(peer_id, args).await {
                        Ok(_) => {}
                        Err(e) => {
                            tracing::warn!("Failed to send AppendEntries to peer: {:?}", e);
                        }
                    }
                }
            });
        }
    }

    pub async fn start_election(&self) -> Result<()> {
        self.current_term.increment();
        self.state.store(RaftState::Candidate);
        self.voted_for.store(Some(self.id));

        let (last_log_index, last_log_term) = {
            let log = self.log.read().await;
            let last_log_index = log.len().saturating_sub(1) as u64;
            let last_log_term = log
                .last()
                .map(|entry| entry.term.clone())
                .unwrap_or(Term::zero());
            (last_log_index, last_log_term)
        };

        let req = RequestVoteRequest {
            term: self.current_term.clone(),
            candidate_id: self.id,
            last_log_index,
            last_log_term,
        };

        let mut votes = 1;
        for peer in self.peers.pin().values() {
            if peer.id == self.id {
                continue;
            }

            match self.send_request_vote(peer.id, req.clone()).await {
                Ok(reply) => {
                    if reply.vote_granted {
                        votes += 1;
                    } else if reply.term > self.current_term {
                        self.current_term.store(reply.term);
                        self.state.store(RaftState::Follower);
                        self.voted_for.store(None);
                        return Ok(());
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to send request vote to peer: {:?}", e);
                    // No response from the peer
                }
            }
        }

        // Check if we have a quorum
        if votes > self.peers.len() / 2 {
            self.become_leader().await?;
        } else {
            // If we didn't get a majority, remain candidate or revert to follower
            // Typically we wait for the election timeout again to start a new round
            self.state.store(RaftState::Follower);
            self.voted_for.store(None);
        }

        Ok(())
    }

    pub async fn become_leader(&self) -> Result<()> {
        self.state.store(RaftState::Leader);

        let last_index = self.log.read().await.len() as u64;
        for peer in self.peers.pin().values() {
            self.next_index.pin().insert(peer.id, last_index);
            self.match_index.pin().insert(peer.id, 0);
        }

        self.send_heartbeats().await?;

        Ok(())
    }

    pub async fn send_request_vote(
        &self,
        peer: NodeId,
        req: RequestVoteRequest,
    ) -> Result<RequestVoteResponse> {
        let addr = self
            .peers
            .pin()
            .get(&peer)
            .ok_or_else(|| Error::PeerNotFound { peer })?
            .address
            .clone();

        let res = self
            .http
            .post(&addr)
            .json(&RaftMessage::<Command>::RequestVoteRequest(req))
            .send()
            .await
            .whatever_context("failed to send request vote")?;

        let data = res
            .json::<RequestVoteResponse>()
            .await
            .whatever_context("failed to parse request vote response")?;

        // Send the request to the peer and return the response
        Ok(data)
    }

    /// Appends a new command to the log (if leader) and replicates it.
    pub async fn append_new_entry(&self, command: Command) -> Result<()> {
        // 1. Only a leader can accept new entries.
        if self.state.load() != RaftState::Leader {
            whatever!("Not the leader");
        }

        // 2. Create a new log entry for the command.
        let new_entry = LogEntry {
            term: self.current_term.clone(),
            command,
        };
        self.log.write().await.push(new_entry);

        // 3. Update our next_index and match_index as needed.
        // For simplicity, assume the new entry goes at the end (index = log.len() - 1).
        let new_entry_index = (self.log.read().await.len() - 1) as u64;
        self.match_index.pin().insert(self.id, new_entry_index);

        // 4. Send AppendEntries RPC to each follower to replicate the new entry.
        // In a real system, you'd also track acknowledgments from peers to detect
        // when the entry is safely replicated by a majority.
        for peer_id in self.peers.pin().keys() {
            if *peer_id == self.id {
                continue;
            }

            // Construct AppendEntries with the newly appended log entry.
            let prev_log_index = new_entry_index.saturating_sub(1);

            let args = {
                let log = self.log.read().await;

                let prev_log_term = {
                    if prev_log_index < log.len() as u64 {
                        log[prev_log_index as usize].term.clone()
                    } else {
                        Term::zero()
                    }
                };

                let args = AppendEntriesRequest {
                    term: self.current_term.clone(),
                    leader_id: self.id,
                    prev_log_index,
                    prev_log_term,
                    // Send just the one new entry for now:
                    entries: vec![log[new_entry_index as usize].clone()],
                    leader_commit: self.commit_index.load(std::sync::atomic::Ordering::SeqCst),
                };

                args
            };

            // Fire off the RPC (stubbed or real, depending on your setup).
            self.send_append_entries(*peer_id, args).await?;
        }

        Ok(())
    }

    pub async fn send_heartbeats(&self) -> Result<()> {
        if self.state.load() != RaftState::Leader {
            return Ok(());
        }

        for peer in self.peers.pin().values() {
            if peer.id == self.id {
                continue;
            }

            let req = AppendEntriesRequest {
                term: self.current_term.clone(),
                leader_id: self.id,
                prev_log_index: 0,
                prev_log_term: Term::zero(),
                entries: vec![],
                leader_commit: self.commit_index.load(std::sync::atomic::Ordering::SeqCst),
            };
            self.send_append_entries(peer.id, req).await?;
        }

        Ok(())
    }

    pub fn apply_entries(&self) -> Result<()> {
        use std::sync::atomic::Ordering;
        while self.last_applied.load(Ordering::SeqCst) < self.commit_index.load(Ordering::SeqCst) {
            self.last_applied.fetch_add(1, Ordering::SeqCst);
            // TODO: Handle applying the command
            // let entry = &self.log[self.last_applied as usize];
        }
        Ok(())
    }

    pub async fn send_append_entries(
        &self,
        peer: NodeId,
        req: AppendEntriesRequest<Command>,
    ) -> Result<AppendEntriesResponse> {
        let addr = self
            .peers
            .pin()
            .get(&peer)
            .ok_or_else(|| Error::PeerNotFound { peer })?
            .address
            .clone();

        let res = self
            .http
            .post(&addr)
            .json(&RaftMessage::<Command>::AppendEntriesRequest(req))
            .send()
            .await
            .whatever_context("failed to send request vote")?;

        let data = res
            .json::<AppendEntriesResponse>()
            .await
            .whatever_context("failed to parse request vote response")?;

        // Send the request to the peer and return the response
        Ok(data)
    }

    pub async fn run(&self) -> Result<()> {
        loop {
            let now = Instant::now();
            // Check if we need to start an election
            if self.state.load() != RaftState::Leader
                && now.duration_since(self.last_heartbeat) > self.election_timeout
            {
                // Move to candidate state
                self.start_election().await?;
            }

            // If we're the leader, send heartbeats periodically
            if self.state.load() == RaftState::Leader {
                // Send heartbeat if enough time has passed
                self.send_heartbeats().await?;
            }

            // Sleep for a short interval to avoid busy looping
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // fn create_test_peers(count: u64) -> HashMap<NodeId, Peer> {
    //     let mut peers = HashMap::new();
    //     for i in 1..=count {
    //         peers.insert(
    //             NodeId::new(i),
    //             Peer {
    //                 id: NodeId::new(i),
    //                 address: format!("http://localhost:30{:02}", i),
    //             },
    //         );
    //     }
    //     peers
    // }

    #[tokio::test]
    async fn test_single_node_leadership() {
        // Single node cluster becomes leader immediately
        let node_id = NodeId::new(1);
        let node = RaftNode::<()>::new(node_id, papaya::HashMap::new());

        node.start_election().await.unwrap();
        assert_eq!(node.state.load(), RaftState::Leader);
    }
}
