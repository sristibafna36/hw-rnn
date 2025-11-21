#!/usr/bin/env python3

# CS465 at Johns Hopkins University.

# Subclass ConditionalRandomFieldBackprop to get a biRNN-CRF model.

from __future__ import annotations
import logging
import torch.nn as nn
import torch.nn.functional as F
from math import inf, log, exp
from pathlib import Path
from typing_extensions import override
from typeguard import typechecked

import torch
from torch import Tensor, cuda
from jaxtyping import Float

from corpus import IntegerizedSentence, Sentence, Tag, TaggedCorpus, Word
from integerize import Integerizer
from crf_backprop import ConditionalRandomFieldBackprop, TorchScalar

logger = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.
    # Note: We use the name "logger" this time rather than "log" since we
    # are already using "log" for the mathematical log!

# Set the seed for random numbers in torch, for replicability
torch.manual_seed(1337)
cuda.manual_seed(69_420)  # No-op if CUDA isn't available

class ConditionalRandomFieldNeural(ConditionalRandomFieldBackprop):
    """A CRF that uses a biRNN to compute non-stationary potential
    matrices.  The feature functions used to compute the potentials
    are now non-stationary, non-linear functions of the biRNN
    parameters."""

    neural = True    # class attribute that indicates that constructor needs extra args
    
    @override
    def __init__(self, 
                 tagset: Integerizer[Tag],
                 vocab: Integerizer[Word],
                 lexicon: Tensor,
                 rnn_dim: int,
                 unigram: bool = False):
        # [doctring inherited from parent method]

        if unigram:
            raise NotImplementedError("Not required for this homework")

        self.rnn_dim = rnn_dim
        self.e = lexicon.size(1) # dimensionality of word's embeddings
        self.E = lexicon

        nn.Module.__init__(self)  
        super().__init__(tagset, vocab, unigram)


    @override
    def init_params(self) -> None:

        """
            Initialize all the parameters you will need to support a bi-RNN CRF
            This will require you to create parameters for M, M', U_a, U_b, theta_a
            and theta_b. Use xavier uniform initialization for the matrices and 
            normal initialization for the vectors. 
        """

        # See the "Parameterization" section of the reading handout to determine
        # what dimensions all your parameters will need.

        # BiRNN parameters (equations 46 from handout)
        if self.rnn_dim > 0:
            # Left-to-right RNN: h[j] = σ(M · [h[j-1]; w[j]])
            # Input: h[j-1] (d) + w[j] (e) = d+e dimensional
            # Output: h[j] (d) dimensional
            self.M = nn.Parameter(torch.empty(self.rnn_dim, self.rnn_dim + self.e))
            nn.init.xavier_uniform_(self.M)
            
            # Right-to-left RNN: h'[j-1] = σ(M' · [w[j]; h'[j]])
            # Input: w[j] (e) + h'[j] (d) = e+d dimensional
            # Output: h'[j-1] (d) dimensional
            self.M_prime = nn.Parameter(torch.empty(self.rnn_dim, self.e + self.rnn_dim))
            nn.init.xavier_uniform_(self.M_prime)
            
            # Feature extraction networks (equations 47, 48)
            # We'll use simplified version: single scalar weights
            # For full version, you'd use nn.Linear layers
            
            # For transitions: context is [1; h[i-2]; s; t; h'[i]]
            # Using one-hot for tags: size k each
            context_dim_A = 1 + 2*self.rnn_dim + 2*self.k
            self.U_A = nn.Parameter(torch.empty(1, context_dim_A))
            nn.init.xavier_uniform_(self.U_A)
            self.theta_A = nn.Parameter(torch.randn(1))
            
            # For emissions: context is [1; h[i-1]; t; w; h'[i]]
            context_dim_B = 1 + 2*self.rnn_dim + self.k + self.e
            self.U_B = nn.Parameter(torch.empty(1, context_dim_B))
            nn.init.xavier_uniform_(self.U_B)
            self.theta_B = nn.Parameter(torch.randn(1))
        else:
            # No RNN - will use stationary potentials
            pass
        
    @override
    def init_optimizer(self, lr: float, weight_decay: float) -> None:
        # [docstring will be inherited from parent]
    
        # Use AdamW optimizer for better training stability
        self.optimizer = torch.optim.AdamW( 
            params=self.parameters(),       
            lr=lr, weight_decay=weight_decay
        )                                   
        self.scheduler = None            
       
    @override
    def updateAB(self) -> None:
        # Nothing to do - self.A and self.B are not used in non-stationary CRFs
        pass

    @override
    def setup_sentence(self, isent: IntegerizedSentence) -> None:
        """Pre-compute the biRNN prefix and suffix contextual features (h and h'
        vectors) at all positions, as defined in the "Parameterization" section
        of the reading handout.  They can then be accessed by A_at() and B_at().
        
        Make sure to call this method from the forward_pass, backward_pass, and
        Viterbi_tagging methods of HiddenMarkovMOdel, so that A_at() and B_at()
        will have correct precomputed values to look at!"""

        n = len(isent) - 2  # exclude BOS, EOS
    
        if self.rnn_dim == 0:
            # No RNN - use empty vectors
            self.H = torch.zeros(n+2, 0)
            self.H_prime = torch.zeros(n+2, 0)
            return
        
        # Get word embeddings for sentence (excluding BOS at [0] and EOS at [n+1])
        words = [isent[j][0] for j in range(1, n+1)]  # w[1] to w[n]
        word_embs = self.E[words]  # n × e
        
        # Forward RNN: h[j] = σ(M · [h[j-1]; w[j]])
        self.H = torch.zeros(n+2, self.rnn_dim)
        # H[-1] = 0 (already zero, position before BOS)
        
        for j in range(n):
            # h[j] from h[j-1] and w[j+1] (word at position j+1 in isent)
            input_vec = torch.cat([self.H[j-1], word_embs[j]])  # (d+e)
            self.H[j] = torch.sigmoid(self.M @ input_vec)  # d
        
        # Backward RNN: h'[j-1] = σ(M' · [w[j]; h'[j]])
        self.H_prime = torch.zeros(n+2, self.rnn_dim)
        # H_prime[n+1] = 0 (already zero, position after EOS)
        
        for j in range(n, 0, -1):
            # h'[j-1] from w[j] and h'[j]
            input_vec = torch.cat([word_embs[j-1], self.H_prime[j]])  # (e+d)
            self.H_prime[j-1] = torch.sigmoid(self.M_prime @ input_vec)  # d

    @override
    def accumulate_logprob_gradient(self, sentence: Sentence, corpus: TaggedCorpus) -> None:
        isent = self._integerize_sentence(sentence, corpus)
        super().accumulate_logprob_gradient(sentence, corpus)

    @override
    @typechecked
    def A_at(self, position, sentence) -> Tensor:
        
        """Computes non-stationary k x k transition potential matrix using biRNN 
        contextual features and tag embeddings (one-hot encodings). Output should 
        be ϕA from the "Parameterization" section in the reading handout."""

        if self.rnn_dim == 0:
            # No context - return uniform potentials
            return torch.ones(self.k, self.k)
    
        # Get context vectors (equation 47)
        # For position i, we need h[i-2] and h'[i]
        prefix = self.H[position-2] if position >= 2 else torch.zeros(self.rnn_dim)
        suffix = self.H_prime[position] if position <= len(self.H_prime)-1 else torch.zeros(self.rnn_dim)
        
        # Compute potentials for all k² tag bigrams
        # φ_A(s,t,w,i) = exp(θ_A · f_A(s,t,w,i))
        # where f_A(s,t,w,i) = σ(U_A · [1; h[i-2]; s; t; h'[i]])
        
        A = torch.zeros(self.k, self.k)
        eye_k = torch.eye(self.k)  # one-hot tag embeddings
        
        for s in range(self.k):
            for t in range(self.k):
                # Build feature vector: [1; prefix; tag_s; tag_t; suffix]
                context = torch.cat([
                    torch.ones(1),      # bias
                    prefix,             # h[i-2]
                    eye_k[s],           # tag s (one-hot)
                    eye_k[t],           # tag t (one-hot)
                    suffix              # h'[i]
                ])
                
                # Compute potential (equation 45)
                features = torch.sigmoid(self.U_A @ context)
                A[s, t] = torch.exp(self.theta_A * features)
        
        return A
        
    @override
    @typechecked
    def B_at(self, position, sentence) -> Tensor:
        """Computes non-stationary k x V emission potential matrix using biRNN 
        contextual features, tag embeddings (one-hot encodings), and word embeddings. 
        Output should be ϕB from the "Parameterization" section in the reading handout."""
            
        if self.rnn_dim == 0:
            # No context - return uniform potentials
            return torch.ones(self.k, self.V)
        
        # Get word at this position
        w_idx = sentence[position][0]
        if w_idx >= self.V:  # Handle EOS/BOS
            return torch.ones(self.k, self.V)
        
        word_emb = self.E[w_idx]  # e
        
        # Get context vectors (equation 48)
        prefix = self.H[position-1] if position >= 1 else torch.zeros(self.rnn_dim)
        suffix = self.H_prime[position] if position <= len(self.H_prime)-1 else torch.zeros(self.rnn_dim)
        
        # Compute potentials for all k tags
        # φ_B(t,w,w,i) = exp(θ_B · f_B(t,w,w,i))
        # where f_B(t,w,w,i) = σ(U_B · [1; h[i-1]; t; w; h'[i]])
        
        B = torch.zeros(self.k, self.V)
        eye_k = torch.eye(self.k)
        
        for t in range(self.k):
            # Build feature vector: [1; prefix; tag_t; word; suffix]
            context = torch.cat([
                torch.ones(1),      # bias
                prefix,             # h[i-1]
                eye_k[t],           # tag t (one-hot)
                word_emb,           # word embedding
                suffix              # h'[i]
            ])
            
            # Compute potential (equation 45)
            features = torch.sigmoid(self.U_B @ context)
            # Only set potential for the actual word at this position
            B[t, w_idx] = torch.exp(self.theta_B * features)
        
        # Set small values for other words (they won't be used)
        B[:, :] = torch.where(B > 0, B, torch.tensor(1e-10))
        
        return B

        
