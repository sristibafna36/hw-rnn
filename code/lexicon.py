#!/usr/bin/env python3

# CS465 at Johns Hopkins University.
# Module for constructing a lexicon of word features.

import logging
from pathlib import Path
from typing import Optional, Set, List

import torch
from torch import Tensor

from corpus import TaggedCorpus, BOS_WORD, EOS_WORD, BOS_TAG, EOS_TAG, OOV_WORD, Word
from integerize import Integerizer

log = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.

def build_lexicon(corpus: TaggedCorpus,
                  one_hot: bool = False,
                  embeddings_file: Optional[Path] = None,
                  newvocab: Optional[Integerizer[Word]] = None,
                  problex: bool = False,
                  affixes: bool = False) -> torch.Tensor:
    
    """Returns a lexicon, implemented as a matrix Tensor where each row defines
    real-valued features (attributes) for one of the word types in corpus.vocab.
    This is a wrapper method that horizontally concatenates 0 or more matrices
    that provide different kinds of features.
       
    If embeddings_file is provided, then we also include all of its word types
    in the lexicon.  This is important in case they show up in test data. We
    must also add them to corpus.vocab (nondestructively: we replace
    corpus.vocab) so that they won't be integerized as OOV but will have their
    own integers, enabling us to look up their embeddings in the lexicon.  (Any
    other feature dimensions for these extra word types will be set to 0.)
       
    However: we've noticed that a large lexicon makes our training
    implementation run much slower on CPU (perhaps due to cache behavior?).
    Thus, if newvocab is provided, we decline to add word types that don't
    appear in newvocab, so they will be OOV.  During training, you can set
    newvocab to contain just the dev words.  Later during testing, you could
    replace the model's lexicon with one that includes the test words. (This
    change is possible because the embeddings of held-out words are just
    features, not model parameters that we trained.)
    """

    matrices = [torch.empty(len(corpus.vocab), 0)]  # start with no features for each word

    if one_hot: 
        matrices.append(one_hot_lexicon(corpus))
    if problex:
        matrices.append(problex_lexicon(corpus))
    if embeddings_file is not None:
        # must go last!
        matrices.append(embeddings_lexicon(corpus, embeddings_file,
                                           newvocab=newvocab))  
    if affixes:
        matrices.append(affixes_lexicon(corpus))
    
    # vocab size may have been increased by embeddings_lexicon.
    # Thus, some matrices may not have enough rows; pad them with zero rows.
    matrices = [torch.cat( (m,
                            torch.zeros(len(corpus.vocab) - m.size(0), 
                                        m.size(1))))
                for m in matrices]

    return torch.cat(matrices, dim=1)   # horizontally concatenate

def one_hot_lexicon(corpus: TaggedCorpus) -> torch.Tensor:
    """Return a matrix with as many rows as corpus.vocab, where 
    each row specifies a one-hot embedding of the corresponding word.
    This allows us to learn features that are specific to the word."""

    return torch.eye(len(corpus.vocab))  # identity matrix

def embeddings_lexicon(corpus: TaggedCorpus, file: Path,
                       newvocab: Optional[Integerizer[Word]] = None) -> torch.Tensor:
    """Return a matrix with as many rows as corpus.vocab, where 
    each row specifies a vector embedding of the corresponding word. But first
    replace corpus.vocab with a larger vocabulary that includes all words in file
    (except that if newvocab is provided, only words in newvocab are added).
    
    The second argument is a lexicon file in the format of Homeworks 2 and 3,
    which is used to look up the word embeddings.

    The lexicon entries BOS, EOS, OOV, and OOL will be treated appropriately if
    present.  In particular, any words in corpus.vocab that are not in the
    lexicon will get the embedding of OOL (or 0 if there is no such embedding).
    """

    vocab = corpus.vocab

    with open(file) as f:
        filerows, cols = [int(i) for i in next(f).split()]   # first line gives num of rows and cols
        words = list(vocab)                      # words that need embeddings
        embeddings: List[Optional[Tensor]] = [None] * len(vocab)   # their embeddings
        seen: Set[int] = set()                   # the words we've found embeddings for
        ool_vector = torch.zeros(cols)           # use this for other words if there is no OOL entry
        specials = {'BOS': BOS_WORD, 'EOS': EOS_WORD, 'OOV': OOV_WORD}

        # Run through the words in the lexicon.
        for line in f:
            first, *rest = line.strip().split("\t")
            word = Word(first)
            vector = torch.tensor([float(v) for v in rest])
            assert len(vector) == cols     # check that the file didn't lie about # of cols

            if word == 'OOL':
                assert word not in vocab   # make sure there's not an actual word "OOL"
                ool_vector = vector
            else:
                if word in specials:    # map the special word names that may appear in lexicon
                    word = specials[word]    
                w = vocab.index(word)             # vocab integer to use as row number
                if w is None:
                    if newvocab is None or word in newvocab:  # filter by newvocab if it exists
                        words.append(word)
                        embeddings.append(vector)     # new word needs new row
                else:
                    embeddings[w] = vector        # existing word goes into existing row (replacing None)

    # Fill in OOL for any other old vocab entries that were never seen in the lexicon.
    ool_words = 0
    for w in range(len(vocab)):
        if embeddings[w] is None:
            embeddings[w] = ool_vector
            ool_words += 1

    log.info(f"From {file.name}, got embeddings for "
             f"{len(vocab)-ool_words} of {len(vocab)} previously known types "
             f"+ {len(words)-len(vocab)} new seen types")

    # Annoyingly, we must now move EOS_WORD, BOS_WORD to the end of the vocab,
    # as required by hmm.py (which asserts that this is so).
    slice = words[len(vocab)-2:len(vocab)]
    assert slice == [EOS_WORD, BOS_WORD]
    del words[len(vocab)-2:len(vocab)]
    words.extend(slice)
    
    slice = embeddings[len(vocab)-2:len(vocab)]
    del embeddings[len(vocab)-2:len(vocab)]
    embeddings.extend(slice)
    
    # Install a new vocabulary created with EOS_WORD, BOS_WORD at the end,
    # and return the corresponding embeddings packaged up as the rows of a 2D tensor (matrix).
    corpus.vocab = Integerizer(words)
    return torch.stack(embeddings)   # type: ignore  (none of the embeddings are None anymore)

def problex_lexicon(corpus: TaggedCorpus) -> torch.Tensor:
    """Return a feature matrix with as many rows as corpus.vocab, where each
    row represents a feature vector for the corresponding word w.
    There is one feature for each tag t in corpus.tagset,
    with value log(p(t|w)).  Finally, there is a feature with
    value log(p(w)).  These probabilities are add-one-smoothing
    estimates."""
    
        k = len(corpus.tagset)  # number of tags
        V = len(corpus.vocab)   # number of words
        
        # Count occurrences: word_counts[w] and tag_word_counts[t, w]
        word_counts = torch.zeros(V)
        tag_word_counts = torch.zeros(k, V)
        
        for sentence in corpus:
            for word, tag in sentence[1:-1]:  # exclude BOS, EOS
                w = corpus.vocab.index(word)
                if w is not None and w < V:
                    word_counts[w] += 1
                    if tag is not None:
                        t = corpus.tagset.index(tag)
                        if t is not None:
                            tag_word_counts[t, w] += 1
        
        # Compute p(w) with add-1 smoothing
        total_words = word_counts.sum()
        p_w = (word_counts + 1) / (total_words + V)
        log_p_w = torch.log(p_w)
        
        # Compute p(t|w) = count(t,w) / count(w) with add-1 smoothing
        # p(t|w) = (count(t,w) + 1) / (count(w) + k)
        log_p_t_given_w = torch.zeros(V, k)
        for w in range(V):
            for t in range(k):
                p_t_given_w = (tag_word_counts[t, w] + 1) / (word_counts[w] + k)
                log_p_t_given_w[w, t] = torch.log(p_t_given_w)
        
        # Concatenate: [log p(t1|w), log p(t2|w), ..., log p(tk|w), log p(w)]
        return torch.cat([log_p_t_given_w, log_p_w.unsqueeze(1)], dim=1)

def affixes_lexicon(corpus: TaggedCorpus,
                    newvocab: Optional[Integerizer[Word]] = None) -> torch.Tensor:
    """Return a feature matrix with as many rows as corpus.vocab, where each
    row represents a feature vector for the corresponding word w.
    Each row has binary features for common suffixes and affixes that the
    word has."""
    
    # If you implement this, you should add the words in newvocab
    # to corpus.vocab so that you can provide affix features for them.

    raise NotImplementedError   # you fill this in!

# Other feature templates could be added, such as word shape.
