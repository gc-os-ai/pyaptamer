"""Sampler for profile HMMs"""

__author__ = ["NoorMajdoub"]
__all__ = ["ProfileHMMSampler"]
import logging

import numpy as np
import torch

from pyaptamer.raptgen.layers._utils import State, Transition, seq_to_indices

logger = logging.getLogger(__name__)


class ProfileHMMSampler:
    """
    Sampler class for generation of aptamer sequences from the dcoder hmm probabilites
    Takes the transition and emission probabilities produced by
    `DecoderPHMM` and uses them to generate concrete A/T/G/C sequences.
    Generation mechanisms are random sampling or the most likely path through the model.
    Supports scoring how well a given sequence fits the model via the
    profile HMM forward algorithm.

    Parameters
    ----------
    transition_proba : ndarray
        Per-position transition probabilities between states.
    emission_proba : ndarray
        Per-position nucleotide emission probabilities.

    proba_is_log : bool, optional, default=False
        If True, `transition_proba`/`emission_proba` are given as
        log-probabilities ( when recieved directly from the decoder).
    Attributes:
    ----------
    e : ndarray
        Emission probabilites,stored as non-log probabilities and
        exponentiated if `proba_is_log` is True.
        Always renormalized so each row sums to 1.
    a : ndarray
        Transition probabilities.stored as non-log probabilities and
        exponentiated if `proba_is_log` is True.
    """

    def __init__(self, transition_proba, emission_proba, proba_is_log=False):
        self.e = emission_proba
        self.a = transition_proba
        if proba_is_log:
            self.e = np.exp(self.e)
            self.a = np.exp(self.a)
        self.e = self.e / np.sum(self.e, axis=1)[:, None]

    def sample(self, sequence_only=False, debug=False):
        """
        Randomly samples a sequence by going over the profile HMM's states.
        Parameters
        ----------
        sequence_only : bool, optional, default=False
                    If True, return only the generated sequence string. If False,
                    also return the full list of (position, state) pairs visited.

        Returns
        -------
        states : list of (int, State), only if sequence_only=False
                    The (position, state) pair visited at each step of the walk.
        seq : str
                The generated skeleton sequence, with "_" for deletions and
                "N" for insertions.
        """
        idx, state = (0, State.M)
        states = [(idx, state)]
        seq = ""
        while True:
            if state == State.M:
                p = self.a[idx][
                    np.array(
                        [
                            Transition.M2M.value,
                            Transition.M2I.value,
                            Transition.M2D.value,
                        ]
                    )
                ]
            elif state == State.I:
                p = np.stack(
                    [
                        self.a[idx][Transition.I2M.value],
                        self.a[idx][Transition.I2I.value],
                        0,
                    ]
                )
            elif state == State.D:
                p = np.stack(
                    [
                        self.a[idx][Transition.D2M.value],
                        0,
                        self.a[idx][Transition.D2D.value],
                    ]
                )
            else:
                logger.info("something wrong")

            state = np.random.choice([State.M, State.I, State.D], p=p / sum(p))
            if state != State.I:
                idx += 1
            states.append((idx, state))
            if idx == self.a.shape[0]:
                break

            if state == State.M:
                seq += np.random.choice(list("ATGC"), p=self.e[idx - 1])
                if debug:
                    logger.info(idx, state, self.e[idx - 1], seq[-1])
            elif state == State.I:
                seq += np.random.choice(list("atgc"))
            else:
                seq += "_"
        if not sequence_only:
            return states, seq
        else:
            return seq

    def most_probable(self, sequence_only=False):
        """
        Generate a greedy step-wise sequence through the profile HMM.
        At each step , the next state (I/M/D) is chosen via argmax over the model
        transition probabilities (self.a).
        The returned sequence is the skeleton of the state to be selected but not
        the nucleotide to insert in case of insert state.
        Parameters
        ----------
        sequence_only : bool, optional, default=False
            If True, return only the generated sequence string. If False,
            also return the full list of (position, state) pairs visited.

        Returns
        -------
        states : list of (int, State), only if sequence_only=False
            The (position, state) pair visited at each step of the walk.
        seq : str
            The generated skeleton sequence, with "_" for deletions and
            "N" for insertions.

        """
        model_len = self.a.shape[0] - 1
        max_steps = 3 * model_len
        idx, state = (0, State.M)
        states = [(idx, state)]
        seq = ""

        for _ in range(max_steps):
            if state == State.M:
                p = self.a[idx][
                    np.array(
                        [
                            Transition.M2M.value,
                            Transition.M2I.value,
                            Transition.M2D.value,
                        ]
                    )
                ]
                state = State(np.argmax(p))
            elif state == State.I:
                state = State.M
            elif state == State.D:
                p = np.array(
                    [
                        self.a[idx][Transition.D2M.value],
                        0,
                        self.a[idx][Transition.D2D.value],
                    ]
                )
                state = State(np.argmax(p))
            else:
                logger.info("something wrong")

            if state != State.I:
                idx += 1
            states.append((idx, state))

            if idx == self.a.shape[0]:
                break

            if state == State.M:
                seq += "ATGC"[np.argmax(self.e[idx - 1])]
            elif state == State.I:
                seq += "N"
            else:
                seq += "_"
        else:
            raise RuntimeError(
                f"most_probable() did not terminate within {max_steps} steps"
            )

        if not sequence_only:
            return states, seq
        else:
            return seq

    def calc_seq_proba(self, seq: str):
        """
        Score a complete nucleotide sequence under this profile HMM, via
        the forward algorithm, used to chose the most probable sequence
        from a set of generated candidates.
        Parameters
        ----------
        seq : str
            A complete A/T/G/C sequence to score.

        Returns
        -------
        log_proba : Tensor
            The log-probability the model assigns to `seq`, computed via
            the profile HMM forward algorithm.
        """
        one_hot_seq = torch.tensor(seq_to_indices(seq))
        model_len = self.e.shape[0]
        random_len = len(seq)

        e = np.log(self.e)
        a = np.log(self.a)

        F = torch.ones((3, model_len + 2, random_len + 1)) * (-100)

        # init
        F[0, 0, 0] = 0

        for i in range(random_len + 1):
            for j in range(model_len + 1):
                # State M
                if j * i != 0:
                    F[State.M, j, i] = e[j - 1][one_hot_seq[i - 1]] + torch.logsumexp(
                        torch.stack(
                            (
                                a[j - 1, Transition.M2M] + F[State.M, j - 1, i - 1],
                                a[j - 1, Transition.I2M] + F[State.I, j - 1, i - 1],
                                a[j - 1, Transition.D2M] + F[State.D, j - 1, i - 1],
                            )
                        ),
                        dim=0,
                    )

                # State I
                if i != 0:
                    F[State.I, j, i] = -1.3863 + torch.logsumexp(
                        torch.stack(
                            (
                                a[j, Transition.M2I] + F[State.M, j, i - 1],
                                a[j, Transition.I2I] + F[State.I, j, i - 1],
                            )
                        ),
                        dim=0,
                    )

                # State D
                if j != 0:
                    F[State.D, j, i] = torch.logsumexp(
                        torch.stack(
                            (
                                a[j - 1, Transition.M2D] + F[State.M, j - 1, i],
                                a[j - 1, Transition.D2D] + F[State.D, j - 1, i],
                            )
                        ),
                        dim=0,
                    )

        F[State.M, model_len + 1, random_len] = torch.logsumexp(
            torch.stack(
                (
                    a[model_len, Transition.M2M] + F[State.M, model_len, random_len],
                    a[model_len, Transition.I2M] + F[State.I, model_len, random_len],
                    a[model_len, Transition.D2M] + F[State.D, model_len, random_len],
                )
            ),
            dim=0,
        )

        return F[State.M, model_len + 1, random_len]
