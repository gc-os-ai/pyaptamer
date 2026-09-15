"""Sampler for profile HMMs"""

__author__ = ["NoorMajdoub"]
__all__ = ["ProfileHMMSampler"]
import logging

import numpy as np

from pyaptamer.raptgen.layers._utils import State, Transition

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
                # logger.info("{:.2f}, {:.2f}, {:.2f}, {:.2f}".format(*self.e[idx-1]))

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
        Generate the most likely step-wise sequence through the profile HMM state.
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
                p = np.array(
                    [
                        self.a[idx][Transition.I2M.value],
                        self.a[idx][Transition.I2I.value],
                        0,
                    ]
                )
            elif state == State.D:
                p = np.array(
                    [
                        self.a[idx][Transition.D2M.value],
                        0,
                        self.a[idx][Transition.D2D.value],
                    ]
                )
            else:
                logger.info("something wrong")

            state = State(np.argmax(p))
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

        if not sequence_only:
            return states, seq
        else:
            return seq
