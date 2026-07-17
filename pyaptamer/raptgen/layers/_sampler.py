
import torch
import numpy as np

from pyaptamer.raptgen.layers._utils import State, Transition, one_hot_index
import logging
logger = logging.getLogger(__name__)




class ProfileHMMSampler():
    def __init__(self, transition_proba, emission_proba, proba_is_log=False):
        self.e = emission_proba
        self.a = transition_proba
        if proba_is_log:
            self.e = np.exp(self.e)
            self.a = np.exp(self.a)
        self.e = self.e / np.sum(self.e, axis=1)[:, None]

    def sample(self, sequence_only=False, debug=False):
        idx, state = (0, State.M)
        states = [(idx, state)]
        seq = ""
        while True:
            if state == State.M:
                p = self.a[idx][np.array([
                    Transition.M2M.value,
                    Transition.M2I.value,
                    Transition.M2D.value])]
            elif state == State.I:
                p = np.stack([
                    self.a[idx][Transition.I2M.value],
                    self.a[idx][Transition.I2I.value],
                    0])
            elif state == State.D:
                p = np.stack([
                    self.a[idx][Transition.D2M.value],
                    0,
                    self.a[idx][Transition.D2D.value]])
            else:
                logger.info("something wrong")

            state = np.random.choice([State.M, State.I, State.D], p=p/sum(p))
            if state != State.I:
                idx += 1
            states.append((idx, state))
            if idx == self.a.shape[0]:
                break

            if state == State.M:
                # logger.info("{:.2f}, {:.2f}, {:.2f}, {:.2f}".format(*self.e[idx-1]))

                seq += np.random.choice(list("ATGC"), p=self.e[idx-1])
                if debug:
                    logger.info(idx, state, self.e[idx-1], seq[-1])
            elif state == State.I:
                seq += np.random.choice(list("atgc"))
            else:
                seq += "_"
        if not sequence_only:
            return states, seq
        else:
            return seq

    def most_probable(self, sequence_only=False):
        idx, state = (0, State.M)
        states = [(idx, state)]
        seq = ""
        while True:
            if state == State.M:
                p = self.a[idx][np.array([
                    Transition.M2M.value,
                    Transition.M2I.value,
                    Transition.M2D.value])]
            elif state == State.I:
                p = [
                    self.a[idx][Transition.I2M.value],
                    0,
                    0]
            elif state == State.D:
                p = [
                    self.a[idx][Transition.D2M.value],
                    0,
                    self.a[idx][Transition.D2D.value]]
            else:
                logger.info("something wrong")
            p[np.argmax(p)] += 1000000
            state = np.random.choice([State.M, State.I, State.D], p=p/sum(p))
            if state != State.I:
                idx += 1
            states.append((idx, state))

            if idx == self.a.shape[0]:
                break

            if state == State.M:
                # logger.info("{:.2f}, {:.2f}, {:.2f}, {:.2f}".format(*self.e[idx-1]))
                p = np.copy(self.e[idx-1])
                p[np.argmax(p)] += 100000
                seq += np.random.choice(list("ATGC"), p=p/sum(p))
            elif state == State.I:
                seq += "N"
            else:
                seq += "_"
        if not sequence_only:
            return states, seq
        else:
            return seq

    def calc_seq_proba(self, seq: str):
        one_hot_seq = torch.tensor(one_hot_index(seq))
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
                if j*i != 0:
                    F[State.M, j, i] = e[j - 1][one_hot_seq[i - 1]] + \
                        torch.logsumexp(torch.stack((
                            a[j - 1, Transition.M2M] +
                            F[State.M, j - 1, i - 1],
                            a[j - 1, Transition.I2M] +
                            F[State.I, j - 1, i - 1],
                            a[j - 1, Transition.D2M] + F[State.D, j - 1, i - 1])), dim=0)

                # State I
                if i != 0:
                    F[State.I, j, i] = - 1.3863 + \
                        torch.logsumexp(torch.stack((
                            a[j, Transition.M2I] + F[State.M, j, i-1],
                            a[j, Transition.I2I] + F[State.I, j, i-1]
                        )), dim=0)

                # State D
                if j != 0:
                    F[State.D, j, i] = \
                        torch.logsumexp(torch.stack((
                            a[j - 1, Transition.M2D] + F[State.M, j - 1, i],
                            a[j - 1, Transition.D2D] + F[State.D, j - 1, i]
                        )), dim=0)

        F[State.M, model_len+1, random_len] = \
            torch.logsumexp(torch.stack((
                a[model_len, Transition.M2M] +
                F[State.M, model_len, random_len],
                a[model_len, Transition.I2M] +
                F[State.I, model_len, random_len],
                a[model_len, Transition.D2M] +
                F[State.D, model_len, random_len]
            )), dim=0)

        return F[State.M, model_len+1, random_len]
