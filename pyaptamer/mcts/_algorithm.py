"""Monte Carlo Tree Search (MCTS) for aptamer candidate generation."""

__author__ = ["nennomp"]
__all__ = ["MCTS", "TreeNode"]

import random
from typing import Optional

import numpy as np
import torch

from pyaptamer.experiment._aptamer import Aptamer
from pyaptamer.utils.rna import rna2vec


class TreeNode:
    """Node of the MCTS tree.

    Adapted from:
    - https://github.com/PNUMLB/AptaTrans/blob/master/mcts.py

    Attributes
    ----------
    visits : int
        Counter tracking the umber of visits to this node.
    children : dict[str, TreeNode]
        Dictionary of child nodes indexed by nucleotide letter.

    Examples
    --------
    >>> from pyaptamer.mcts.algorithm import TreeNode
    >>> node = TreeNode(nucleotide="A")
    >>> child = node.create_child(nucleotide="C", is_terminal=True)
    >>> child.backpropagate(score=0.5)
    >>> print(node.uct_score())
    inf
    >>> print(child.uct_score())
    np.float64(0.6663)
    """

    def __init__(
        self,
        nucleotide: str = "",
        parent = None,
        depth: int = 0,
        states: int = 8,
        is_root: bool = True,
        is_terminal: bool = False,
        exploitation_score: float = 0.0,
    ) -> None:
        """
        Parameters
        ----------
        nucleotide : str, optional
            Nucleotide letter for this node.
        parent : TreeNode, optional
            Reference to the parent node.
        depth : int, optional
            Depth of the node in the tree.
        states : int, optional
            Number of possible children states.
        is_root : bool, optional
            Whether this node is the root of the tree.
        is_terminal : bool, optional
            Whether this node is the last one in the path.
        exploitation_score : float, optional
            Accumulated exploitation score from simulations.
        """
        self.nucleotide = nucleotide
        self.parent = parent
        self.depth = depth
        self.states = states
        self.is_root = is_root
        self.is_terminal = is_terminal
        self.exploitation_score = exploitation_score

        self.n_visits = 1
        self.children = {}

    def is_fully_expanded(self) -> bool:
        """
        Check if all possible children of this node have been created (i.e., whether
        the node is fully-expanded).

        Returns
        -------
        bool
            True if the node is fully-expanded, False otherwise.
        """
        return len(self.children) == self.states

    def uct_score(self) -> float:
        """Compute upper confidence bound applied to trees (UCT) score.

        UCT balances the trade-off between exploration (visting new paths) and
        exploitation (visting known paths).
        See:
        - https://en.wikipedia.org/wiki/Monte_Carlo_tree_search

        Returns
        -------
        float
            The UCT score for this node.
        """
        if self.parent is None:
            return float("inf")

        # exploration term
        exploration = np.sqrt(np.log(self.parent.n_visits) / (2 * self.n_visits))
        # exploitation term
        exploitation = self.exploitation_score / self.n_visits

        return exploitation + exploration

    def get_child(self, nucleotide: str):
        """Retrieve the child node of the current node by nucleotide letter.

        Parameters
        ----------
        nucleotide : str, optional
            Nucleotide letter used to find the child node.

        Returns
        -------
        TreeNode
            The child node corresponding to the given nucleotide letter, if it exists.

        Raises
        ------
        KeyError
            If the child with the given nucleotide letter does not exist.
        """
        if nucleotide in self.children:
            return self.children[nucleotide]
        else:
            raise KeyError(
                f"Child with nucleotide {nucleotide} does not exist for this node"
            )

    def get_best_child(self):
        """Select the best child based on UCT scores.

        If multiple children have the same UCT score, one of them is randomly selected.

        Returns
        -------
        TreeNode
            The child node with the highest UCT score.
        """
        best_children = []
        best_uct_score = float("-inf")
        for _, child in self.children.items():
            uct_score = child.uct_score()
            if uct_score > best_uct_score:
                best_uct_score = uct_score
                best_children = [child]
            elif uct_score == best_uct_score:
                best_children.append(child)

        # break ties randomly
        return random.choice(best_children)

    def create_child(self, nucleotide: str, is_terminal: bool = False):
        """
        Create a new child node with the given nucleotide letter. If the child already
        exists, it will return it.

        Parameters
        ----------
        nucleotide : str
            Nucleotide letter to assign to the newly created node.
        is_terminal : bool, optional
            Whether this child node is the last one in the path.

        Returns
        -------
        TreeNode
            The newly created (or existing) child node.
        """
        if nucleotide in self.children:
            return self.children[nucleotide]

        node = TreeNode(
            nucleotide=nucleotide,
            parent=self,
            depth=self.depth + 1,
            states=self.states,
            is_root=False,
            is_terminal=is_terminal,
        )
        self.children[nucleotide] = node
        return node

    def backpropagate(self, score: float) -> None:
        """Backpropagate the score up the tree to all ancestors of the current node.

        The visit count and exploitation score is updated as we traverse up the tree.

        Parameters
        ----------
        score : float
            Score to backpropagate.
        """
        curr = self
        while curr is not None:
            curr.n_visits += 1
            if not curr.is_root:
                curr.exploitation_score += score
            curr = curr.parent


class MCTS:
    """
    MCTS algorithm implementation for aptamer generation as described in [1]_,
    originally introduced in [2]_.

    Adapted from:
    - https://github.com/PNUMLB/AptaTrans/blob/master/mcts.py
    - https://github.com/leekh7411/Apta-MCTS/blob/master/src/mcts.py

    Parameters
    ----------
    root : TreeNode
        Root node of the MCTS tree.
    base : str
        Best sequence found so far.
    candidate : str
        Final candidate sequence.

    Attributes
    ----------
    nucleotides : list[str]
        Possible nucleotide letters for the nodes. Underscores indicate whether the
        nucleotide is supposed to be prepended or appended to the sequence.
    states : int
        Number of possible states (8 for 4 nucleotides with prepend/append option
        for each one).

    References
    ----------
    .. [1] Shin, Incheol, et al. "AptaTrans: a deep neural network for predicting
    aptamer-protein interaction using pretrained encoders." BMC bioinformatics 24.1
    (2023): 447.
    .. [2] Lee, Gwangho, et al. "Predicting aptamer sequences that interact with target
    proteins using an aptamer-protein interaction classifier and a Monte Carlo tree
    search approach." PloS one 16.6 (2021): e0253760.

    Examples
    --------
    >>> from pyaptamer.mcts.algorithm import MCTS
    >>> from pyaptamer.experiment import Aptamer
    >>> experiment = Aptamer(target_encoded, target, model, device)
    >>> mcts = MCTS(experiment, depth=10)
    >>> candidate = mcts.run(verbose=True)
    >>> print(candidate['candidate'])
    ACGUACGUAU
    >>> print(candidate['score'])
    0.85
    >>> print(len(candidate))
    10
    """

    nucleotides = ["A_", "_A", "C_", "_C", "G_", "_G", "U_", "_U"]
    states = 8

    def __init__(
        self,
        experiment: Aptamer,
        depth: int = 20,
        n_iterations: int = 1000,
    ) -> None:
        """
        Parameters
        ----------
        experiment : Aptamer
            An instance of the Aptamer() class specifying the goal function.
        depth : int, optional
            Maximum depth of the search tree.
        n_iterations : int, optional
            Number of iterations per round for the MCTS algorithm.

        Raises
        ------
        TypeError
            If `experiment` is not an instance of the Aptamer class.
        """
        if not isinstance(experiment, Aptamer):
            raise TypeError("`experiment` must be an instance of class `Aptamer`.")

        self.experiment = experiment
        self.depth = depth
        self.n_iterations = n_iterations

        self.root = TreeNode(
            states=self.states,
        )
        self.base = ""
        self.candidate = ""

    def _reset(self) -> None:
        """Reset the MCTS algorithm to its initial state."""
        self.root = TreeNode(
            states=self.states,
        )
        self.base = ""
        self.candidate = ""

    def _reconstruct(self, sequence: str = "") -> np.ndarray:
        """Reconstruct the actual RNA sequence from the encoded representation.

        The encoding uses pairs like 'A_' (add A to left) and '_A' (add A to right).
        This method converts these pairs back to the actual sequence.

        Parameters
        ----------
        seq : str
            Encoded sequence with direction markers (underscores).

        Returns
        -------
        np.ndarray
            The reconstructed RNA sequence as a numpy array.
        """
        result = ""
        for i in range(0, len(sequence), 2):
            match sequence[i]:
                case "_":
                    # append the next nucleotide
                    result = result + sequence[i + 1]
                case _:
                    # prepend the current nucleotide
                    result = sequence[i] + result
        
        return np.array([result])

    def _selection(self, node: TreeNode) -> TreeNode:
        """Select a node for expansion.

        The tree is traversed recursively based on the more promising nodes according
        to their UCT scores. When a node is fully expanded, the best child is selected
        and the search continues down that path. The expansion stops when a node that
        has not been fully expanded is found, or when a terminal node is reached.

        Parameters
        ----------
        node : TreeNode
            Starting node for selection.

        Returns
        -------
        TreeNode
            The node selected for expansion.
        """
        while not node.is_terminal:
            if node.is_fully_expanded():  # fully expanded, select best one and continue
                node = node.get_best_child()
            else:  # expand
                return node
        return node

    def _expansion(self, node: TreeNode) -> TreeNode:
        """Expand the selected node.

        The selected node is expanded by creating a new child to which a randomly
        selected nucleotide is assigned. The nucleotide is randomly chosen from the set
        of uenxpanded nucleotides (those that have not been added to the node's
        children yet).

        Parameters
        ----------
        node : TreeNode
            Node to expand from.

        Returns
        -------
        TreeNode
            The newly created child node from the expansion.
        """
        is_terminal = node.depth == self.depth - 1

        # find all unexpanded nucleotides for this node
        unexpanded = list(set(self.nucleotides) - set(node.children.keys()))

        # randomly selected one nucleotide from unexpanded ones
        nucleotide = random.choice(unexpanded)

        return node.create_child(nucleotide=nucleotide, is_terminal=is_terminal)

    @torch.no_grad()
    def _simulation(self, node: TreeNode) -> float:
        """Simulate a random playout for the node/path and generate a candidate aptamer.

        Starting from the given node, a random walk is performed: random nucleotides
        are added to the sequence until the desired length (depth) is reached. The
        candidate is then evaluated leveraging the classifier model for scoring.

        Parameters
        ----------
        node : TreeNode
            Node to start simulation from.

        Returns
        -------
        float
            The score for the simulate sequence, assigned according to the goal 
            function defined by `self.experiment`.
        """
        curr = node
        sequence = ""

        # build the current sequence from node to root
        while not curr.is_root:
            sequence = curr.nucleotide + sequence
            curr = curr.parent

        # prepend the `self.base` sequence
        sequence = self.base + sequence

        # fill the rest of the sequence with random nucleotides
        remaining_length = (self.depth * 2) - len(sequence)
        for _ in range(remaining_length):
            sequence += random.choice(self.nucleotides)

        # evaluate the sequence (i.e., the candidate aptamer) with the model
        aptamer_candidate = rna2vec([self._reconstruct(sequence)])

        return self.experiment.run(torch.tensor(aptamer_candidate))
        #return float(self.experiment.run(aptamer_candidate))

    def _find_best_subsequence(self) -> str:
        """Retrieve the best sequence found so far, according to the UCT scores.

        Returns
        -------
        str
            The best subsequence found in the current tree.
        """
        curr = self.root
        subsequence = self.base

        # traverse the tree
        max_steps = (self.depth * 2) - len(self.base)
        for _ in range(max_steps):
            if not curr.children:
                break

            curr = curr.get_best_child()
            subsequence += curr.nucleotide

        return subsequence

    def run(self, verbose: bool = True) -> dict:
        """
        Perform a full recommendation run consisting of `self.n_iterations` rounds of
        (selection -> expansion -> simulation -> backpropagation)

        Parameters
        ----------
        verbose : bool
            Whether to print progress information.

        Returns
        -------
        dict
            Dictionary containing the final candidate sequence (`candidate`) and its 
            score (`score`).
        """
        self._reset()

        # continue until we reach the target sequence length (i.e, depth * 2)
        round_count = 0
        while len(self.base) < self.depth * 2:
            if verbose:
                print(f"\n ----- Round: {round_count + 1} -----")

            for _ in range(self.n_iterations):
                # selection
                node = self._selection(node=self.root)

                # expansion
                if not node.is_terminal:
                    node = self._expansion(node=node)

                # simulation
                score = self._simulation(node=node)

                # backpropagation
                node.backpropagate(score)

            self.base = self._find_best_subsequence()

            if verbose:
                print("#" * 50)
                print(f"Best subsequence: {self.base}")
                print(f"Depth: {len(self.base) // 2}")
                print("#" * 50)

            # reset for next iteration
            self.root = TreeNode(
                states=self.states,
                depth=len(self.base) // 2,  # adjust depth based on current base
            )

            round_count += 1

        self.candidate = self.base
        candidate = self._reconstruct(self.candidate)
        return {
            'candidate': candidate,
            'score': self.experiment.run(torch.tensor(rna2vec(np.array([candidate]))))
        }
