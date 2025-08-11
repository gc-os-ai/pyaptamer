"""Monte Carlo Tree Search (MCTS) for string optimization."""

__author__ = ["nennomp"]
__all__ = ["MCTS"]

import random

import numpy as np
from skbase.base import BaseObject


class MCTS(BaseObject):
    """
    MCTS algorithm implementation for string optimization, specifically for aptamr
    generation as described in aptamer generation as described in [1]_, originally
    introduced in [2]_.

    Adapted from:

    - https://github.com/PNUMLB/AptaTrans/blob/master/mcts.py
    - https://github.com/leekh7411/Apta-MCTS/blob/master/src/mcts.py

    Parameters
    ----------
    states : list[str]
        Possible values for the nodes. Underscores indicate whether the values are
        supposed to be prepended or appended to the sequence.
    depth : int, optional
        Maximum depth of the search tree, also the length of the generated sequences.
    n_iterations : int, optional
        Number of iterations per round for the MCTS algorithm.
    experiment : BaseExperiment, optional, default=None
        An instance of an experiment class definingthe goal function for the algorithm.

    Attributes
    ----------
    base : str
        Best sequence found so far.
    candidate : str
        Final candidate sequence.
    root : TreeNode
        Root node of the MCTS tree.

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
    >>> import torch
    >>> from pyaptamer.experiments import Aptamer
    >>> from pyaptamer.mcts import MCTS
    >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    >>> target = "MCKY"
    >>> target_encoded = torch.tensor([1, 0, 0, 1, 0, 1], dtype=torch.float32).to
    ... (device)
    >>> experiment = Aptamer(target_encoded, target, model, device)
    >>> mcts = MCTS(depth=10, experiment=experiment)
    >>> candidate = mcts.run()
    >>> print((candidate["candidate"], len(candidate["candidate"])))
    ('CUUUAUGUCA', 10)
    >>> print((candidate["sequence"], len(candidate["sequence"])))
    ('_GU_A__U_CU__AU_U_C_', 20)
    >>> print(candidate["score"])
    tensor([0.5000])
    """

    def __init__(
        self,
        states: list[str] | None = None,
        depth: int = 20,
        n_iterations: int = 1000,
        experiment=None,
    ) -> None:
        """
        Parameters
        ----------
        experiment : Aptamer
            An instance of the Aptamer() class specifying the goal function.
        states : list[str], optional
            A list containing possible values for the nodes. Underscores indicate
            whether the values are supposed to be prepended or appended to the sequence.
        depth : int, optional
            Maximum depth of the search tree.
        n_iterations : int, optional
            Number of iterations per round for the MCTS algorithm.
        """
        self.experiment = experiment
        self.depth = depth
        self.n_iterations = n_iterations

        super().__init__()

        if states is None:
            states = ["A_", "C_", "G_", "U_", "_A", "_C", "_G", "_U"]
        self.states = states

        self.root = TreeNode(
            n_states=len(states),
        )
        self.base = ""
        self.candidate = ""

    def _reset(self) -> None:
        """Reset the MCTS algorithm to its initial state."""
        self.root = TreeNode(
            n_states=len(self.states),
        )
        self.base = ""
        self.candidate = ""

    def _selection(self, node: "TreeNode") -> "TreeNode":
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

    def _expansion(self, node: "TreeNode") -> "TreeNode":
        """Expand the selected node.

        The selected node is expanded by creating a new child to which a randomly
        selected value is assigned. The value is randomly chosen from the set of
        unexpanded states (those that have not been added to the node's children yet).

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

        # find all unexpanded values for this node
        unexpanded = list(set(self.states) - set(node.children.keys()))

        # randomly selected one value from unexpanded ones
        val = random.choice(unexpanded)

        return node.create_child(val=val, is_terminal=is_terminal)

    def _simulation(self, node: "TreeNode") -> float:
        """
        Simulate a random playout for the node/path and generate a candidate sequence.

        Starting from the given node, a random walk is performed: random values are
        added to the sequence until the desired length (depth) is reached. The sequence
        is then evaluated leveraging the goal function defined by `self.experiment`.

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
            sequence = curr.val + sequence
            curr = curr.parent

        # prepend the `self.base` sequence
        sequence = self.base + sequence

        # fill the rest of the sequence with random possible values
        remaining_length = (self.depth * 2) - len(sequence)
        for _ in range(remaining_length):
            sequence += random.choice(self.states)

        # evaluate the candidate sequence with the goal function
        return self.experiment.evaluate(sequence)

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
            subsequence += curr.val

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
                n_states=len(self.states),
                depth=len(self.base) // 2,  # adjust depth based on current base
            )

            round_count += 1

        self.candidate = self.base
        return {
            "candidate": self.experiment.reconstruct(self.candidate)[0],
            "sequence": self.candidate,
            "score": self.experiment.evaluate(self.candidate),
        }


class TreeNode:
    """Node of the MCTS tree.

    Adapted from:
    - https://github.com/PNUMLB/AptaTrans/blob/master/mcts.py

    Parameters
    ----------
    val : str, optional
        Value for this node.
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

    Attributes
    ----------
    visits : int
        Counter tracking the umber of visits to this node.
    children : dict[str, TreeNode]
        Dictionary of child nodes indexed by value.

    Examples
    --------
    >>> from pyaptamer.mcts._algorithm import TreeNode
    >>> node = TreeNode(val="A")
    >>> child = node.create_child(val="C", is_terminal=True)
    >>> child.backpropagate(score=0.5)
    >>> print(node.uct_score())
    inf
    >>> print(child.uct_score())
    np.float64(0.6663)
    """

    def __init__(
        self,
        val: str = "",
        parent=None,
        depth: int = 0,
        n_states: int = 8,
        is_root: bool = True,
        is_terminal: bool = False,
        exploitation_score: float = 0.0,
    ) -> None:
        """
        Parameters
        ----------
        val : str, optional
            Value for this node.
        parent : TreeNode, optional
            Reference to the parent node.
        depth : int, optional
            Depth of the node in the tree.
        n_states : int, optional
            Number of possible children states.
        is_root : bool, optional
            Whether this node is the root of the tree.
        is_terminal : bool, optional
            Whether this node is the last one in the path.
        exploitation_score : float, optional
            Accumulated exploitation score from simulations.
        """
        self.val = val
        self.parent = parent
        self.depth = depth
        self.n_states = n_states
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
        return len(self.children) == self.n_states

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

    def get_child(self, val: str):
        """Retrieve the child node of the current node by value.

        Parameters
        ----------
        val : str, optional
            Value used to find the child node.

        Returns
        -------
        TreeNode
            The child node corresponding to the given value, if it exists.

        Raises
        ------
        KeyError
            If the child with the given value does not exist.
        """
        if val in self.children:
            return self.children[val]
        else:
            raise KeyError(f"Child with value {val} does not exist for this node")

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

    def create_child(self, val: str, is_terminal: bool = False):
        """
        Create a new child node with the given value. If the child already
        exists, it will return it.

        Parameters
        ----------
        val : str
            Value to assign to the newly created node.
        is_terminal : bool, optional
            Whether this child node is the last one in the path.

        Returns
        -------
        TreeNode
            The newly created (or existing) child node.
        """
        if val in self.children:
            return self.children[val]

        node = TreeNode(
            val=val,
            parent=self,
            depth=self.depth + 1,
            n_states=self.n_states,
            is_root=False,
            is_terminal=is_terminal,
        )
        self.children[val] = node
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
