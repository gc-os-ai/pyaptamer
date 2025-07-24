"""Test suite for the Monte Carlo Tree Search (MCTS) algorithm."""

__author__ = ["nennomp"]

import numpy as np
import pytest
import torch

from pyaptamer.mcts._algorithm import MCTS, TreeNode
from pyaptamer.experiment._aptamer import Aptamer

NUCLEOTIDES = ["A_", "_A", "C_", "_C", "G_", "_G", "U_", "_U"]

@pytest.fixture
def nucleotide():
    return 'A_'

@pytest.fixture
def nucleotide_not_found():
    return 'X_'

@pytest.fixture
def tree_node():
    """Provide a basic TreeNode for testing."""
    return TreeNode()

@pytest.fixture
def tree_with_children(tree_node):
    """Provide a TreeNode with some children already added."""
    for nucleotide in ["A_", "C_", "G_"]:
        tree_node.create_child(nucleotide=nucleotide)
    return tree_node


class TestTreeNode:
    """Tests for the TreeNode() class."""

    def test_init(self, nucleotide):
        """Check correct initialization."""
        node1 = TreeNode()

        # check node initialization (passing parameters)
        node2 = TreeNode(
            nucleotide=nucleotide,
            parent=node1,
            depth=1,
            is_root=False,
            is_terminal=True,
            exploitation_score=0.1,
        )
        assert node2.nucleotide == nucleotide
        assert node2.parent == node1
        assert node2.depth == 1
        assert node2.states == 8
        assert node2.is_root is False
        assert node2.is_terminal is True
        assert node2.exploitation_score == 0.1
        assert node2.n_visits == 1
        assert len(node2.children) == 0

    def test_is_fully_expanded(self, tree_node):
        """Check that (not) fully-expended nodes are properly tracked."""
        assert not tree_node.is_fully_expanded()

        # add all possible children
        for nucleotide in NUCLEOTIDES:
            tree_node.create_child(nucleotide=nucleotide)
        assert tree_node.is_fully_expanded()

    def test_uct_score(self, nucleotide):
        """Test UCT score calculation."""
        node = TreeNode()
        child = node.create_child(nucleotide=nucleotide, is_terminal=True)

        # check whether the correct value is returned
        child.backpropagate(score=0.5)
        new_uct_score = child.uct_score()
        assert new_uct_score == pytest.approx(0.6663, rel=1e-4)

    def test_uct_score_parent_none(self):
        """Test UCT score calculation when parent is None, should return inf."""
        root = TreeNode()
        uct = root.uct_score()
        assert uct == float("inf")

    def test_get_child(self, nucleotide):
        """Check whether a child node is properly retrieved."""
        node = TreeNode()
        child = node.create_child(nucleotide=nucleotide, is_terminal=True)
        assert child == node.get_child(nucleotide)

    def test_get_child_fail(self, nucleotide_not_found, tree_node):
        """
        Check whether a KeyError exception is raised when trying to retrieve a child
        that does not exist.
        """
        with pytest.raises(KeyError) as exc_info:
            tree_node.get_child(nucleotide_not_found)
        
        assert nucleotide_not_found in str(exc_info.value)

    def test_get_best_child(self):
        """Check retrieving best child based on UCT."""
        node = TreeNode(is_root=True, states=8)
        child1 = node.create_child(nucleotide="A_", is_terminal=True)
        child2 = node.create_child(nucleotide="C_", is_terminal=True)

        child1.backpropagate(20.0)
        child2.backpropagate(5.0)

        # best child should be the one with higher score
        best_child = max([child1, child2], key=lambda node: node.uct_score())
        assert node.get_best_child() == best_child

    def test_get_best_child_ties(self):
        """Check retrieving best child based on UCT in presence of ties."""
        node = TreeNode()
        child1 = node.create_child(nucleotide="A_", is_terminal=True)
        child2 = node.create_child(nucleotide="C_", is_terminal=True)
        child3 = node.create_child(nucleotide="_C", is_terminal=True)

        # give the same score to two children
        child1.backpropagate(20.0)
        child2.backpropagate(5.0)
        child3.backpropagate(20.0)

        # best child should be randomly selected among those with highest score
        assert node.get_best_child() in [child1, child3]

    @pytest.mark.parametrize("nucleotide", NUCLEOTIDES)
    def test_create_child_all_nucleotides(self, nucleotide):
        """Test child creation works for all nucleotide types."""
        node = TreeNode()
        child = node.create_child(nucleotide=nucleotide)
        assert child.nucleotide == nucleotide
        assert child.parent == node

    def test_create_child(self, nucleotide):
        """Check successful child creation."""
        node = TreeNode()
        child = node.create_child(nucleotide=nucleotide, is_terminal=True)
        assert child is not None
        assert child.nucleotide == nucleotide
        assert child.parent == node
        assert nucleotide in node.children
        assert node.children[nucleotide] == child

    def test_create_child_already_exists(self, nucleotide):
        """Check that attempting to create a child that already exists returns it."""
        node = TreeNode(is_root=True, states=8)
        child1 = node.create_child(nucleotide=nucleotide, is_terminal=True)
        child2 = node.create_child(nucleotide=nucleotide, is_terminal=True)
        # should be the same object
        assert child1 == child2

    def test_backpropagate(self):
        """Check (exploitation) score backpropagation."""
        node = TreeNode(is_root=True, states=8)
        child1 = node.create_child(nucleotide="A_")
        child2 = child1.create_child(nucleotide="_C", is_terminal=True)

        # backpropagate from leaf
        child2.backpropagate(score=10.0)

        # check that visits are updated
        assert node.n_visits == 2
        assert child1.n_visits == 2
        assert child2.n_visits == 2
        # check that (exploitation) scores are updated
        assert node.exploitation_score == 0.0
        assert child1.exploitation_score == 10.0
        assert child2.exploitation_score == 10.0


class MockExperiment(Aptamer):
    def __init__(
        self, 
        target_encoded=torch.randn(1, 20),
        target="ACGU",
        model=None,
        device=torch.device("cpu"),
        fixed_score=0.5,
    ):
        super().__init__(target_encoded, target, model, device)
        self.fixed_score = fixed_score

    def run(self, aptamer_candidate):
        # return a fixed score for deterministic testing
        return torch.tensor([self.fixed_score])

@pytest.fixture
def mcts():
    experiment = MockExperiment()
    mcts = MCTS(
        experiment=experiment,
        depth=5,
        n_iterations=10,
    )
    return mcts


class TestMCTS:
    """Tests for the MCTS() class."""

    def test_init(self, mcts):
        """Check correct initialization."""
        assert isinstance(mcts.experiment, Aptamer)
        assert mcts.depth == 5
        assert mcts.n_iterations == 10

    def test_init_experiment_not_aptamer_instance(self):
        """
        Check whether a TypeError exception is raised when `experiment` is not an
        instance of the Aptamer class.
        """
        with pytest.raises(
            TypeError, 
            match="`experiment` must be an instance of class `Aptamer`."
        ):
            experiment = None
            mcts = MCTS(
                experiment=experiment,
                depth=5,
                n_iterations=10,
            )
            return mcts

    def test_reset(self, mcts):
        """Check correct reset of the inner state."""
        # modify its inner state
        mcts.base = "ACGU"
        mcts.candidate = "AUGCC"
        mcts.root.create_child(nucleotide="A_")

        # check that reset works
        mcts._reset()
        assert mcts.base == ""
        assert mcts.candidate == ""
        assert len(mcts.root.children) == 0

    def test_reconstruct(self, mcts):
        """Check sequence reconstruction."""
        assert mcts._reconstruct("") == ""
        assert mcts._reconstruct("A_C__G_U") == "CAGU"
        assert mcts._reconstruct("_A_C_G_U") == "ACGU"
        assert mcts._reconstruct("A__CC__G") == "CACG"

    def test_selection_not_fully_expanded(self, mcts):
        """Check selection step when the node is not fully expanded."""
        # from root with no childrem, should return the root itself
        selected = mcts._selection(node=mcts.root)
        assert selected == mcts.root

        child1 = mcts.root.create_child(nucleotide="A_", is_terminal=True)

        # should expand `child1` since it's not fully expanded
        selected = mcts._selection(node=child1)
        assert selected == child1

    def test_selection_fully_expanded(self, mcts):
        """Check selection step when the node is fully expanded."""
        for nucleotide in NUCLEOTIDES:
            child = mcts.root.create_child(nucleotide=nucleotide)
            child.backpropagate(np.random.rand())

        expected = max(
            [mcts.root.children[nucleotide] for nucleotide in NUCLEOTIDES],
            key=lambda node: node.uct_score(),
        )
        selected = mcts._selection(node=mcts.root)
        assert expected == selected

    def test_expansion(self, mcts):
        """Check expansion step."""
        # Test expansion creates a new child
        expanded = mcts._expansion(node=mcts.root)

        assert expanded is not None
        assert expanded.parent == mcts.root
        assert expanded.nucleotide in NUCLEOTIDES

    def test_simulation(self, mcts):
        """Check simulation step runs without errors."""
        node = mcts.root.create_child(nucleotide="A_")
        _ = mcts._simulation(node=node)
        assert True

    def test_find_best_subsequence(self, mcts):
        """Check whether the best subsequence is returned based on UCT scores."""
        node = mcts.root.create_child(nucleotide="A_")
        # create two paths
        child1 = node.create_child(nucleotide="C_", is_terminal=True)
        child11 = child1.create_child(nucleotide="G_", is_terminal=True)
        child2 = node.create_child(nucleotide="U_", is_terminal=True)
        child21 = child2.create_child(nucleotide="C_", is_terminal=True)

        # backpropagate scores
        child11.backpropagate(20.0)
        child21.backpropagate(5.0)

        # highest score path should be 'A_C_G_'
        best = mcts._find_best_subsequence()
        assert best == "A_C_G_"

    def test_run_verbose(self, mcts):
        """Check that a run (with verbose enabled) completes without errors."""
        candidate = mcts.run(verbose=True)
        assert isinstance(candidate, dict)
        assert "candidate" in candidate
        assert "score" in candidate