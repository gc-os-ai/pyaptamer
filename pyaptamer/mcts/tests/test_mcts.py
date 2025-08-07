"""Test suite for the Monte Carlo Tree Search (MCTS) algorithm."""

__author__ = ["nennomp"]

import numpy as np
import pytest
import torch
import torch.nn as nn

from pyaptamer.experiments import Aptamer
from pyaptamer.mcts import MCTS
from pyaptamer.mcts._algorithm import TreeNode

NUCLEOTIDES = ["A_", "_A", "C_", "_C", "G_", "_G", "U_", "_U"]


@pytest.fixture
def root():
    return TreeNode()


@pytest.fixture
def val():
    return "A_"


@pytest.fixture
def val_not_found():
    return "X_"


@pytest.fixture
def tree_with_children(tree_node):
    """Provide a TreeNode with some children already added."""
    for val in ["A_", "C_", "G_"]:
        tree_node.create_child(val=val)
    return tree_node


class TestTreeNode:
    """Tests for the TreeNode() class."""

    def test_init(self, root, val):
        """Check correct initialization."""
        # check node initialization (passing parameters)
        node2 = TreeNode(
            val=val,
            parent=root,
            depth=1,
            is_root=False,
            is_terminal=True,
            exploitation_score=0.1,
        )
        assert node2.val == val
        assert node2.parent == root
        assert node2.depth == 1
        assert node2.n_states == 8
        assert node2.is_root is False
        assert node2.is_terminal is True
        assert node2.exploitation_score == 0.1
        assert node2.n_visits == 1
        assert len(node2.children) == 0

    def test_is_fully_expanded(self, root):
        """Check that (not) fully-expended nodes are properly tracked."""
        assert not root.is_fully_expanded()

        # add all possible children
        for val in NUCLEOTIDES:
            root.create_child(val=val)
        assert root.is_fully_expanded()

    def test_uct_score(self, root, val):
        """Test UCT score calculation."""
        child = root.create_child(val=val, is_terminal=True)

        # check whether the correct value is returned
        child.backpropagate(score=0.5)
        new_uct_score = child.uct_score()
        assert new_uct_score == pytest.approx(0.6663, rel=1e-4)

    def test_uct_score_parent_none(self, root):
        """Test UCT score calculation when parent is None, should return inf."""
        uct = root.uct_score()
        assert uct == float("inf")

    def test_get_child(self, root, val):
        """Check whether a child node is properly retrieved."""
        child = root.create_child(val=val, is_terminal=True)
        assert child == root.get_child(val)

    def test_get_child_fail(self, root, val_not_found):
        """
        Check whether a KeyError exception is raised when trying to retrieve a child
        that does not exist.
        """
        with pytest.raises(KeyError) as exc_info:
            root.get_child(val_not_found)

        assert val_not_found in str(exc_info.value)

    def test_get_best_child(self, root):
        """Check retrieving best child based on UCT."""
        child1 = root.create_child(val="A_", is_terminal=True)
        child2 = root.create_child(val="C_", is_terminal=True)

        child1.backpropagate(20.0)
        child2.backpropagate(5.0)

        # best child should be the one with higher score
        best_child = max([child1, child2], key=lambda root: root.uct_score())
        assert root.get_best_child() == best_child

    def test_get_best_child_ties(self, root):
        """Check retrieving best child based on UCT in presence of ties."""
        child1 = root.create_child(val="A_", is_terminal=True)
        child2 = root.create_child(val="C_", is_terminal=True)
        child3 = root.create_child(val="_C", is_terminal=True)

        # give the same score to two children
        child1.backpropagate(20.0)
        child2.backpropagate(5.0)
        child3.backpropagate(20.0)

        # best child should be randomly selected among those with highest score
        assert root.get_best_child() in [child1, child3]

    @pytest.mark.parametrize("val", NUCLEOTIDES)
    def test_create_child_all_nucleotides(self, root, val):
        """Test child creation works for all nucleotide types."""
        child = root.create_child(val=val)
        assert child.val == val
        assert child.parent == root

    def test_create_child(self, root, val):
        """Check successful child creation."""
        child = root.create_child(val=val, is_terminal=True)
        assert child is not None
        assert child.val == val
        assert child.parent == root
        assert val in root.children
        assert root.children[val] == child

    def test_create_child_already_exists(self, root, val):
        """Check that attempting to create a child that already exists returns it."""
        child1 = root.create_child(val=val, is_terminal=True)
        child2 = root.create_child(val=val, is_terminal=True)
        # should be the same object
        assert child1 == child2

    def test_backpropagate(self, root):
        """Check (exploitation) score backpropagation."""
        child1 = root.create_child(val="A_")
        child2 = child1.create_child(val="_C", is_terminal=True)

        # backpropagate from leaf
        child2.backpropagate(score=10.0)

        # check that visits are updated
        assert root.n_visits == 2
        assert child1.n_visits == 2
        assert child2.n_visits == 2
        # check that (exploitation) scores are updated
        assert root.exploitation_score == 0.0
        assert child1.exploitation_score == 10.0
        assert child2.exploitation_score == 10.0


class MockModel(nn.Module):
    def __init__(self, fixed_score=0.5):
        super().__init__()
        self.fixed_score = fixed_score

    def forward(self, x_apta, x_prot):
        # return a fixed score for deterministic testing
        return torch.tensor([self.fixed_score])

    def eval(self):
        pass


class MockExperiment(Aptamer):
    def __init__(
        self,
        target_encoded,
        target,
        model,
        device,
        fixed_score=0.5,
    ):
        super().__init__(target_encoded, target, model, device)
        self.fixed_score = fixed_score

    def evaluate(self, aptamer_candidate):
        # return a fixed score for deterministic testing
        return torch.tensor([self.fixed_score])


@pytest.fixture
def mcts():
    target_encoded = torch.randn(1, 20)
    mock_model = MockModel()
    device = torch.device("cpu")

    experiment = MockExperiment(
        target_encoded=target_encoded, target="ACGU", model=mock_model, device=device
    )
    mcts = MCTS(
        experiment=experiment,
        states=NUCLEOTIDES,
        depth=5,
        n_iterations=10,
    )
    return mcts


class TestMCTS:
    """Tests for the MCTS() class."""

    def test_reset(self, mcts):
        """Check correct reset of the inner state."""
        # modify its inner state
        mcts.base = "ACGU"
        mcts.candidate = "AUGCC"
        mcts.root.create_child(val="A_")

        # check that reset works
        mcts._reset()
        assert mcts.base == ""
        assert mcts.candidate == ""
        assert len(mcts.root.children) == 0

    def test_selection_not_fully_expanded(self, mcts):
        """Check selection step when the node is not fully expanded."""
        # from root with no childrem, should return the root itself
        selected = mcts._selection(node=mcts.root)
        assert selected == mcts.root

        child1 = mcts.root.create_child(val="A_", is_terminal=True)

        # should expand `child1` since it's not fully expanded
        selected = mcts._selection(node=child1)
        assert selected == child1

    def test_selection_fully_expanded(self, mcts):
        """Check selection step when the node is fully expanded."""
        for val in NUCLEOTIDES:
            child = mcts.root.create_child(val=val)
            child.backpropagate(np.random.rand())

        expected = max(
            [mcts.root.children[val] for val in NUCLEOTIDES],
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
        assert expanded.val in NUCLEOTIDES

    def test_simulation(self, mcts):
        """Check simulation step runs without errors."""
        node = mcts.root.create_child(val="A_")
        _ = mcts._simulation(node=node)
        assert True

    def test_find_best_subsequence(self, mcts):
        """Check whether the best subsequence is returned based on UCT scores."""
        node = mcts.root.create_child(val="A_")
        # create two paths
        child1 = node.create_child(val="C_", is_terminal=True)
        child11 = child1.create_child(val="G_", is_terminal=True)
        child2 = node.create_child(val="U_", is_terminal=True)
        child21 = child2.create_child(val="C_", is_terminal=True)

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
        assert "sequence" in candidate
        assert "score" in candidate
        assert len(candidate["candidate"]) == 5
        # length of sequence should be 2 * 5 (i.e., 2 * 5) as it still contains
        # the underscores
        assert len(candidate["sequence"]) == 10
