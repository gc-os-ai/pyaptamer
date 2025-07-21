"""Test suite for the Monte Carlo Tree Search (MCTS) algorithm."""

__author__ = ["nennomp"]

import numpy as np
import pytest
import torch
import torch.nn as nn

from pyaptamer.mcts.algorithm import TreeNode, MCTS

NUCLEOTIDES = ['A_', '_A', 'C_', '_C', 'G_', '_G', 'U_', '_U']


class MockModel(nn.Module):
    """Model mock to use for assigning scores to sequences within MCTS() class."""
    def __init__(self):
        super().__init__()
        
    def forward(self, x_apta: torch.Tensor, x_prot: torch.Tensor) -> torch.Tensor:
        # return a random score for testing purposes
        return torch.randn(1)
    
    def eval(self):
        pass


class TestTreeNode:
    """Tests for the TreeNode() class."""
    
    def test_init(self):
        """Check correct initialization."""
        # check node initialization (default values)
        node1 = TreeNode()
        assert node1.nucleotide == ''
        assert node1.parent is None
        assert node1.depth == 0
        assert node1.states == 8
        assert node1.is_root is True
        assert node1.is_terminal is False
        assert node1.exploitation_score == 0.0
        assert node1.n_visits == 1
        assert len(node1.children) == 0
        
        # check node initialization (passing parameters)
        node2 = TreeNode(
            nucleotide='A_', 
            parent=node1, 
            depth=1,
            is_root=False,
            is_terminal=True,
            exploitation_score=0.1,
        )
        assert node2.nucleotide == 'A_'
        assert node2.parent == node1
        assert node2.depth == 1
        assert node2.states == 8
        assert node2.is_root is False
        assert node2.is_terminal is True
        assert node2.exploitation_score == 0.1
        assert node2.n_visits == 1
        assert len(node2.children) == 0
        
        # check node initialization (passing parameters)
        node2 = TreeNode(
            nucleotide='A_', 
            parent=node1, 
            depth=1,
            is_root=False,
            is_terminal=True,
            exploitation_score=0.1,
        )
        assert node2.nucleotide == 'A_'
        assert node2.parent == node1
        assert node2.depth == 1
        assert node2.states == 8
        assert node2.is_root is False
        assert node2.is_terminal is True
        assert node2.exploitation_score == 0.1
        assert node2.n_visits == 1
        assert len(node2.children) == 0

    def test_is_fully_expanded(self):
        """Check that (not) fully-expended nodes are properly tracked."""
        node = TreeNode(is_root=True, states=8)
        assert not node.is_fully_expanded()
        
        # add all possible children
        for nucleotide in NUCLEOTIDES:
            node.create_child(nucleotide=nucleotide)
        assert node.is_fully_expanded()

    def test_uct_score(self):
        """Test UCT score calculation."""
        node = TreeNode(is_root=True, states=8)
        child = node.create_child(nucleotide='A_', is_terminal=True)

        # check whether the correct value is returned
        child.backpropagate(score=0.5)
        new_uct_score = child.uct_score()
        assert np.round(new_uct_score, 4) == 0.6663

    def test_uct_score_parent_none(self):
        """Test UCT score calculation when parent is None, should return inf."""
        root = TreeNode(is_root=True, states=8)
        uct = root.uct_score()
        assert uct == float('inf')

    def test_get_child(self):
        """Check whether a child node is properly retrieved."""
        node = TreeNode(is_root=True, states=8)
        child = node.create_child(nucleotide='A_', is_terminal=True)
        assert child == node.get_child('A_')

    def test_get_child_fail(self):
        """
        Check whether a KeyError exception is raised when trying to retrieve a child that does not exist.
        """
        node = TreeNode(is_root=True, states=8)
        with pytest.raises(
            KeyError, 
            match='Child with nucleotide A_ does not exist for this node'
        ):
            node.get_child('A_')

    def test_get_best_child(self):
        """Check retrieving best child based on UCT."""
        node = TreeNode(is_root=True, states=8)
        child1 = node.create_child(nucleotide='A_', is_terminal=True)
        child2 = node.create_child(nucleotide='C_', is_terminal=True)
        
        child1.backpropagate(20.0)
        child2.backpropagate(5.0)
        
        # best child should be the one with higher score
        best_child = max(
            [child1, child2], 
            key=lambda node: node.uct_score()
        )
        assert node.get_best_child() == best_child

    def test_get_best_child_ties(self):
        """Check retrieving best child based on UCT in presence of ties."""
        node = TreeNode(is_root=True, states=8)
        child1 = node.create_child(nucleotide='A_', is_terminal=True)
        child2 = node.create_child(nucleotide='C_', is_terminal=True)
        child3 = node.create_child(nucleotide='_C', is_terminal=True)
        
        # give the same score to two children
        child1.backpropagate(20.0)
        child2.backpropagate(5.0)
        child3.backpropagate(20.0)
        
        # best child should be randomly selected among those with highest score
        assert node.get_best_child() in [child1, child3]
    
    def test_create_child(self):
        """Check successful child creation."""
        node = TreeNode(is_root=True, states=8)
        nucleotide = 'A_'
        child = node.create_child(nucleotide=nucleotide, is_terminal=True)
        assert child is not None
        assert child.nucleotide == nucleotide
        assert child.parent == node
        assert nucleotide in node.children
        assert node.children[nucleotide] == child

    def test_create_child_already_exists(self):
        """Check that attempting to create a child that already exists returns it."""
        node = TreeNode(is_root=True, states=8)
        nucleotide = 'A_'
        child1 = node.create_child(nucleotide=nucleotide, is_terminal=True)
        child2 = node.create_child(nucleotide=nucleotide, is_terminal=True)
        # should be the same object
        assert child1 == child2

    def test_backpropagate(self):
        """Check (exploitation) score backpropagation."""
        node = TreeNode(is_root=True, states=8)
        child1 = node.create_child(nucleotide='A_')
        child2 = child1.create_child(nucleotide='_C', is_terminal=True)
        
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


class TestMCTS:
    """Tests for the MCTS() class."""
    
    @pytest.fixture
    def dummy_mcts(self) -> MCTS:
        """Setup an dummy MCTS() instance with mock components."""
        mock_model = MockModel()
        mcts = MCTS(
            device=torch.device('cpu'),
            model=mock_model,
            target_encoded=torch.randn(1, 100),
            target='ACGU',
            depth=5,
            n_iterations=10,
        )
        return mcts
    
    def test_init(self, dummy_mcts: MCTS):
        """Check correct initialization."""
        assert dummy_mcts.depth == 5
        assert dummy_mcts.n_iterations == 10
        assert dummy_mcts.states == 8
        assert dummy_mcts.target == 'ACGU'
        assert dummy_mcts.base == ''
        assert dummy_mcts.candidate == ''
        assert dummy_mcts.root.is_root is True

    def test_init_too_small_depth(self):
        """
        Check whether a ValueError exception is raised when using a too small depth 
        value (< 5).
        """
        mock_model = MockModel()
        with pytest.raises(ValueError, match='Depth is too small'):
            MCTS(
                device=torch.device('cpu'),
                model=mock_model,
                target_encoded=torch.randn(1, 100),
                target='ACGU',
                depth=1, # should raise ValueError
            )

    def test_reset(self, dummy_mcts: MCTS):
        """Check correct reset of the inner state."""
        # modify its inner state
        dummy_mcts.base = 'ACGU'
        dummy_mcts.candidate = 'AUGCC'
        dummy_mcts.root.create_child(nucleotide='A_')
        
        # check that reset works
        dummy_mcts._reset()
        assert dummy_mcts.base == ''
        assert dummy_mcts.candidate == ''
        assert len(dummy_mcts.root.children) == 0
    
    def test_reconstruct(self, dummy_mcts: MCTS):
        """Check sequence reconstruction."""
        assert dummy_mcts._reconstruct('') == ''
        assert dummy_mcts._reconstruct('A_C__G_U') == 'CAGU'
        assert dummy_mcts._reconstruct('_A_C_G_U') == 'ACGU'
        assert dummy_mcts._reconstruct('A__CC__G') == 'CACG'

    def test_selection_not_fully_expanded(self, dummy_mcts: MCTS):
        """Check selection step when the node is not fully expanded."""
        # from root with no childrem, should return the root itself
        selected = dummy_mcts._selection(node=dummy_mcts.root)
        assert selected == dummy_mcts.root
        
        child1 = dummy_mcts.root.create_child(nucleotide='A_', is_terminal=True)
        
        # should expand `child1` since it's not fully expanded
        selected = dummy_mcts._selection(node=child1)
        assert selected == child1

    def test_selection_fully_expanded(self, dummy_mcts: MCTS):
        """Check selection step when the node is fully expanded."""
        for nucleotide in NUCLEOTIDES:
            child = dummy_mcts.root.create_child(nucleotide=nucleotide)
            child.backpropagate(np.random.rand())
        
        expected = max(
            [dummy_mcts.root.children[nucleotide] for nucleotide in NUCLEOTIDES], 
            key=lambda node: node.uct_score()
        )
        selected = dummy_mcts._selection(node=dummy_mcts.root)
        assert expected == selected

    def test_expansion(self, dummy_mcts: MCTS):
        """Check expansion step."""
        # Test expansion creates a new child
        expanded = dummy_mcts._expansion(node=dummy_mcts.root)
        
        assert expanded is not None
        assert expanded.parent == dummy_mcts.root
        assert expanded.nucleotide in NUCLEOTIDES
    
    def test_simulation(self, dummy_mcts: MCTS):
        """Check simulation step."""
        node = dummy_mcts.root.create_child(nucleotide='A_')
        score = dummy_mcts._simulation(node=node)
        assert isinstance(score, float)
    
    def test_find_best_subsequence(self, dummy_mcts: MCTS):
        """Check whether the best subsequence is returned based on UCT scores."""
        node = dummy_mcts.root.create_child(nucleotide='A_')
        # create two paths
        child1 = node.create_child(nucleotide='C_', is_terminal=True)
        child11 = child1.create_child(nucleotide='G_', is_terminal=True)
        child2 = node.create_child(nucleotide='U_', is_terminal=True)
        child21 = child2.create_child(nucleotide='C_', is_terminal=True)

        # backpropagate scores
        child11.backpropagate(20.0)
        child21.backpropagate(5.0)

        # highest score path should be 'A_C_G_'
        best = dummy_mcts._find_best_subsequence()
        assert best == 'A_C_G_'

    def test_run_verbose(self, dummy_mcts: MCTS):
        """Check that a run (with verbose enabled) completes without errors."""
        candidate = dummy_mcts.run(verbose=True)
        assert isinstance(candidate, str)
        assert len(candidate) > 0