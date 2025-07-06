# TODO: Currently, this is purely a placeholder to run a quick test on AptaTrans.
# This code needs refactoring and docstrings, and we still need to determine where
# it's best to put it.
import numpy as np
import torch

from pyaptamer.aptatrans.utils import rna2vec


class Node:
    def __init__(
        self, 
        letter: str = '', 
        parent=None, 
        root: bool = False, 
        last: bool = False, 
        depth: int = 0, 
        states: int = 8
    ) -> None:
        """
        Args:
            visits: Number of visits to be performed.
            letter: Node's letter.
            parent: Node's parent.
            states: Number of states in the node.
            children: Node's children
            root: Whether this node is the root.
            last: Whether this node is the last one.
            depth: Maximum depth.
        """
        self.exploitation_score = 0
        self.visits = 1
        self.letter = letter
        self.parent = parent
        self.states = states
        self.children = np.array([None for _ in range(self.states)])
        self.children_stat = np.zeros(self.states, dtype=bool)
        self.root = root
        self.last = last
        self.depth = depth

        self.letters =["A_", "C_", "G_", "U_", "_A", "_C", "_G", "_U"]

    def next_node(self, child: int = 0):
        """Return the next node."""
        assert self.children_stat[child] == True, 'No child in here.'
        return self.children[child]
    
    def generate_child(self, child: int = 0, last: bool = False):
        assert self.children_stat[child] == False, 'Already tree generated child at here'
        
        self.children[child] = Node(letter=self.letters[child], parent=self, last=last, depth=self.depth+1, states=self.states) #New node
        self.children_stat[child] = True #Stat = True
        
        return self.children[child]
    
    def backpropagation(self, score: float = 0.):
        self.visits += 1 # +1 to visit
        if self.root == True: # if root, then stop
            return self.exploitation_score
        
        else:
            self.exploitation_score += score #Add score to exploitation score
            return self.parent.backpropagation(score=score) #Backpropagation to parent node
    
    def UCT(self) -> float:
        return (self.exploitation_score / self.visits) + np.sqrt(np.log(self.parent.visits) / (2 * self.visits)) #UCT score
    

class MCTS:
    def __init__(
        self, 
        device,
        target_encoded, 
        target: str = '',
        depth: int = 20, 
        n_iterations: int = 1000, 
        states: int = 8, 
    ) -> None:
        # TODO: Not sure why but the algorithm fails for any depth value below 5
        # need to investigate if this is intended behaviour or not
        if depth < 5:
            print(f"Too small depth: {depth}. Must be at least 5, defaulting to such value.")
            depth = 5
        
        self.states = states
        self.root = Node(letter='', parent=None, root=True, last=False, states=self.states)
        self.depth = depth
        self.n_iterations = n_iterations
        self.target = target
        self.device = device
        self.target_encoded = target_encoded
        self.base = ''
        self.candidate = ''
        self.letters =['A_', 'C_', 'G_', 'U_', '_A', '_C', '_G', '_U']

    def _reconstruct(self, seq: str = '') -> str:
        r_seq = ""
        for i in range(0, len(seq), 2):
            if seq[i] == '_':
                r_seq = r_seq + seq[i+1]
            else:
                r_seq = seq[i] + r_seq
        return r_seq

    def make_candidate(self, classifier):
        now = self.root
        n = 0 # rounds
        
        while len(self.base) < self.depth * 2: #If now is last node, then stop
            n += 1
            print(f'Round {n}')
            for _ in range(self.n_iterations):
                now = self.select(classifier, now=now) #Select & Expand
            
            base = self.find_best_subsequence() #Find best subsequence
            self.base = base

            print(f'Best subsequence: {base}')
            print(f'Depth: {int(len(base)/2)}')
            print('=' * 80)

            self.root = Node(letter="", parent=None, root=True, last=False, states=self.states, depth=len(self.base)/2)
            now = self.root
            
        self.candidate = self.base
        
        return self.candidate
            
    def select(self, classifier, now=None):
        if now.depth == self.depth: #If last node, then stop
            return self.root
        
        next_node = 0
        #If every child is expanded, then go to best child
        if np.sum(now.children_stat) == self.states:
            best = 0
            for i in range(self.states):
                if best < now.children[i].UCT():
                    next_node = i
                    best = now.children[i].UCT()
                    
        else: #If not, then random
            next_node = np.random.randint(0, self.states)
            if now.children_stat[next_node] == False: #If selected child is not expanded, then expand and simulate
                next_node = self.expand(classifier, child=next_node, now=now)
    
                return self.root #start iteration at this node
            
        return now.next_node(child=next_node)
    
    def expand(self, classifier, child=None, now=None):
        last = False
        #If depth of this node is maximum depth -1, then next node is last
        if now.depth == (self.depth-1):
            last = True
        
        expanded_node = now.generate_child(child=child, last=last) #Expand
        
        score = self.simulate(classifier, target=expanded_node)  #Simulate
        expanded_node.backpropagation(score=score) #Backporpagation
        
        return child
    
    def simulate(self, classifier, target=None):
        now = target #Target node
        sim_seq = ''
        
        while now.root != True: #Parent's letters
            sim_seq = now.letter + sim_seq
            now = now.parent
            
        sim_seq = self.base + sim_seq
        
        for i in range((self.depth * 2) - len(sim_seq)): #Random child letters
            r = np.random.randint(0,self.states)
            sim_seq += self.letters[r]
        
        sim_seq = self._reconstruct(sim_seq)

        classifier.eval()
        with torch.no_grad():
            sim_seq = self._reconstruct(sim_seq)
            sim_seq = np.array([sim_seq])
            apta = torch.tensor(rna2vec(sim_seq), dtype=torch.int64).to(self.device)

            
            score = classifier(apta, self.target_encoded)

        return score

    def get_candidate(self) -> str:
        """Recommend a candidate."""
        return self._reconstruct(self.candidate)
    
    def find_best_subsequence(self) -> str:
        now = self.root
        base = self.base
        
        for _ in range((self.depth*2) - len(base)):
            best = 0
            next_node = 0
            for j in range(self.states):
                if now.children_stat[j] == True:
                    if best < now.children[j].UCT():
                        next_node = j
                        best = now.children[j].UCT()

            now = now.next_node(child=next_node)
            base += now.letter
            
            if np.sum(now.children_stat) == 0:
                break
                
        return base
    
    def reset(self) -> None:
        """Reset inner state."""
        self.base = ''
        self.candidate = ''
        self.root = Node(letter='', parent=None, root=True, last=False, states=self.states)