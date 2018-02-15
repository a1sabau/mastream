import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy
from networkx.drawing.nx_agraph import graphviz_layout

class Forest:
    def __init__(self, tree_no, tree_train_size, max_lvl, replace_rate):
        HdTree.max_lvl = max_lvl
        HdTree.train_size = tree_train_size 
        self.tree_no = tree_no
        self.tree_train_size = tree_train_size            
        self.trees = []
        self.tree_idx = 0
        self.status = "grow"  #grow, growComplete, evolve, evolveComplete
        
    def tree_in_training_exists(self):
        return False if (len(self.trees) == 0) else self.trees[-1].in_training        
        
    def train(self, entry):        
        # a tree in training does not exists atm, create and train one 
        if (not self.tree_in_training_exists()):  
            # only if total number is not exceeded   
            if (len(self.trees) == self.tree_no + 1):
                return       
            tree = HdTree(id = self.tree_idx)
            self.tree_idx += 1
            self.trees.append(tree)
            tree.train(entry)
        # a tree in training exists, train it (it's always the last tree in the list)
        else:
            self.trees[-1].train(entry)
            
        self.update_status()
        
    def update_status(self):
        is_complete = len(self.trees) == self.tree_no and not self.tree_in_training_exists()        
        if (self.status == "grow"):
            if (is_complete):
                print("tree complete")
                self.status = "growComplete"
        else:
            if (is_complete):
                self.status = "evolveComplete"
            else:
                self.status = "evolve"
    
    def get_mass(self, entry):
        mass_full = 0
        nodes = []
        for tree in self.trees:
            # only used fully trained trees
            if (tree.in_training): continue
            
            node, mass = tree.get_mass(entry)
            if (node != None):
                nodes.append(node)
            mass_full += mass
            
        return (nodes, mass_full/len(self.trees))
    
    def plot_trees(self, row_no = None, col_no = 2, plt_idx = 1, show = False):
        if (row_no is None):
            row_no = len(self.trees) * 2
        col_no = 2
        plt_idx = 1
        for tree in self.trees:
            # plot graph
            plt.subplot(row_no, col_no, plt_idx)
            plt_idx += 2         
            edge_list = tree.gen_edge_list()
            g = nx.DiGraph(edge_list)
            pos = graphviz_layout(g, prog='dot')
            nx.draw(g, pos, arrows=False)
            
            # plot text
            plt.subplot(row_no, col_no, plt_idx)
            plt_idx += 2        
            plot_ref = plt.gca()
            plot_ref.get_xaxis().set_visible(False)
            plot_ref.axes.get_yaxis().set_visible(False)
            plt.setp(plot_ref, frame_on=False)        
            info = tree.info()
            plt.text(0, 1, info, horizontalalignment='left', verticalalignment='top')                
        
        # show plot
        if (show):
            plt.show()
            
    def info(self):
        info = ""
        for tree in self.trees:
            info += "T %s, ws: %s" % (tree.id, tree.ws.dim_intervals) + "\n"
        return info
        



class HdTree:
    max_lvl = 1
    train_size = 10
        
    def __init__(self, id):
        self.id = id
        self.data = []
        self.in_training = True
        self.ws = None
        self.root = None
        
    def train(self, entry):
        self.data.append(entry)
        if (len(self.data) == HdTree.train_size):
            self.in_training = False
            self.ws = WorkingSpace(self.data, id_prefix = id)
            self.root = self.expand(self.data, dim_idx = 0,  lvl = 0)
            self.ws.reset()
        
    def expand(self, data, dim_idx, lvl):
        # construct current node
        node = HdNode(id = self.ws.get_next_id())
        node.size = len(data)        
        node.lvl = lvl
        
        # keep building branches until max lvl is reached by spliting idxs for left and right node until both left and right node have elements        
        while (lvl < HdTree.max_lvl): # by not multiplying with self.ws.dim_no, we get more granular control over cluster merging            
            # split idxs for left and right node                          
            split_val = self.ws.get_split_val(dim_idx)
            
            left_data, right_data = self.split_on(dim_idx, split_val, data)
            
            # if we don't create empty nodes (size = 0), too much merger is going on
            # construct subnodes
            node.lvl = lvl
            node.dim_idx = dim_idx
            node.split_val = split_val                
                            
            next_dim_idx = self.ws.get_next_dim_idx(dim_idx)
            
            max_val = self.ws.dim_intervals[dim_idx][1]
            self.ws.dim_intervals[dim_idx][1] = split_val
            node.left = self.expand(left_data, next_dim_idx, lvl + 1)
            self.ws.dim_intervals[dim_idx][1] = max_val
            
            min_val = self.ws.dim_intervals[dim_idx][0]
            self.ws.dim_intervals[dim_idx][0] = split_val
            node.right = self.expand(right_data, next_dim_idx, lvl + 1)
            self.ws.dim_intervals[dim_idx][0] = min_val
            break        
            
        # return constructed node
        return node
        
        
    def split_on(self, dim_idx, split_val, data):
        left_data = []
        right_data= []
        
        for entry in data:
            if (entry[dim_idx] < split_val):
                left_data.append(entry)
            else:
                right_data.append(entry)
                        
        return (left_data, right_data)
        
        
        
        
    def get_mass(self, entry, node = None):
        # 1st recursion, init node and check if entry is inside working space                
        if (node == None):
            node = self.root
            if (self.ws.contains(entry) is False):
                return (None, 0)
            
        # this is a split node, follow path accordingly
        if (node.dim_idx is not None):            
            if (entry[node.dim_idx] < node.split_val):
                return self.get_mass(entry, node.left)
            else:
                return self.get_mass(entry, node.right)
        # node is leaf, recursion stops        
        else:    
            return (node, node.size)

    def gen_edge_list(self, node = None):
        if (node == None):
            node = self.root
            
        edge_list = []            
        if (node.left):
            edge_list.append((node.id, node.left.id))
            edge_list.extend(self.gen_edge_list(node.left))            
        if (node.right):
            edge_list.append((node.id, node.right.id))
            edge_list.extend(self.gen_edge_list(node.right))            
        return edge_list
    
    def plot(self):
        # plot graph
        plt.subplot(121)         
        edge_list = self.gen_edge_list()
        g = nx.DiGraph(edge_list)
        pos = graphviz_layout(g, prog='dot')
        nx.draw(g, pos, arrows=False)
        
        # plot text
        plt.subplot(122)        
        plot_ref = plt.gca()
        plot_ref.get_xaxis().set_visible(False)
        plot_ref.axes.get_yaxis().set_visible(False)
        plt.setp(plot_ref, frame_on=False)        
        info = self.info()
        plt.text(0, 1, info, horizontalalignment='left', verticalalignment='top')                
        
        # show plot
        plt.show()
        
    def info(self, node = None):
        if (node == None):
            node = self.root
        
        info = node.info() + "\n" 
        #"node: %s, dim_idx: %s, split_val: %s, size: %s, lvl: %s \n" % (node.id, node.dim_idx, node.split_val, node.size, node.lvl)        
        if (node.left):
            info += self.info(node.left)
        if (node.right):
            info += self.info(node.right)
            
        return info

class HdNode:       
    max_lvl = 4
    
    def __init__(self, id):
        # node related
        self.id = id
        self.lvl = None
        self.size = None
        self.dim_idx = None
        self.split_val = None
        self.left = None
        self.right = None
        
        # cluster related
        self.cluster_id = None
                
    def info(self):
        return "node: %s, dim_idx: %s, split_val: %s, size: %s, lvl: %s" % (self.id, self.dim_idx, self.split_val, self.size, self.lvl)    
        
        
class WorkingSpace:
    def __init__(self, data, id_prefix):
        self.id = 0
        self.id_prefix = id_prefix 
        self.dims = np.arange(len(data[0]))
        self.dim_no = len(self.dims)                       
        self.dim_intervals = self.gen_dim_intervals(data)
        self.orig_dim_intervals = deepcopy(self.dim_intervals)      

    def gen_dim_intervals(self, data):
        space_matrix = np.vstack(data)       
        min_vals = space_matrix.min(axis=0)
        max_vals = space_matrix.max(axis=0)
        
        dim_intervals = []
        for dim_idx in range(self.dim_no):
            # too much fluctuation, merges too much the clustering process
            split_val = random.uniform(min_vals[dim_idx], max_vals[dim_idx])

            range_val = max( (split_val - min_vals[dim_idx]), (max_vals[dim_idx] - split_val) )
            dim_intervals.append([split_val - range_val, split_val + range_val])
                    
        return dim_intervals
    
    def get_next_dim_idx(self, dim_idx):
        dim_idx += 1
        if (dim_idx >= len(self.dims)):
            dim_idx = 0
            
        return dim_idx
    
    def get_next_id(self):
        self.id += 1
        return "%s.%s" % (self.id_prefix, self.id)
    
    def contains(self, entry):
        for dim_idx in self.dims:
            # entry has value for current dimension outside tree workingspace
            if (entry[dim_idx] < self.dim_intervals[dim_idx][0] or entry[dim_idx] > self.dim_intervals[dim_idx][1]):
                return False
            
        # all values for all dims are within tree workingspace
        return True
    
    def get_split_val(self, dim_idx):
        split_val = self.dim_intervals[dim_idx][0] + (self.dim_intervals[dim_idx][1] - self.dim_intervals[dim_idx][0])/2
        return split_val
    
    def reset(self):
        self.dim_intervals = deepcopy(self.orig_dim_intervals)
    
    def info(self):
        print("dim_intervals: %s" % (self.dim_intervals))
        

class RandExtraction:
    def __init__(self, arr):
        self.orig_arr = deepcopy(arr)
        self.arr = deepcopy(arr)
        
    def extract(self, size):        
        if (len(self.arr) < size):
            self.arr = deepcopy(self.orig_arr)
        
        extracted = np.random.choice(self.arr, size = size, replace = False)
        extracted_idx = [np.where(self.arr == i) for i in extracted]                
        self.arr = np.delete(self.arr, extracted_idx)
            
        return extracted

class RoundRobinExtraction:
    def __init__(self, arr):
        self.arr = deepcopy(arr)
        self.idx = 0
        
    def extract(self):
        extracted = self.arr[self.idx]
        self.idx += 1
        if (self.idx >= len(self.arr)):
            self.idx = 0
        return extracted
            
        
        