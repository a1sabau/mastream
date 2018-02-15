import numpy as np
import matplotlib.pyplot as plt

from .MaTree import Forest

class MaStream:
    def __init__(self, tree_no=1, tree_train_size=60, max_lvl=10, replace_rate=1, horizon=1000):
        self.horizon = horizon
        self.forest = Forest(tree_no=tree_no, tree_train_size=tree_train_size, max_lvl=max_lvl, replace_rate=replace_rate)
        self.clusters = {}
        self.initital_entries = []

    def parse_entry(self, entry_idx, entry):
        # permanently keep training the forest
        self.forest.train(entry)

        # forest does not have necesarry number of trees in order to do clustering, store the entry for later clustering
        if self.forest.status == "grow":
            self.initital_entries.append(entry)
        # forest is complete, can start clustering the initital_entries used just for training so far
        elif self.forest.status == "growComplete":
            self.initital_entries.append(entry)

            for initial_idx, initital_entry in enumerate(self.initital_entries):
                self.cluster_entry(initial_idx, initital_entry)
            # remove from memory, initialEntries are no longer needed
            #print("len initial entrues", len(self.initital_entries))
            self.initital_entries = None
        # forest is evolving, at this point each entry can be clustered as it is received
        elif (self.forest.status == "evolve" or self.forest.status == "evolveComplete"):
            #print("cluster entry evolve")
            self.cluster_entry(entry_idx, entry)

        # if horizon condition is met, attempt to merge the current clusters
        if (entry_idx > 0 and (entry_idx + 1) % self.horizon == 0):
            self.merge_clusters()

    def cluster_entry(self, entry_idx, entry):
        # get nodes (areas) where entry is positioned within each tree, get entry mass
        nodes, mass = self.forest.get_mass(entry)

        # there are 3 situations regarding found partitions/nodes, one for each tree:
        # a) no node has cluster id, a new cluster will be created
        # b) one node has cluster id, the other nodes will have this cluster id
        # c) multiple nodes have cluster id, a cluster pair needs to be formed
        cluster_ids = []
        for node in nodes:
            if node.cluster_id != None:
                cluster_ids.append(node.cluster_id)
        cluster_ids = list(set(cluster_ids))

        # case a), create new cluster
        if len(cluster_ids) == 0:
            cluster = Cluster()
            self.clusters[cluster.id] = cluster
            for node in nodes:
                node.cluster_id = cluster.id
        # case b), one node has a cluster_id
        elif len(cluster_ids) == 1:
            cluster = self.clusters[cluster_ids[0]]
            for node in nodes:
                node.cluster_id = cluster.id
        # case c), multiple nodes have cluster_id, add merge pairs to first cluster
        else:
            cluster = self.clusters[cluster_ids[0]]
            cluster.add_merge_ids(cluster_ids)
            for node in nodes:
                node.cluster_id = cluster.id

        # assign new entry to selected cluster
        cluster.add_entry_idx(entry_idx)

    def merge_clusters(self):
        # do actual merging
        for key in self.clusters:
            cluster = self.clusters[key]
            cluster.entry_idxs = self.get_all_idxs(cluster)

        # do not remove empty clusters, just reset them
        to_be_del_keys = []
        for key in self.clusters:
            if len(self.clusters[key].entry_idxs) == 0:
                to_be_del_keys.append(key)
        for key in to_be_del_keys:
            if key in self.clusters:
                self.clusters[key].reset()

    def get_all_idxs(self, cluster):
        idxs = cluster.entry_idxs
        merge_ids = cluster.merge_ids
        cluster.reset()

        for merge_id in merge_ids:
            idxs.extend(self.get_all_idxs(self.clusters[merge_id]))

        return list(set(idxs))

    def get_labels(self):
        # calculate total number of labels
        label_total = 0
        for key in self.clusters:
            cluster = self.clusters[key]
            label_total += len(cluster.entry_idxs)

        label_total += 1

        # create and populate the labels np.array
        labels = np.empty(label_total)
        labels.fill(-1)
        label_no = 0
        for key in self.clusters:
            cluster = self.clusters[key]
            if len(cluster.entry_idxs) > 0:
                labels[cluster.entry_idxs] = label_no
                label_no += 1

        return labels

    def plot_full_solution(self):
        plt.subplots_adjust(hspace=0., wspace=0.)

        # plot tree graphs
        row_no = max(len(self.forest.trees) * 2, 2)
        col_no = 2
        self.forest.plot_trees(row_no=row_no, col_no=col_no)

        # plot clusters
        row_no = 2
        col_no = 2
        plt_idx = 2
        plt.subplot(row_no, col_no, plt_idx)

        #DataPlotting.instance.plotClusters(self.data, self.get_labels(), title = "", subTitle = "", showAxes = True, fixedCoord = None, showLegend = False)
        plt.legend(loc='lower right', bbox_to_anchor=(1, -0.2), ncol=5)
        # plot boundingBox
        #DataPlotting.instance.plotAbBoxes(self.get_bounding_boxes())

        # plot cluster info
        plt_idx += 2
        plt.subplot(row_no, col_no, plt_idx)
        #DataPlotting.instance.plotText(self.get_snapshot_info() + "\n\n" + self.forest.info())

        # show plot
        plt.tight_layout()
        plt.show()

    def get_snapshot_info(self):
        info = ""
        for key in self.clusters:
            info += "Cluster%s, identifiedEntries:%s, merge_ids: %s \n" % (int(key) - 1, len(self.clusters[key].entry_idxs), list(self.clusters[key].merge_ids))
        return info

    def get_bounding_boxes(self, data):
        boxes = []
        for key in self.clusters:
            boxes.append(self.clusters[key].get_bounding_box(data))
        return boxes

class Cluster:
    id = 1
    def __init__(self):
        self.entry_idxs = []
        self.id = Cluster.id
        self.merge_ids = set()
        Cluster.id += 1

    def add_entry_idx(self, idx):
        self.entry_idxs.append(idx)

    def add_entry_idxs(self, idxs):
        self.entry_idxs.extend(idxs)

    def add_merge_ids(self, ids):
        self.merge_ids = self.merge_ids | set(ids)

    def reset(self):
        self.entry_idxs = []
        self.merge_ids = set()

    def get_bounding_box(self, data):
        ag_matrix = np.vstack(data[self.entry_idxs])
        min_vals = ag_matrix.min(axis=0)
        max_vals = ag_matrix.max(axis=0)

        return (min_vals, max_vals)
