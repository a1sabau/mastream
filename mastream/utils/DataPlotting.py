import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
from matplotlib import pylab
from matplotlib.patches import Rectangle
from pylab import cm
from prettytable import PrettyTable

class DataPlotting:
    # singleton
    instance = None

    def __init__(self, calibration_data):
        self.pca = None
        # compute dimension reduction if needed
        if calibration_data.shape[1] > 2:
            self.pca = decomposition.PCA(n_components=2)
            self.pca.fit(calibration_data)

        # add singleton reference
        DataPlotting.instance = self

    def dim_reduction(self, input_data):
        if self.pca is not None:
            input_data = self.pca.transform(input_data)
            return input_data

        # apply dimension reduction if needed
        if input_data.shape[1] > 2:
            self.pca = decomposition.PCA(n_components=2)
            self.pca.fit(input_data)

        return input_data

    def plot_text(self, info):
        # get reference to current plot and hide axis
        plot_ref = plt.gca()
        plot_ref.get_xaxis().set_visible(False)
        plot_ref.axes.get_yaxis().set_visible(False)
        plt.setp(plot_ref, frame_on=False)

        plt.axis([0, 1, 0, 1])
        plt.text(0, 1, info, horizontalalignment='left', verticalalignment='top', transform=plot_ref.axes.transAxes)

    def plot_clusters(self, input_data, labels, show_axes, title, sub_title, fixed_coord=None, show_legend=False):
        # generate colors for labels
        cluster_no = len(set(labels))
        color_map = cm.get_cmap("rainbow", cluster_no) #rainbow, jet
        colors = np.array([color_map(int(x)) for x in labels])

        # apply dimension reduction if needed
        input_data = self.dim_reduction(input_data)

        # resolve axis
        if fixed_coord is not None:
            min_x = fixed_coord[0]
            max_x = fixed_coord[1]
            min_y = fixed_coord[2]
            max_y = fixed_coord[3]
            plt.axis([min_x, max_x, min_y, max_y])
        else:
            min_x = input_data[:, 0].min()
            max_x = input_data[:, 0].max()
            min_y = input_data[:, 1].min()
            max_y = input_data[:, 1].max()
            plt.axis([min_x - max_x/5, max_x + max_x/5, min_y - max_y/5, max_y + max_y/5])

        # get reference to current plot and hide axis
        plot_ref = plt.gca()

        if not show_axes:
            plot_ref.get_xaxis().set_visible(False)
            plot_ref.axes.get_yaxis().set_visible(False)
            # plot text, top and bottom
            plt.text(0, 1, title, horizontalalignment='left', verticalalignment='bottom', transform=plot_ref.axes.transAxes)
            plt.text(0, 0, '\n' + sub_title, horizontalalignment='left', verticalalignment='top', transform=plot_ref.axes.transAxes)
        else:
            # plot text, top and bottom
            plt.text(0, 1, title, horizontalalignment='left', verticalalignment='bottom', transform=plot_ref.axes.transAxes)
            plt.text(0, 0, '\n\n' + sub_title, horizontalalignment='left', verticalalignment='top', transform=plot_ref.axes.transAxes)
            new_lines = sub_title.count("\n") + 1
            plt.gcf().subplots_adjust(bottom=new_lines * 0.06)

        # plot data, one scatter cmd for each cluster designating also the label
        labels = np.array(labels)
        for label_no in range(cluster_no):
            idxs = np.where(labels == label_no)
            plt.scatter(input_data[idxs][:, 0], input_data[idxs][:, 1], c=colors[idxs], s=50, label="C" + str(label_no))

        if show_legend is True:
            plt.legend(loc='lower left', bbox_to_anchor=(1.1, 0), ncol=5)

    def plot_self_cells(self, points):
        self.plot_points(points, 'x', 'red')

    def plot_ab_boxes(self, box_list):
        # get reference to current plot
        plot_ref = plt.gca()
        for box in box_list:
            x = box[0][0]
            y = box[0][1]
            w = box[1][0] - x
            h = box[1][1] - y
            plot_ref.add_patch(Rectangle((x,y), w, h, color='blue', alpha=0.2))

    def plot_points(self, points, marker, color):
        if len(points) == 0:
            return

        # apply dimension reduction if needed
        points = self.dim_reduction(points)
        # http://matplotlib.org/api/artist_api.html#matplotlib.lines.Line2D.set_marker
        if points.shape[0] > 0:
            plt.scatter(points[:, 0], points[:, 1], marker=marker, c=color)

    @staticmethod
    def plot_snapshot(temp_data, temp_labels, temp_bounding_boxes, temp_self_points, snapshot_info, fixed_coord=None, plt_idx=1):
        # start new figure
        plt.figure(num=None, figsize=(14, 6), dpi=80, facecolor='w', edgecolor='k')
        # plot points
        pylab.subplot(1, 2, 1)
        plt_idx += 1
        DataPlotting.instance.plot_clusters(temp_data, temp_labels, title="", sub_title="", show_axes=True, fixed_coord=fixed_coord, show_legend=True)
        DataPlotting.instance.plot_self_cells(temp_self_points)
        # plot Ab boundingBox
        DataPlotting.instance.plot_ab_boxes(temp_bounding_boxes)

        # plot text
        pylab.subplot(1, 2, 2)
        plt_idx += 1
        DataPlotting.instance.plot_text(snapshot_info)
        pylab.show()

class DataPrinter:
    def tabular(self, dct):
        # construct table data
        table_data = []
        alg_list = set()
        for input_data in dct:
            row_data = []
            row_data.append(input_data)
            for alg in dct[input_data]:
                row_data.append(dct[input_data][alg])
                alg_list.add(alg)
            table_data.append(row_data)
        table_data = np.array(table_data).ravel()
        col_no = len(alg_list) + 1
        table_data.shape = (len(table_data)/col_no, col_no)

        # construct table
        table = PrettyTable()
        table.add_column("Input Data", table_data[:, 0])
        table.align["Input Data"] = "l"
        col_idx = 1
        for alg in alg_list:
            table.add_column(alg, table_data[:, col_idx])
            col_idx += 1

        # print the ascii table
        print(table)
