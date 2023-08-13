from sklearn.decomposition import PCA
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import math

class Visualizer:

    def __init__(self, input_data: np.ndarray, compress_method: str = 'pca', compress_model=None, number_of_dimensions: int = 2, label_list: list = None, color_list: list = None):
        self.input_data = np.array(input_data)
        self.compress_method = compress_method
        self.number_of_dimensions = number_of_dimensions
        self.label_list = label_list
        self.color_list = color_list
        self.compress_model = compress_model

        self._generate_plot_visualization()

    def _generate_plot_visualization(self):

        if self.compress_model is None:
            if self.compress_method == 'pca':
                compressor = PCA(n_components=self.number_of_dimensions)
            else:
                raise ValueError("Compressor method not supported")
        else:
            compressor = self.compress_model

        try:
            components = compressor.transform(self.input_data)
        except:
            components = compressor.fit_transform(self.input_data)
        #explained_variance = compressor.explained_variance_ratio_.sum() * 100

        if self.color_list is None:
            if self.number_of_dimensions == 2:
                fig = px.scatter_matrix(
                    components,
                    dimensions=range(self.number_of_dimensions),
                    #title=f'Total Explained Variance: {explained_variance:.2f}%',
                )
                fig.update_traces(diagonal_visible=False)

            elif self.number_of_dimensions == 3:
                fig = px.scatter_3d(
                    components,
                    x=0, y=1, z=2,
                    #title=f'Total Explained Variance: {explained_variance:.2f}%',
                )
            else:
                raise ValueError("Number of dimensions must be equal to 2 or 3")
        else:
            if self.number_of_dimensions == 2:
                red_indices = []
                blue_indices = []
                for index, color in enumerate(self.color_list):
                    if self.color_list[index] == 'red':
                        red_indices.append(index)
                    else:
                        blue_indices.append(index)

                red_components = components[red_indices]
                blue_components = components[blue_indices]

                fig_red = plt.figure()
                ax_red = fig_red.add_subplot()
                ax_red.scatter(red_components[:, 0], red_components[:, 1],
                               color=['red'] * len(red_indices))
                plt.savefig("projection_red.png")

                fig_blue = plt.figure()
                ax_blue = fig_blue.add_subplot()
                ax_blue.scatter(blue_components[:, 0], blue_components[:, 1],
                                color=['blue'] * len(blue_indices))
                plt.savefig("projection_blue.png")

                fig = plt.figure()
                ax = fig.add_subplot()
                ax.scatter(components[:, 0], components[:, 1], color=self.color_list)
                plt.savefig("projection.png")

            elif self.number_of_dimensions == 3:

                color_labels = []
                for color in self.color_list:
                    if color == 'blue':
                        color_labels.append('bonafide')
                    elif color == 'red':
                        color_labels.append('attack_glasses')
                    elif color == 'green':
                        color_labels.append('attack_print')
                    elif color == 'purple':
                        color_labels.append('attack_mannequin')
                    elif color == 'yellow':
                        color_labels.append('attack_replay')
                    elif color == 'orange':
                        color_labels.append('attack_rigid_mask')
                    elif color == 'gray':
                        color_labels.append('attack_flexible_mask')
                    elif color == 'black':
                        color_labels.append('attack_paper_mask')
                    elif color == 'pink':
                        color_labels.append('attack_wigs')
                    elif color == 'cyan':
                        color_labels.append('attack_tattoo')
                    elif color == 'magenta':
                        color_labels.append('attack_makeup')

                fig = px.scatter_3d(
                    components,
                    x=0, y=1, z=2,
                    color=color_labels,
                    #title=f'Total Explained Variance: {explained_variance:.2f}%',
                )
            else:
                raise ValueError("Number of dimensions must be equal to 2 or 3")
        try:
            fig.show()
        except:
            plt.show()
