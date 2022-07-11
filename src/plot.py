import matplotlib.pyplot as plt
from seaborn import pairplot

TITLE_FONT_SIZE = 16
LABEL_FONT_SIZE = 12
FIGURE_SIZE = 15

class Plot():

    def __init__(self):
        pass

    def plot_2d(self, df, target_column:str = "label", title:str = None) -> None:

        plt.figure(figsize=((FIGURE_SIZE, FIGURE_SIZE)))

        category = df.label.unique()

        for c in category:
            plt.scatter(df.loc[df[target_column] == c, "component_1"],
                df.loc[df[target_column] == c, "component_2"],
                marker = 'o',
                alpha = 0.5,
                label=c)
        
        plt.legend()

        plt.title(f"{title} projection")
        plt.xlabel("component 1")
        plt.ylabel("component 2")
        plt.grid(b=True)
        plt.show(block=False)

        return None

    def plot_3d(self, df, target_column:str = "label", title:str = None) -> None:

        fig = plt.figure(figsize = (FIGURE_SIZE, FIGURE_SIZE))
        chart = fig.gca(projection = '3d')

        category = df.label.unique()

        for c in category:
            chart.scatter(df.loc[df[target_column] == c, "component_1"],
                df.loc[df[target_column] == c, "component_2"], 
                df.loc[df[target_column] == c, "component_3"],
                marker = 'o',
                alpha = 0.5, 
                label=c)

        chart.legend()

        plt.title(f"{title} projection")
        plt.xlabel("component 1")
        plt.ylabel("component 2")
        plt.show(block=False)

        return None

    def matrix_plot(self, df, target_column:str = "label", title:str = None) -> None:

        g = pairplot(df, hue=target_column, height=2.5)
        plt.show(block = False)

        return None