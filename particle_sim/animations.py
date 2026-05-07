import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from shapely.plotting import plot_polygon
from shapely.geometry import Point
from mesh_generation.stats import plot_mesh_pdf

class AnimationHandler:
    def __init__(self, n_bodies=1, width=6, height=4, dpi=100, fps=20, polygon=None):
        self.n_bodies = n_bodies
        self.dpi = dpi
        self.fps = fps
        self.polygon = polygon
        self.width = width
        self.height = height
        self.scatter = None
        self.solution = np.empty((0, n_bodies, 2))
        self.pdfs = False
        self.pdf_interval = 100

        self.approx_dist = max(self.width, self.height) / np.sqrt(n_bodies) # rough approximate of ideal distance

        if self.width <= 2 or self.height <= 2:
            self.width *= 2
            self.height *= 2


    def configure_plot(self):
        if self.pdfs:
            self.fig, self.ax = plt.subplots(1, 2, dpi=self.dpi)
            primary_ax = self.ax[0]
            self.ax[0].set_aspect('equal')
            self.fig.set_size_inches(self.width * 2, self.height)
        else:
            self.fig, self.ax = plt.subplots(1, 1, dpi=self.dpi)
            primary_ax = self.ax
            self.fig.set_size_inches(self.width, self.height)
            self.ax.set_aspect('equal')
        
        primary_ax.set_xlim(self.polygon.bounds[0] - 1, self.polygon.bounds[2] + 1)
        primary_ax.set_ylim(self.polygon.bounds[1] - 1, self.polygon.bounds[3] + 1)
        primary_ax.set_title(f'{self.n_bodies}-Body Simulation')

        if self.polygon is not None:
            plot_polygon(self.polygon, ax=primary_ax, color='lightgray', edgecolor='black')

        return primary_ax


    def _animate(self, i):
        positions = self.solution[i, :self.n_bodies, :]
        self.scatter.set_offsets(positions)

        if self.pdfs and (i+1) % self.pdf_interval == 0:
            self.ax[1].clear()
            plot_mesh_pdf(positions, approx_step=self.approx_dist, ax=self.ax[1], color='lightblue')
        
        colors = []
        for pos in positions:
            p = Point(pos)
            colors.append('red' if not self.polygon.contains(p) else 'blue')
        self.scatter.set_color(colors)
        return self.scatter,


    def animate(self, solution, pdfs=True, pdf_interval=25):
        self.pdfs = pdfs
        self.pdf_interval = pdf_interval
        self.solution = solution
        primary_ax = self.configure_plot()

        self.scatter = primary_ax.scatter(solution[0, :self.n_bodies, 0], solution[0, :self.n_bodies, 1], c='blue', marker='o')
        if self.pdfs:
            plot_mesh_pdf(solution[0, :self.n_bodies, :], approx_step=self.approx_dist, ax=self.ax[1], color='lightblue')

        anim = FuncAnimation(self.fig, self._animate, frames=len(self.solution), interval=1000 / self.fps)
        anim.save('./animation.gif', dpi=self.dpi, fps=self.fps)
        print("Animation saved as animation.gif")
