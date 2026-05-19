import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from shapely.plotting import plot_polygon
from shapely.geometry import Point
from mesh_generation.stats import plot_mesh_pdf, update_mesh_pdf


class Graphic:
    def __init__(self, anim: bool, ax, func):
        self.ax = ax
        self.anim = anim
        if anim:
            self.func = lambda setup, idx: func(setup=setup, ax=ax, idx=idx)
        else:
            self.func = lambda: func(ax=ax)



class AnimationHandler:
    def __init__(self, n_bodies=1, width=6, height=4, dpi=100, fps=30, plots=None, polygon=None):
        self.n_bodies = n_bodies
        self.dpi = dpi
        self.fps = fps
        self.polygon = polygon
        self.width = width
        self.height = height
        self.scatter = None
        self.sol = np.empty((0, n_bodies, 2))
        self.out = "/animation.mp4"

        self.plots = plots # list - allows "pdf-anim", "pdf-final", "max-vel" (max 3 allowed)
        self.graphics = []
        self.plot_interval = int(self.fps / 2)
        self.primary_ax = self.configure_plot()
        self.mesh_pdf_bars = None
        self.approx_dist = max(self.width, self.height) / np.sqrt(n_bodies) # rough approximate of ideal distance - used for plotting


    def configure_plot(self):
        if self.plots:
            self.fig, self.ax = plt.subplots(2, 2, dpi=self.dpi)
            primary_ax = self.ax[0,0]
            primary_ax.set_aspect('equal')
            self.fig.set_size_inches(self.width * 2, self.height * 2)

            positions = [[0, 1], [1, 0], [1, 1]] 
            if 'pdf-anim' in self.plots:
                self.graphics.append(Graphic(anim=True, func=self.update_pdf_anim_graphic, ax=self.ax[positions[0][0], positions[0][1]]))
                positions = positions[1:]
            if 'pdf-final' in self.plots:
                self.graphics.append(Graphic(anim=False, func=self.setup_pdf_final_graphic, ax=self.ax[positions[0][0], positions[0][1]]))
                positions = positions[1:]
            if 'max-vel' in self.plots:
                self.graphics.append(Graphic(anim=False, func=self.setup_max_vel_graphic, ax=self.ax[positions[0][0], positions[0][1]]))
                positions = positions[1:]

        else:
            self.fig, self.ax = plt.subplots(1, 1, dpi=self.dpi)
            primary_ax = self.ax
            self.fig.set_size_inches(self.width, self.height)
            self.ax.set_aspect('equal')
        
        primary_ax.set_xlim(self.polygon.bounds[0] - 0.1, self.polygon.bounds[2] + 0.1)
        primary_ax.set_ylim(self.polygon.bounds[1] - 0.1, self.polygon.bounds[3] + 0.1)
        primary_ax.set_title(f'{self.n_bodies}-Body Simulation')

        if self.polygon is not None:
            plot_polygon(self.polygon, ax=primary_ax, color='lightgray', edgecolor='black')

        return primary_ax
    

    def update_pdf_anim_graphic(self, setup: bool, ax, idx=None):
        if setup:
            ax.set_title("Mesh PDF Evolution")
            ax.set_xlabel("Distance")
            ax.set_ylabel("Probability")
            mesh_pdf_bars = plot_mesh_pdf(
                mesh_points=self.sol[0, :self.n_bodies, :], 
                approx_step=self.approx_dist, 
                ax=ax, 
                color='lightblue', 
                chart_details=False
            )
            ax.set_ylim(0, 1)
            self.mesh_pdf_bars = mesh_pdf_bars
        else:
            update_mesh_pdf(bars=self.mesh_pdf_bars, mesh_points=self.sol[idx, :self.n_bodies, :], approx_step=self.approx_dist)
                

    def setup_pdf_final_graphic(self, ax):
        ax.set_title("Final PDF")
        ax.set_xlabel("Distance")
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1)
        plot_mesh_pdf(self.sol[-1, :self.n_bodies, :], approx_step=self.approx_dist, ax=ax, color='lightblue', chart_details=False)


    def setup_max_vel_graphic(self, ax):
        ax.set_title("Max Velocity vs. Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Velocity")

        t = range(self.sol.shape[0])
        max_vels = np.zeros_like(t)

        for i in t:
            vel_series = self.sol[i, self.n_bodies:, :]
            vel_series = [np.linalg.norm(vel) for vel in vel_series]
            max_vel = np.max(vel_series)
            max_vels[i] = max_vel

        ax.plot(t, max_vels)


    def _animate(self, i, interval):
        positions = self.sol[i*interval, :self.n_bodies, :]
        self.scatter.set_offsets(positions)
        self.info_text.set_text((
            f"Step: {i*interval}/{len(self.sol)}, {i*interval/len(self.sol)*100:.1f}%\n"
        ))

        if self.plots and i % self.plot_interval == 0:
            for graphic in self.graphics:
                graphic.func(setup=False, idx=i)
                
        return self.scatter,


    def animate(self, solution):
        self.sol = solution
        self.ax_map = {}
        self.scatter = self.primary_ax.scatter(solution[0, :self.n_bodies, 0], solution[0, :self.n_bodies, 1], c='blue', marker='o')

        for graphic in self.graphics:
            if graphic.anim:
                graphic.func(setup=True, idx=None)
            else:
                graphic.func()
        self.graphics = [graphic for graphic in self.graphics if graphic.anim] # retains only animated graphics for later use

        self.info_text = self.primary_ax.text(
            0.95, 0.95, '', transform=self.primary_ax.transAxes, 
            ha='right', va='top', fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
        )
        
        desired_length = 20
        frames = desired_length * self.fps
        interval = int(self.sol.shape[0] / frames)

        if interval == 0: # handles edge case for small animations
            interval = 1
            frames = self.sol.shape[0]

        format = self.out.split('.')[1]
        if format == 'gif':
            writer = 'pillow'
        else:
            writer = 'ffmpeg'
        anim = FuncAnimation(self.fig, lambda i: self._animate(i, interval), frames=frames)
        anim.save(filename=self.out, writer=writer, dpi=self.dpi, fps=self.fps)
        print(f"Animation saved as {self.out}.")
