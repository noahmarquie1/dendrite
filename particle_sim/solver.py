import numpy as np
from particle_sim.physics import PhysicsHandler
from particle_sim.animations import AnimationHandler
from scipy.integrate import RK45
import matplotlib.pyplot as plt


class PointCloudSolver:
    def __init__(self, dpi=100, width=5, height=4, n_bodies=1,
                 force_multiplier=100, drag_coefficient=0.01,
                 polygon=None):

        self.solution = np.empty((0, 2))
        self.dpi = dpi
        self.width = width
        self.height = height
        self.n_bodies = n_bodies
        self.force_multiplier = force_multiplier
        self.drag_coefficient = drag_coefficient
        self.scatter = plt.scatter(np.zeros((self.n_bodies, 1)), np.zeros((self.n_bodies, 1)), c='blue', marker='o')

        self.phys = PhysicsHandler(
            n_bodies = self.n_bodies,
            force_multiplier = self.force_multiplier,
            drag_coefficient = self.drag_coefficient,
            width = self.width,
            height = self.height,
            polygon=polygon,
        )

        self.anim = AnimationHandler(
            n_bodies=self.n_bodies,
            width=self.width,
            height=self.height,
            polygon=polygon,
            dpi=self.dpi,
        )


    def add_polygon_bounds(self, polygon):
        self.phys.polygon = polygon


    def generate_random_initial_state(self):
        rng = np.random.default_rng()
        bounds = self.phys.polygon.bounds
        pos = rng.uniform(
            [bounds[0] + 0.2, bounds[1] + 0.2], 
            [bounds[2] - 0.2, bounds[3] - 0.2], 
            size=(self.n_bodies, 2)
        )
        random_deviation = np.random.uniform(-0.1, 0.1, size=(self.n_bodies, 2))
        pos += random_deviation

        vel = np.zeros((self.n_bodies, 2))
        state = np.concatenate((pos, vel))
        return state


    def calculate_derivatives(self, phys, state):
        inter_object_force = phys.calculate_repulsive_force
        state = state.reshape(phys.n_bodies * 2, 2)

        initial_positions = state[:phys.n_bodies]
        initial_velocities = state[phys.n_bodies:]
        res = np.zeros_like(state)

        # sum forces acting on each body
        for i in range(phys.n_bodies):
            for j in range(phys.n_bodies):
                if i != j:
                    res[phys.n_bodies + i] += inter_object_force(this=initial_positions[i],
                                                                 other=initial_positions[j])
            # calculate repulsion from walls
            res[phys.n_bodies + i] += phys.calculate_wall_force(this=initial_positions[i])
            # calculate drag force
            res[phys.n_bodies + i] += phys.calculate_drag_force(vel=initial_velocities[i])

        # derivatives of each position are equal to velocity
        res[0:phys.n_bodies] = initial_velocities
        return res.flatten()


    def solve(self, state0=None, h=0.05, steps=5):
        print("Beginning simulation.")
        state0 = self.generate_random_initial_state() if state0 is None else state0
        y0 = state0.flatten()
        self.solution = np.array([state0])

        solver = RK45(
            lambda t, y: self.calculate_derivatives(phys=self.phys, state=y),
            0, y0=y0, t_bound=h * (steps + 1), max_step=h,
        )
        for i in range(steps):
            if (i + 1) % 100 == 0:
                print(f"Completed {i + 1}/{steps} steps.")
            solver.step()
            y = solver.y.reshape(state0.shape)
            y = np.expand_dims(y, axis=0)
            self.solution = np.append(self.solution, y, axis=0)
        return self.solution


    def animate(self):
        if self.solution.shape[0] == 0:
            print("No solution to animate.")
            return

        print("Beginning animation.")
        self.anim.animate(self.solution)


if __name__ == '__main__':
    N_BODIES = 37
    REPULSIVE_STRENGTH = 10
    WIDTH = 6
    HEIGHT = 4

    FPS = 20
    DPI = 50
    RUNTIME = 10
    DRAG_COEF = 0.7

    h = 1 / FPS
    steps = int(RUNTIME / h)

    solver = PointCloudSolver(
        n_bodies=N_BODIES,
        force_multiplier=REPULSIVE_STRENGTH,
        width=WIDTH,
        height=HEIGHT,
        drag_coefficient=DRAG_COEF,
    )
    solver.solve(h=h, steps=steps)
    solver.animate()

