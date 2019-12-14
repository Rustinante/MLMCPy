"""
The basis for this Projectile model is found here:
C. Hill, “Reaching Orbit.” (Oct., 2018).
https://scipython.com/blog/reaching-orbit/
"""
import numpy as np
from scipy.constants import G
from MLMCPy.model import Model

# Convert Newtonian constant of gravitation from m3.kg-1.s-2 to km3.kg-1.s-2
G /= 1.e9

# Planet radius, km
R = 6371
# Planet mass, kg
M = 5.9722e24

fac = G * M


class Projectile(Model):
    def __init__(self, num_time_steps):
        self.num_time_steps = num_time_steps

    def evaluate(self, inputs):
        h, launch_speed, launch_angle = inputs
        return Projectile.get_trajectory(h, launch_speed, launch_angle, self.num_time_steps)

    @staticmethod
    def get_trajectory(h, launch_speed, launch_angle, num_time_steps):
        """
        Do the (very simple) numerical integration of the equation of motion.
        The satellite is released at altitude h (km) with speed launch_speed (km/s)
        at an angle launch_angle (degrees) from the normal to the planet's surface.
        """
        v0 = launch_speed
        theta = np.radians(launch_angle)

        tgrid, dt = np.linspace(0, 10000, num_time_steps, retstep=True)
        # trajectory = np.empty((num_time_steps, 2))
        trajectory = []
        v = np.zeros((num_time_steps, 2))
        # Initial rocket position, velocity and acceleration
        trajectory.append(np.array([0, R + h]))
        # trajectory[0] = 0, R + h
        v[0] = v0 * np.cos(theta), v0 * np.sin(theta)
        a = Projectile.get_acceleration(trajectory[0])

        for i, t in enumerate(tgrid[1:]):
            # Calculate the rocket's next position based on its instantaneous velocity.
            r = trajectory[-1] + v[i] * dt
            if np.hypot(*r) < R:
                # Our rocket crashed.
                break
            # Update the rocket's position, velocity and acceleration.
            trajectory.append(r)
            v[i + 1] = v[i] + a * dt
            a = Projectile.get_acceleration(trajectory[-1])

        # print(h, launch_speed, launch_angle, trajectory[-1])
        return trajectory[-1]
        # return np.array(trajectory)

    @staticmethod
    def get_acceleration(r):
        """Calculate the acceleration of the rocket due to gravity at position r."""
        r3 = np.hypot(*r) ** 3
        return -fac * r / r3


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle


    def plot_trajectory(ax, tr):
        """Plot the trajectory tr on Axes ax."""
        earth_circle = Circle((0, 0), R, facecolor=(0.9, 0.9, 0.9))
        ax.set_facecolor('k')
        ax.add_patch(earth_circle)
        ax.plot(*tr.T, c='y')
        # Make sure our planet looks circular!
        ax.axis('equal')

        # Set Axes limits to trajectory coordinate range, with some padding.
        xmin, xmax = min(tr.T[0]), max(tr.T[0])
        ymin, ymax = min(tr.T[1]), max(tr.T[1])
        dx, dy = xmax - xmin, ymax - ymin
        PAD = 0.05
        ax.set_xlim(xmin - PAD * dx, xmax + PAD * dx)
        ax.set_ylim(ymin - PAD * dy, ymax + PAD * dy)


    launch_speed, launch_angle = 2.92, 30
    # Rocket launch altitute (km)
    h = 200

    n_rows, n_cols = 4, 4
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols)
    num_samples = n_rows * n_cols
    samples = np.tile(np.array([200, 2.92, 30]), (num_samples, 1)) + np.random.uniform(low=-1, high=1, size=(
    num_samples, 3)) * np.array([[0.1, 0.5, 1]])
    for i, (h, launch_speed, launch_angle) in enumerate(samples):
        # for i, launch_speed in enumerate([3, 6.5, 7.7, 8]):
        tr = Projectile.get_trajectory(h, launch_speed, launch_angle, num_time_steps=100000)
        ax = axes[i // n_rows, i % n_cols]
        plot_trajectory(ax, tr)
        ax.set_title('{:.2f} km/s'.format(launch_speed))
    plt.tight_layout()
    # plt.savefig('orbit.png')
    plt.show()
