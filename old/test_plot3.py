import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


class ContourAnimation(object):
    def __init__(self, count):
        self.count = count
        self.xs = np.arange(count)
        self.ys = np.arange(count)
        self.x_coords, self.y_coords = np.meshgrid(self.xs, self.ys)
        self.z = np.zeros(self.x_coords.shape)
        self.contour = None
        self.colorbar_axes = None

    def update(self, n):
        t = (n + 1) / 100
        for x in self.xs:
            for y in self.ys:
                self.z[y][x] = ((x - self.count/2) ** 2 +
                                ((y - self.count/2)/t) ** 2)
        if self.contour:
            for artist in self.contour.collections:
                artist.remove()

        self.contour = plt.contourf(self.x_coords, self.y_coords, self.z)
        self.colorbar = plt.colorbar(cax=self.colorbar_axes)
        _, self.colorbar_axes = plt.gcf().get_axes()
        return self.contour,


def main():
    fig = plt.figure()
    fib = ContourAnimation(30)
    plt.title('Contour Animation')
    fib_ani = animation.FuncAnimation(fig,
                                      fib.update,
                                      interval=500)
    plt.show()


if __name__ == '__main__':
    main()
