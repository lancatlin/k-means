from typing import List, Dict, Tuple
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class KMenas:
    def __init__(self, k: int, points: np.ndarray):
        self.points = np.array(points)
        self.result: List[Group]
        self.best_sum = -1
        self.k = k
        for _ in range(10):
            self.execute()
        print(self.best_sum, self.result)

    def execute(self):
        random = np.random.choice(len(self.points), self.k, False)
        self.groups: List[Group] = []
        for point in self.points[random]:
            self.groups.append(Group(point))

        self.changed = True
        while self.changed:
            self.update()

        if self.sum < self.best_sum or self.best_sum == -1 :
            self.result = self.groups
            self.best_sum = self.sum

    def update(self):
        for point in self.points:
            self.selectGroup(point).add(point)
            
        self.sum = 0
        self.changed = False 
        for group in self.groups:
            if group.update():
                self.changed = True
            self.sum += group.sum
            group.clear()

    def show(self):
        fig = plt.figure()
        ax = Axes3D(fig)
        self.groups = self.result
        x, y, z, color = [], [], [], []
        for point in self.points:
            x.append(point[0])
            y.append(point[1])
            z.append(point[2])
            color.append(self.selectGroup(point).color)

        for group in self.groups:
            x.append(group.centroid[0])
            y.append(group.centroid[1])
            z.append(group.centroid[2])
            color.append('black')

        ax.scatter(x, y, z, c=color, marker='o')
        plt.show()
            
    def selectGroup(self, point: np.ndarray):
        closest: Group
        distance = -1
        for group in self.groups:
            d = group.distance(point) 
            if d < distance or distance == -1:
                closest = group
                distance = d
        return closest

def random_color() -> str :
    return "#"+''.join(np.random.choice(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F'], 6))

class Group:
    def __init__(self, point: np.ndarray):
        self.centroid: np.ndarray = point
        self.points: np.ndarray = np.empty([0, 3])
        self.color = random_color()
        self.sum = 0

    def __repr__(self) -> str:
        return self.centroid.__repr__()

    def add(self, point: np.ndarray):
        self.sum += self.distance(point)
        point = np.array([point])
        self.points = np.append(self.points, point, axis=0)

    def clear(self):
        self.points = np.empty([0, 3])
        self.sum = 0

    def update(self):
        old = np.copy(self.centroid)
        self.centroid = np.mean(self.points, axis=0)
        return not (old == self.centroid).all()

    def distance(self, point: np.ndarray) -> float:
        return ((self.centroid - point) ** 2).sum()


if __name__ == "__main__":
    p1 = np.random.random_integers(0, 100, (40, 3))
    print(p1)
    k_means = KMenas(4, p1)
    k_means.show()
