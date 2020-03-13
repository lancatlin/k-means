from typing import List, Dict, Tuple
import numpy as np 
import matplotlib.pyplot as plt

class KMenas:
    def __init__(self, k: int, points: np.ndarray):
        self.points: np.ndarray = np.array(points)
        random = np.random.choice(self.points.shape[0], k, False)
        self.groups: List[Group] = []
        for point in self.points[random]:
            self.groups.append(Group(point))

        self.changed = True
        while self.changed:
            self.update()
            print(self.groups)

    def update(self):
        for point in self.points:
            self.selectGroup(point).add(point)
            
        self.changed = False 
        for group in self.groups:
            if group.update():
                self.changed = True
            group.clear()

    def show(self):
        x, y, color = [], [], []
        for point in self.points:
            x.append(point[0])
            y.append(point[1])
            color.append(self.selectGroup(point).color)

        for group in self.groups:
            print(group.centroid)
            x.append(group.centroid[0])
            y.append(group.centroid[1])
            color.append('black')

        plt.scatter(x=x, y=y, c=color)
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
        self.points: np.ndarray = np.empty([0, 2])
        self.color = random_color()

    def __repr__(self) -> str:
        return self.centroid.__repr__()

    def add(self, point: np.ndarray):
        point = np.array([point])
        self.points = np.append(self.points, point, axis=0)

    def clear(self):
        self.points = np.empty([0, 2])

    def update(self):
        old = np.copy(self.centroid)
        self.centroid = np.mean(self.points, axis=0)
        return not (old == self.centroid).all()

    def distance(self, point: np.ndarray) -> float:
        return ((self.centroid - point) ** 2).sum()


if __name__ == "__main__":
    p1 = np.random.random_integers(0, 50, (50, 2))
    k_means = KMenas(2, p1)
    k_means.show()