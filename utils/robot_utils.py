import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon
from typing import List, Tuple, Dict, Optional


def forward_kinematics(theta1: float, theta2: float, l1: float = 1.0, l2: float = 1.0) -> List[Tuple[float, float]]:
    x1 = l1 * np.cos(theta1)
    y1 = l1 * np.sin(theta1)
    x2 = x1 + l2 * np.cos(theta1 + theta2)
    y2 = y1 + l2 * np.sin(theta1 + theta2)
    return [(0, 0), (x1, y1), (x2, y2)]

def inverse_kinematics(x: float, y: float, l1: float = 1.0, l2: float = 1.0) -> Optional[Tuple[float, float]]:
    """Berechnet inverse Kinematik für einen 2-Link-Planararm."""
    dist_sq = x**2 + y**2
    cos_theta2 = (dist_sq - l1**2 - l2**2) / (2 * l1 * l2)
    if np.abs(cos_theta2) > 1:
        return None  # Ziel nicht erreichbar

    theta2 = np.arccos(cos_theta2)
    k1 = l1 + l2 * np.cos(theta2)
    k2 = l2 * np.sin(theta2)
    theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)
    return theta1, theta2

def check_collision(theta1: float, theta2: float, scene: Dict[str, object], l1: float = 1.0, l2: float = 1.0) -> bool:
    points = forward_kinematics(theta1, theta2, l1, l2)
    link1 = LineString([points[0], points[1]])
    link2 = LineString([points[1], points[2]])
    for obstacle in scene.values():
        if link1.intersects(obstacle) or link2.intersects(obstacle):
            return True
    return False

def plot_planar_arm(theta1: float, theta2: float, path: List[Tuple[float, float]] = None,
                     l1: float = 1.0, l2: float = 1.0, scene: Dict[str, object] = None):
    points = forward_kinematics(theta1, theta2, l1, l2)
    xs, ys = zip(*points)
    plt.figure(figsize=(5, 5))

    if scene:
        for shape in scene.values():
            xs_obs, ys_obs = shape.exterior.xy if hasattr(shape, 'exterior') else shape.xy
            plt.fill(xs_obs, ys_obs, color='lightcoral', alpha=0.6)

    plt.plot(xs, ys, '-o', lw=3, color='blue')
    if path:
        for t1, t2 in path:
            px, py = zip(*forward_kinematics(t1, t2, l1, l2))
            plt.plot(px, py, color='gray', alpha=0.2)
    plt.xlim(-l1 - l2 - 0.5, l1 + l2 + 0.5)
    plt.ylim(-l1 - l2 - 0.5, l1 + l2 + 0.5)
    plt.gca().set_aspect('equal')
    plt.grid(True)
    plt.title(f"Theta1: {theta1:.2f}, Theta2: {theta2:.2f}")
    plt.show()

# --- Vollständige PRM-Planung mit Visualisierung ---
if __name__ == "__main__":
    import ipywidgets as widgets
    from IPython.display import display
    import networkx as nx
    from scipy.spatial import distance

    # Hindernisse definieren
    scene = {
        "block1": Polygon([(-0.5, 1.0), (0.5, 1.0), (0.5, 2.0), (-0.5, 2.0)]),
        "block2": Polygon([(1.0, -1.0), (2.0, -1.0), (2.0, -0.2), (1.0, -0.2)])
    }

    # Start- und Zielposition im Arbeitsraum
    xy_path = [(0.1 * i, 0.05 * i) for i in range(20)]  # Beispielhafte Koordinaten

    joint_path = []
    for x, y in xy_path:
        ik = inverse_kinematics(x, y)
        if ik is not None:
            joint_path.append(ik)

    def animate(step):
        theta1, theta2 = joint_path[step]
        plot_planar_arm(theta1, theta2, path=joint_path, scene=scene)

    if joint_path:
        widgets.interact(animate, step=widgets.IntSlider(0, 0, len(joint_path)-1, 1))
    else:
        print("⚠️ Keine gültigen Konfigurationen für Pfad berechnet.")
