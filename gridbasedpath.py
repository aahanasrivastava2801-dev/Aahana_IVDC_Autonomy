
import heapq
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# GRID-BASED PATH PLANNING
# Algorithms: Breadth-First Search (BFS) and A*
# 0 = free cell, 1 = obstacle


# 1. Create the grid map
ROWS = 15
COLS = 20

grid = np.zeros((ROWS, COLS), dtype=int)

# Add obstacles (walls)
grid[3:12, 5] = 1
grid[3, 5:12] = 1
grid[8, 5:15] = 1
grid[4:13, 12] = 1
grid[12, 2:9] = 1

# Create a gap in some walls so a path can exist
grid[7, 5] = 0
grid[3, 8] = 0
grid[8, 10] = 0
grid[10, 12] = 0
grid[12, 5] = 0

# 2. Define start and goal as (row, column)
start = (1, 1)
goal = (13, 18)

# Possible movements: up, down, left, right
MOVES = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1)
]


def valid_neighbors(position, grid):
    """Return all free, in-bounds neighboring cells."""
    row, col = position
    neighbors = []

    for dr, dc in MOVES:
        nr = row + dr
        nc = col + dc

        inside_grid = 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]

        if inside_grid and grid[nr, nc] == 0:
            neighbors.append((nr, nc))

    return neighbors


# 3. Breadth-First Search
def bfs(grid, start, goal):
    """Find a shortest path when every move has equal cost."""
    queue = deque([start])
    parent = {start: None}
    explored_order = []

    while queue:
        current = queue.popleft()
        explored_order.append(current)

        if current == goal:
            break

        for neighbor in valid_neighbors(current, grid):
            if neighbor not in parent:
                parent[neighbor] = current
                queue.append(neighbor)

    if goal not in parent:
        return [], explored_order

    # Reconstruct the path by tracing parents backward.
    path = []
    current = goal

    while current is not None:
        path.append(current)
        current = parent[current]

    path.reverse()
    return path, explored_order


# 4. A* Search
def heuristic(a, b):
    """Manhattan distance for four-direction grid movement."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(grid, start, goal):
    """Find a shortest path using cost-so-far plus a heuristic."""
    priority_queue = []
    heapq.heappush(priority_queue, (heuristic(start, goal), 0, start))

    parent = {start: None}
    cost_so_far = {start: 0}
    explored_order = []
    explored_set = set()

    while priority_queue:
        _, current_cost, current = heapq.heappop(priority_queue)

        # Ignore outdated queue entries.
        if current_cost != cost_so_far.get(current):
            continue

        if current in explored_set:
            continue

        explored_set.add(current)
        explored_order.append(current)

        if current == goal:
            break

        for neighbor in valid_neighbors(current, grid):
            new_cost = current_cost + 1

            if new_cost < cost_so_far.get(neighbor, float("inf")):
                cost_so_far[neighbor] = new_cost
                parent[neighbor] = current
                priority = new_cost + heuristic(neighbor, goal)
                heapq.heappush(
                    priority_queue,
                    (priority, new_cost, neighbor)
                )

    if goal not in parent:
        return [], explored_order

    path = []
    current = goal

    while current is not None:
        path.append(current)
        current = parent[current]

    path.reverse()
    return path, explored_order


# 5. Run both algorithms
bfs_path, bfs_explored = bfs(grid, start, goal)
astar_path, astar_explored = astar(grid, start, goal)

if not bfs_path:
    raise RuntimeError(
        "No path found. Adjust the obstacles or start/goal positions."
    )

if not astar_path:
    raise RuntimeError(
        "A* could not find a path. Check the grid and start/goal positions."
    )


# 6. Create a visualization for each algorithm
def draw_result(ax, grid, path, explored, start, goal, title):
    """
    Cell values in the visualization:
    0 = free
    1 = obstacle
    2 = explored
    3 = final path
    4 = start
    5 = goal
    """
    display_grid = np.zeros_like(grid)

    # Obstacles
    display_grid[grid == 1] = 1

    # Explored cells
    for cell in explored:
        if cell != start and cell != goal and grid[cell] == 0:
            display_grid[cell] = 2

    # Final path overrides the explored color
    for cell in path:
        if cell != start and cell != goal:
            display_grid[cell] = 3

    display_grid[start] = 4
    display_grid[goal] = 5

    colors = [
        "#FFFFFF",  # Free cells
        "#333333",  # Obstacles
        "#8ECAE6",  # Explored cells
        "#FFB703",  # Final path
        "#2A9D8F",  # Start
        "#E63946"   # Goal
    ]

    cmap = ListedColormap(colors)

    ax.imshow(
        display_grid,
        cmap=cmap,
        vmin=0,
        vmax=5,
        interpolation="nearest"
    )

    ax.set_title(title, fontsize=12)
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which="minor", color="#BBBBBB", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_xticks(range(grid.shape[1]))
    ax.set_yticks(range(grid.shape[0]))
    ax.tick_params(labelsize=7)

    legend_items = [
        Patch(facecolor=colors[0], edgecolor="#BBBBBB", label="Free"),
        Patch(facecolor=colors[1], label="Obstacle"),
        Patch(facecolor=colors[2], label="Explored"),
        Patch(facecolor=colors[3], label="Final path"),
        Patch(facecolor=colors[4], label="Start"),
        Patch(facecolor=colors[5], label="Goal")
    ]

    ax.legend(
        handles=legend_items,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        fontsize=8
    )


# 7. Plot BFS and A* side by side
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

draw_result(
    axes[0], grid, bfs_path, bfs_explored,
    start, goal,
    f"BFS | Path length: {len(bfs_path) - 1} moves"
)

draw_result(
    axes[1], grid, astar_path, astar_explored,
    start, goal,
    f"A* | Path length: {len(astar_path) - 1} moves"
)

fig.suptitle("Grid-Based Path Planning: BFS vs A*", fontsize=16)
plt.tight_layout()
plt.savefig("grid_path_planning.png", dpi=200, bbox_inches="tight")
plt.show()


# 8. Print comparison statistics
print("\n========== PATH PLANNING RESULTS ==========")

print("\nBFS:")
print("Path found:", bool(bfs_path))
print("Path length:", len(bfs_path) - 1, "moves")
print("Explored cells:", len(bfs_explored))

print("\nA*:")
print("Path found:", bool(astar_path))
print("Path length:", len(astar_path) - 1, "moves")
print("Explored cells:", len(astar_explored))

print("\nSaved visualization: grid_path_planning.png")
