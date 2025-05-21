import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import gaussian_filter, binary_dilation, label

# ================== Configuration Parameters ==================
RADIUS = 6371
GRID_RES = 180
NUM_PLATES = 10
TIME_STEP = 1e3
ROTATION_SCALE = 5e-6
NUM_STEPS = 1000
MAX_ELEVATION = 8000
MIN_ELEVATION = -6000
UPLIFT_FACTOR = 8000  # stronger uplift for collision zones
RIFT_DROP_FACTOR = 4000  # sharper drop for divergence
EROSION_RATE = 0.4
SMOOTHING_SIGMA = 5
ISOLATED_DROP = 0.4
GHOST_CLEAN_THRESHOLD = -3000

# ================== Utility Functions ==================
def normalize(v):
    return v / np.linalg.norm(v)

def latlon_to_cartesian(lat, lon):
    lat, lon = np.radians(lat), np.radians(lon)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.stack((x, y, z), axis=-1)

def cartesian_to_latlon(cartesian):
    x, y, z = cartesian[..., 0], cartesian[..., 1], cartesian[..., 2]
    lat = np.degrees(np.arcsin(z))
    lon = np.degrees(np.arctan2(y, x))
    return lat, lon

def rotate_vector(vectors, axis, angle):
    axis = normalize(axis)
    v_dot_axis = np.dot(vectors, axis)
    return (vectors * np.cos(angle)[:, np.newaxis] +
            np.cross(np.tile(axis, (vectors.shape[0], 1)), vectors) * np.sin(angle)[:, np.newaxis] +
            axis * v_dot_axis[:, np.newaxis] * (1 - np.cos(angle)[:, np.newaxis]))

# ================== Plate Generation ==================
def generate_plate_seeds(n):
    lats = np.random.uniform(-90, 90, n)
    lons = np.random.uniform(-180, 180, n)
    return np.column_stack((lats, lons))

def assign_plates(lat_grid, lon_grid, seeds):
    plate_map = np.zeros(lat_grid.shape, dtype=int)
    seed_cart = latlon_to_cartesian(seeds[:, 0], seeds[:, 1])
    for i in range(lat_grid.shape[0]):
        for j in range(lat_grid.shape[1]):
            point_cart = latlon_to_cartesian(lat_grid[i, j], lon_grid[i, j])
            dists = np.linalg.norm(seed_cart - point_cart, axis=1)
            plate_map[i, j] = np.argmin(dists)
    return plate_map

def assign_plate_properties(num_plates):
    crust_types = np.random.choice(['continental', 'oceanic'], num_plates, p=[0.3, 0.7])
    crust_thickness = np.where(crust_types == 'continental',
                                np.random.uniform(30, 70, num_plates),
                                np.random.uniform(5, 15, num_plates))
    elevation_base = np.where(crust_types == 'continental', MAX_ELEVATION // 8, MIN_ELEVATION)
    return crust_types, crust_thickness, elevation_base

# ================== Plate Motion ==================
def rotate_plate_map(lat_grid, lon_grid, plate_map, axes, velocities, time_step):
    flat_lat = lat_grid.flatten()
    flat_lon = lon_grid.flatten()
    points_cart = latlon_to_cartesian(flat_lat, flat_lon)
    new_cart = np.zeros_like(points_cart)
    for pid in range(NUM_PLATES):
        mask = plate_map.flatten() == pid
        if not np.any(mask):
            continue
        angle = velocities[pid] * time_step
        new_cart[mask] = rotate_vector(points_cart[mask], axes[pid], np.full(np.sum(mask), angle))
    new_lat, new_lon = cartesian_to_latlon(new_cart)
    return new_lat.reshape(lat_grid.shape), new_lon.reshape(lon_grid.shape)

def build_elevation_map(plate_map, elevation_base):
    return elevation_base[plate_map]

def classify_boundaries(plate_map):
    boundary_mask = np.zeros_like(plate_map, dtype=bool)
    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
        rolled = np.roll(np.roll(plate_map, dx, axis=0), dy, axis=1)
        boundary_mask |= (plate_map != rolled)
    return boundary_mask

def compute_motion_vectors(lat_grid, lon_grid, plate_map, axes, velocities):
    motion_vectors = np.zeros(lat_grid.shape + (3,))
    flat_lat = lat_grid.flatten()
    flat_lon = lat_grid.flatten()
    cart_coords = latlon_to_cartesian(flat_lat, flat_lon)
    for pid in range(NUM_PLATES):
        mask = plate_map.flatten() == pid
        if not np.any(mask): continue
        axis = axes[pid]
        angle = velocities[pid]
        cross = np.cross(np.tile(axis, (np.sum(mask), 1)), cart_coords[mask])
        motion_vectors.reshape(-1, 3)[mask] = cross * angle
    return motion_vectors

def simulate_uplift(elevation_map, lat_grid, lon_grid, motion_vectors, plate_map, boundary_mask):
    uplift = np.zeros_like(elevation_map)
    for i in range(1, elevation_map.shape[0]-1):
        for j in range(1, elevation_map.shape[1]-1):
            if not boundary_mask[i, j]: continue
            pid = plate_map[i, j]
            for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                ni, nj = i+di, j+dj
                if plate_map[ni, nj] != pid:
                    mv1 = motion_vectors[i, j]
                    mv2 = motion_vectors[ni, nj]
                    rel_velocity = mv1 - mv2
                    norm_vec = latlon_to_cartesian(lat_grid[i, j], lon_grid[i, j])
                    dot = np.dot(rel_velocity, norm_vec)
                    if dot < -1e-10:
                        uplift[i, j] += UPLIFT_FACTOR * (-dot)
                    elif dot > 1e-10:
                        uplift[i, j] -= RIFT_DROP_FACTOR * dot
    return elevation_map + uplift

def apply_erosion(elevation_map):
    smoothed = gaussian_filter(elevation_map, sigma=SMOOTHING_SIGMA)
    delta = elevation_map - smoothed
    eroded = elevation_map - delta * EROSION_RATE

    isolated = np.zeros_like(elevation_map, dtype=bool)
    for i in range(1, elevation_map.shape[0] - 1):
        for j in range(1, elevation_map.shape[1] - 1):
            neighborhood = elevation_map[i-1:i+2, j-1:j+2]
            center = elevation_map[i, j]
            if np.all(np.abs(neighborhood - center) > 500):
                isolated[i, j] = True
    eroded[isolated] -= ISOLATED_DROP * np.abs(eroded[isolated])

    land_mask = eroded > GHOST_CLEAN_THRESHOLD
    ghost_isolated = land_mask & (~binary_dilation(land_mask))
    eroded[ghost_isolated] = MIN_ELEVATION

    return eroded

# ================== Main Simulation ==================
def simulate_geology():
    lats = np.linspace(-90, 90, GRID_RES)
    lons = np.linspace(-180, 180, GRID_RES * 2)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    seeds = generate_plate_seeds(NUM_PLATES)
    plate_map = assign_plates(lat_grid, lon_grid, seeds)
    axes = np.array([normalize(np.random.randn(3)) for _ in range(NUM_PLATES)])
    velocities = np.random.uniform(0.5, 2.0, NUM_PLATES) * ROTATION_SCALE
    crust_types, crust_thickness, elevation_base = assign_plate_properties(NUM_PLATES)
    current_lat_grid = lat_grid.copy()
    current_lon_grid = lon_grid.copy()
    current_plate_map = plate_map
    terrain_history = []
    for step in range(NUM_STEPS):
        new_lat, new_lon = rotate_plate_map(current_lat_grid, current_lon_grid, current_plate_map, axes, velocities, TIME_STEP)
        new_plate_map = assign_plates(new_lat, new_lon, seeds)
        elevation_map = build_elevation_map(new_plate_map, elevation_base)
        boundary_mask = classify_boundaries(new_plate_map)
        motion_vectors = compute_motion_vectors(lat_grid, lon_grid, new_plate_map, axes, velocities)
        elevated = simulate_uplift(elevation_map, lat_grid, lon_grid, motion_vectors, new_plate_map, boundary_mask)
        eroded = apply_erosion(elevated)
        terrain_history.append(eroded)
        current_lat_grid, current_lon_grid = new_lat, new_lon
        current_plate_map = new_plate_map
    return lat_grid, lon_grid, terrain_history

# ================== Run and Save ==================
if __name__ == "__main__":
    lat_grid, lon_grid, terrain_history = simulate_geology()
    fig, ax = plt.subplots(figsize=(14, 7))
    cmap = plt.get_cmap("terrain")
    quad = ax.pcolormesh(lon_grid, lat_grid, terrain_history[0], shading='auto', cmap=cmap)
    cbar = plt.colorbar(quad, ax=ax, label="Elevation (m)")
    ax.set_title("Time-Evolving Planetary Topography")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    def update(frame):
        quad.set_array(terrain_history[frame].ravel())
        ax.set_title(f"Topography at Step {frame + 1}")
        return quad,
    ani = animation.FuncAnimation(fig, update, frames=len(terrain_history), blit=False, repeat=False)
    ani.save("planetary_topography_evolution.mp4", writer="ffmpeg", fps=30)