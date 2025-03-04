#scripts/VLMNavAgent/graph_utils/occupancy_map.py

import math
import numpy as np
import cv2

class OccupancyMap:
    def __init__(self, resolution=0.1, size=500, origin=(-25.0, -25.0)):
        self.resolution = resolution
        self.size = size
        self.origin = origin
        
        self.grid = np.zeros((size, size), dtype=np.int8)
        self._initialize_borders()
        
    def _initialize_borders(self):
        """Initialize map borders as obstacles"""
        self.grid[0, :] = 2  # Top border
        self.grid[-1, :] = 2  # Bottom border
        self.grid[:, 0] = 2  # Left border
        self.grid[:, -1] = 2  # Right border

    def world_to_grid(self, x, y):
        """Convert world coordinates to grid indices"""
        grid_x = int((x - self.origin[0]) / self.resolution)
        grid_y = int((y - self.origin[1]) / self.resolution)
        return (
            np.clip(grid_x, 0, self.size-1),
            np.clip(grid_y, 0, self.size-1)
        )

    def grid_to_world(self, grid_x, grid_y):
        """Convert grid indices to world coordinates"""
        return (
            grid_x * self.resolution + self.origin[0],
            grid_y * self.resolution + self.origin[1]
        )

    @staticmethod
    def bresenham_line(x0, y0, x1, y1):
        """Generate line coordinates using Bresenham's algorithm"""
        points = []
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        err = dx + dy

        x, y = x0, y0
        while True:
            points.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy
        return points

    def update(self, robot_pos, depth_data, yaw, subsample_factor=10):
        """
        Update occupancy map based on depth sensor data
        Args:
            robot_pos: (x, y) tuple in world coordinates
            depth_data: 2D numpy array of depth values
            yaw: Robot's current yaw angle in radians
            subsample_factor: Spatial subsampling factor for depth columns
        """
        x, y = robot_pos
        robot_grid_x, robot_grid_y = self.world_to_grid(x, y)
        
        height, width = depth_data.shape
        h_fov = math.radians(90)  # Assuming 90° horizontal FOV

        for col_j in range(0, width, subsample_factor):
            column_depths = depth_data[:, col_j]
            valid_depths = column_depths[(column_depths > 0.5) & (column_depths < 5.0)]
            
            if len(valid_depths) == 0:
                continue
                
            d = np.min(valid_depths)
            theta = (col_j / width) * h_fov - (h_fov / 2.0)
            angle = yaw + theta
            
            # Calculate obstacle position
            obs_x = x + d * math.cos(angle)
            obs_y = y + d * math.sin(angle)
            obs_grid_x, obs_grid_y = self.world_to_grid(obs_x, obs_y)
            
            # Update grid cells along the line
            line_points = self.bresenham_line(robot_grid_x, robot_grid_y, obs_grid_x, obs_grid_y)
            
            for px, py in line_points[:-1]:
                if 0 <= px < self.size and 0 <= py < self.size:
                    self.grid[px, py] = 1  # Free space
                    
            if line_points:
                last_x, last_y = line_points[-1]
                if 0 <= last_x < self.size and 0 <= last_y < self.size:
                    self.grid[last_x, last_y] = 2  # Obstacle

    def get_visualization(self, robot_path=None, expert_path=None):
        """
        Generate visualization image with optional paths
        Args:
            robot_path: List of (x,y) tuples in world coordinates
            expert_path: List of (x,y) tuples in world coordinates
        Returns:
            3-channel BGR image (H, W, 3)
        """
        vis_img = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        
        # Color mapping
        vis_img[self.grid == 1] = [200, 200, 200]  # Free space
        vis_img[self.grid == 2] = [0, 0, 255]      # Obstacles
        
        # Convert to contiguous array after flipping
        vis_img = np.ascontiguousarray(np.flipud(vis_img))  # FIX HERE
        
        # Draw paths if provided
        if expert_path is not None and len(expert_path) > 0:
            if isinstance(expert_path, np.ndarray):
                expert_path = [tuple(p) for p in expert_path[:, :2]]
            self._draw_path(vis_img, expert_path, (255, 0, 0))
            
        if robot_path is not None and len(robot_path) > 0: 
            if isinstance(robot_path, np.ndarray):
                robot_path = [tuple(p) for p in robot_path[:, :2]]
            self._draw_path(vis_img, robot_path, (0, 255, 0))
        return cv2.resize(vis_img, (256, 256))

    def _draw_path(self, img, path, color):
        """Helper to draw path on visualization image"""
        if len(path) < 2:
            return
            
        path_points = []
        for x, y in path:
            grid_x, grid_y = self.world_to_grid(x, y)
            # Convert to visualization coordinates (flipped Y-axis)
            vis_y = self.size - 1 - grid_y
            path_points.append((grid_x, vis_y))
            
        for i in range(len(path_points)-1):
            cv2.line(img, path_points[i], path_points[i+1], color, 1)