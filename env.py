from dataclasses import dataclass
import gymnasium as gym
import numpy as np
import pymunk
import arcade

from gymnasium import spaces


@dataclass
class GameEnvConfig:
    # General env params
    name: str = "Life"
    render_mode: str | None = None
    dt: float = 1 / 20.0  # Step and render at 20 Hz
    physics_steps_per_render: int = 2
    max_steps: int = 1000

    # World geometry params
    width: int = 800
    height: int = 800
    horizontal_padding: int = 50
    vertical_padding: int = 150

    # World physics params
    wall_friction: float = 0.5
    wall_elasticity: float = 0.8

    # Agent params
    agent_radius: int = 20
    agent_mass: float = 1.0
    move_speed: float = 200.0
    turn_speed: float = 3.0
    energy_cost_pixel: float = 0.5
    energy_cost_rad: float = 1.0
    energy_per_mass_per_time: float = 0.1
    agent_initial_energy: float = 1000.0
    agent_friction: float = 0.5
    agent_elasticity: float = 0.8

    # Vision params
    fov: float = np.radians(150)  # Field of view in radians
    view_radius: int = 200  # Radius of vision in pixels
    num_rays: int = 20  # Number of rays to cast

    # Goal params
    goal_success_threshold: int = 20

    # Reward params
    success_reward: float = 5.0
    failure_reward: float = -5.0
    progress_weight: float = 1.0
    energy_weight: float = 0.001


class GameEnv(gym.Env):
    """Game Environment"""

    metadata = {
        "render_modes": [None, "human", "rgb_array", "human_play"],
        "render_fps": 20,
    }

    def __init__(self, config: GameEnvConfig, window=None):
        """Initialize the environment"""
        self.config = config
        self.name = config.name
        self.render_mode = config.render_mode
        assert self.render_mode in self.metadata["render_modes"]
        self.dt = config.dt
        self.metadata["render_fps"] = 1 / self.dt
        self.physics_steps_per_render = config.physics_steps_per_render
        self.max_steps = config.max_steps

        # World params
        self.width = config.width
        self.height = config.height
        self.horizontal_padding = config.horizontal_padding
        self.vertical_padding = config.vertical_padding

        # World physics params
        self.wall_friction = config.wall_friction
        self.wall_elasticity = config.wall_elasticity

        # Agent params
        self.agent_radius = config.agent_radius
        self.agent_mass = config.agent_mass
        self.move_speed = config.move_speed
        self.turn_speed = config.turn_speed
        self.energy_cost_pixel = config.energy_cost_pixel
        self.energy_cost_rad = config.energy_cost_rad
        self.energy_per_mass_per_time = config.energy_per_mass_per_time
        self.agent_initial_energy = config.agent_initial_energy
        self.agent_friction = config.agent_friction
        self.agent_elasticity = config.agent_elasticity

        # Vision params
        self.fov = config.fov
        self.view_radius = config.view_radius
        self.num_rays = config.num_rays

        # Goal params
        self.goal_success_threshold = config.goal_success_threshold

        # Reward params
        self.success_reward = config.success_reward
        self.failure_reward = config.failure_reward
        self.progress_weight = config.progress_weight
        self.energy_weight = config.energy_weight

        # Observation Space
        # Dict of proprioception and visual observations
        # - mass: np.ndarray(1)
        # - energy: np.ndarray(1)
        # - vel: np.ndarray(2)
        # - ang_vel: np.ndarray(1)
        # - image: np.ndarray(width, height, 3)
        self.observation_space = spaces.Dict(
            {
                "mass": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
                "energy": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
                "vel": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
                "ang_vel": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                "image": spaces.Box(
                    low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
                ),
            }
        )

        # Action Space
        # Move: 0=Stop, 1=Forward
        # Turn: 0=None, 1=Left, 2=Right
        self.action_space = spaces.MultiDiscrete([2, 3])

        # Physics State
        self.space = None
        self.agent_body = None
        self.agent_shape = None
        self.agent_energy = 0.0
        self._prev = {}

        # Rendering Window
        if window is not None:
            self.window = window
        else:
            self.window = arcade.Window(
                self.width, self.height, self.name, visible=(self.render_mode == "human")
            )
        arcade.set_window(self.window)

        # Initialize scene
        self._init_scene()

        # Initialize current step
        self.current_step = 0

    def _init_scene(self):
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)

        # Walls
        static_body = self.space.static_body
        room_left = self.horizontal_padding
        room_right = self.width - self.horizontal_padding
        room_bottom = self.vertical_padding
        room_top = self.height - self.vertical_padding
        walls = [
            ((room_left, room_bottom), (room_right, room_bottom)),
            ((room_right, room_bottom), (room_right, room_top)),
            ((room_right, room_top), (room_left, room_top)),
            ((room_left, room_top), (room_left, room_bottom)),
        ]
        for w in walls:
            seg = pymunk.Segment(static_body, w[0], w[1], 10)
            seg.friction = self.wall_friction
            seg.elasticity = self.wall_elasticity
            self.space.add(seg)

        self.room_bounds = (room_left, room_right, room_bottom, room_top)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset step counter
        self.current_step = 0

        # Create agent body
        if self.agent_body in self.space.bodies:
            self.space.remove(self.agent_body, self.agent_shape)
        moment = pymunk.moment_for_circle(self.agent_mass, 0, self.agent_radius)
        self.agent_body = pymunk.Body(self.agent_mass, moment)

        # Set agent start position
        if options and "agent_x" in options and "agent_y" in options:
            self.agent_body.position = options["agent_x"], options["agent_y"]
        else:
            # Randomize agent start position
            padding = self.agent_radius + 1  # Padding to avoid wall clipping
            room_l, room_r, room_b, room_t = self.room_bounds
            start_x = self.np_random.uniform(room_l + padding, room_r - padding)
            start_y = self.np_random.uniform(room_b + padding, room_t - padding)
            self.agent_body.position = start_x, start_y
        
        # Set agent heading angle
        if options and "agent_angle" in options:
            self.agent_body.angle = options["agent_angle"]
        else:
            self.agent_body.angle = self.np_random.uniform(-np.pi, np.pi)

        # Create agent shape
        self.agent_shape = pymunk.Circle(self.agent_body, self.agent_radius)
        self.agent_shape.friction = self.agent_friction
        self.agent_shape.elasticity = self.agent_elasticity
        self.space.add(self.agent_body, self.agent_shape)

        # Reset agent energy
        self.agent_energy = self.agent_initial_energy

        # Set goal position
        if options and "goal_x" in options and "goal_y" in options:
            self.goal_x = options["goal_x"]
            self.goal_y = options["goal_y"]
        else:
            padding = self.agent_radius + 1  # Padding to avoid wall clipping
            room_l, room_r, room_b, room_t = self.room_bounds
            self.goal_x = self.np_random.uniform(room_l + padding, room_r - padding)
            self.goal_y = self.np_random.uniform(room_b + padding, room_t - padding)

        # Initial distance for reward calc
        self._prev["agent_goal_dist"] = np.linalg.norm(
            np.array([self.agent_body.position.x - self.goal_x, self.agent_body.position.y - self.goal_y])
        )

        return self._get_obs(), {}

    def step(self, action):
        # Apply action
        self._apply_action(action)

        # Step physics
        for _ in range(self.physics_steps_per_render):
            self.space.step(self.dt / self.physics_steps_per_render)

        # Calculate energy
        speed = self.agent_body.velocity.length
        d_dist = speed * self.dt
        d_angle = abs(self.agent_body.angular_velocity) * self.dt
        cost_per_mass = d_dist * self.energy_cost_pixel + d_angle * self.energy_cost_rad + self.energy_per_mass_per_time * self.dt
        cost = self.agent_mass * cost_per_mass
        self.agent_energy -= cost

        # Calculate reward terms
        ## Progress Reward: Positive if getting closer
        current_pos = np.array([self.agent_body.position.x, self.agent_body.position.y])
        goal_pos = np.array([self.goal_x, self.goal_y])
        current_dist = np.linalg.norm(current_pos - goal_pos)
        r_dist = self.progress_weight * (self._prev["agent_goal_dist"] - current_dist)

        ## Energy cost penalty
        r_energy = self.energy_weight * cost

        # Total reward
        reward = r_dist - r_energy

        # Check termination
        terminated, info = self._check_termination(current_dist)
        if terminated:
            if info["reason"] == "Failure":
                reward += self.failure_reward
            elif info["reason"] == "Success":
                reward += self.success_reward

        # Update step counter
        self.current_step += 1

        # Check truncation
        truncated = self._check_truncation()

        # Update previous distance
        self._prev["agent_goal_dist"] = current_dist

        return self._get_obs(), reward, terminated, truncated, info

    def _apply_action(self, action):
        move, turn = action

        angle = self.agent_body.angle
        vx, vy = 0, 0
        w = 0

        if move == 1:
            vx = np.cos(angle) * self.move_speed
            vy = np.sin(angle) * self.move_speed

        if turn == 1:
            w = self.turn_speed
        elif turn == 2:
            w = -self.turn_speed

        self.agent_body.velocity = (vx, vy)
        self.agent_body.angular_velocity = w

    def _get_obs(self):
        """Get observation from the environment."""

        # Proprioception
        mass = self.agent_body.mass
        energy = self.agent_energy
        vel = self.agent_body.velocity
        angular_vel = self.agent_body.angular_velocity

        # Visual
        self._img_obs = self._render_frame()

        return {
            "mass": np.array([mass], dtype=np.float32),
            "energy": np.array([energy], dtype=np.float32),
            "vel": np.array([vel.x, vel.y], dtype=np.float32),
            "angular_vel": np.array([angular_vel], dtype=np.float32),
            "image": self._img_obs,
        }

    def _check_termination(self, current_dist):
        terminated = False
        info = {}
        if self.agent_energy <= 0:
            terminated = True
            info["reason"] = "Failure"

        if current_dist < self.goal_success_threshold:
            terminated = True
            info["reason"] = "Success"

        return terminated, info

    def _check_truncation(self):
        return self.current_step >= self.max_steps

    def _get_vision_polygon(self):
        """Calculates the vision polygon using vectorized raycasting."""
        p = self.agent_body.position
        heading = self.agent_body.angle

        # Vectorized angle computation
        angles = np.linspace(heading - self.fov/2, heading + self.fov/2, self.num_rays)
        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)
        
        # End points for all rays
        end_x = p.x + cos_angles * self.view_radius
        end_y = p.y + sin_angles * self.view_radius
        
        points = [(p.x, p.y)]  # Center point
        
        for i in range(self.num_rays):
            end_point = (end_x[i], end_y[i])
            res = self.space.segment_query_first(p, end_point, 1, pymunk.ShapeFilter())
            
            if res:
                points.append((res.point.x, res.point.y))
            else:
                points.append(end_point)
                
        return points

    def _render_frame(self):
        arcade.set_window(self.window)
        # 1. Clear to BLACK (darkness everywhere)
        arcade.get_window().clear(color=arcade.color.BLACK)

        # 2. Draw vision polygon as the lit area (only inside room will show)
        vision_poly = self._get_vision_polygon()
        arcade.draw_polygon_filled(vision_poly, arcade.color.WHITE)
        
        # 3. Black outside walls (so vision can't "see" past walls even if rays escape)
        # Draw black rectangles outside the room bounds
        room_left, room_right = self.room_bounds[0], self.room_bounds[1]
        room_bottom, room_top = self.room_bounds[2], self.room_bounds[3]
        
        # Left side
        arcade.draw_lrbt_rectangle_filled(0, room_left, 0, self.height, arcade.color.BLACK)
        # Right side
        arcade.draw_lrbt_rectangle_filled(room_right, self.width, 0, self.height, arcade.color.BLACK)
        # Bottom
        arcade.draw_lrbt_rectangle_filled(room_left, room_right, 0, room_bottom, arcade.color.BLACK)
        # Top
        arcade.draw_lrbt_rectangle_filled(room_left, room_right, room_top, self.height, arcade.color.BLACK)
        
        # 4. Draw walls (black lines on white background = visible)
        arcade.draw_line(
            room_left, room_bottom, room_right, room_bottom,
            arcade.color.BLACK, 10,
        )
        arcade.draw_line(
            room_right, room_bottom, room_right, room_top,
            arcade.color.BLACK, 10,
        )
        arcade.draw_line(
            room_right, room_top, room_left, room_top,
            arcade.color.BLACK, 10,
        )
        arcade.draw_line(
            room_left, room_top, room_left, room_bottom,
            arcade.color.BLACK, 10,
        )

        # 5. Goal - only draw if visible (within vision cone and not blocked)
        gx, gy = self.goal_x, self.goal_y
        p = self.agent_body.position
        heading = self.agent_body.angle
        
        # Check if goal is in vision
        goal_dist = np.sqrt((gx - p.x)**2 + (gy - p.y)**2)
        goal_angle = np.arctan2(gy - p.y, gx - p.x)
        angle_diff = abs(goal_angle - heading)
        if angle_diff > np.pi:
            angle_diff = 2 * np.pi - angle_diff
        
        if goal_dist < self.view_radius and angle_diff < self.fov / 2:
            # Check if wall blocks view to goal (exclude agent's own shape)
            goal_point = (gx, gy)
            res = self.space.segment_query_first(p, goal_point, 1, pymunk.ShapeFilter())

            # Draw goal if no wall blocking OR if the hit is the agent itself
            if res is None or res.shape == self.agent_shape:
                arcade.draw_line(gx - 15, gy, gx + 15, gy, arcade.color.RED, 3)
                arcade.draw_line(gx, gy - 15, gx, gy + 15, arcade.color.RED, 3)

        # 6. Agent (always visible at center of vision)
        arcade.draw_circle_filled(p.x, p.y, self.agent_radius, arcade.color.BLUE)

        # Arrow
        ang = self.agent_body.angle
        tip = (p.x + np.cos(ang) * 30, p.y + np.sin(ang) * 30)
        left = (p.x + np.cos(ang + 2.5) * 12, p.y + np.sin(ang + 2.5) * 12)
        right = (p.x + np.cos(ang - 2.5) * 12, p.y + np.sin(ang - 2.5) * 12)
        arcade.draw_triangle_filled(
            tip[0], tip[1], left[0], left[1], right[0], right[1], arcade.color.YELLOW
        )

        # Render to image
        image_data = arcade.get_image().convert("RGB")

        # If rendering for human, flip the image buffer to display
        if self.render_mode == "human":
            arcade.get_window().flip()

        return np.array(image_data, dtype=np.uint8)

    def render(self):
        if self.render_mode == "rgb_array":
            return self._img_obs
        return None
