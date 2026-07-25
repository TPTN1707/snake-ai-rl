from collections import namedtuple
from enum import Enum
import random
import pygame
from src.config import (
    BLOCK_SIZE,
    COLOR_BLACK,
    COLOR_BLUE_DARK,
    COLOR_BLUE_LIGHT,
    COLOR_RED,
    COLOR_WHITE,
    GAME_SPEED,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)

# Initialize Pygame and fonts
pygame.init()
font = pygame.font.SysFont("arial", 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


# Simple helper class to represent coordinates
Point = namedtuple("Point", "x, y")


class SnakeGameAI:

    def __init__(self, w=WINDOW_WIDTH, h=WINDOW_HEIGHT):
        self.w = w
        self.h = h
        # Initialize display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Snake AI - Reinforcement Learning")
        self.clock = pygame.clock.get_timer() if hasattr(pygame.clock, "get_timer") else pygame.time.Clock()
        self.reset()

    def reset(self):
        """Resets the game state to start a new episode."""
        self.direction = Direction.RIGHT

        # Spawn the snake head in the center of the window
        self.head = Point(self.w / 2, self.h / 2)
        # Initial snake body with 3 segments
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y),
        ]

        self.score = 0
        self.food = None
        self._place_food()
        # Keep track of frames elapsed since the last food eaten
        self.frame_iteration = 0

    def _place_food(self):
        """Randomly places a food block on the grid, avoiding the snake body."""
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = (
            random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        )
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        """Executes one step in the environment based on the agent's action.

        action: [straight, right, left] (e.g., [1, 0, 0] means go straight)
        Returns: reward, game_over, score
        """
        self.frame_iteration += 1

        # 1. Handle window close event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. Move the snake head
        self._move(action)
        self.snake.insert(0, self.head)

        # 3. Check for game over (collision or infinite loop timeout)
        reward = 0
        game_over = False

        # Timeout logic: If snake takes too long without eating (100 frames per segment)
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10  # Heavily penalize dying
            return reward, game_over, self.score

        # 4. Check if snake ate the food
        if self.head == self.food:
            self.score += 1
            reward = 10  # Reward for eating food
            self._place_food()
            self.frame_iteration = 0  # Reset the timeout tracker
        else:
            self.snake.pop()  # Remove the tail segment to simulate movement

        # 5. Render the screen and regulate game speed
        self._update_ui()
        self.clock.tick(GAME_SPEED)

        # 6. Return reward, game over state, and current score
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        """Checks if a point (default: snake head) hits the wall or itself."""
        if pt is None:
            pt = self.head

        # Hit boundary walls
        if (
            pt.x > self.w - BLOCK_SIZE
            or pt.x < 0
            or pt.y > self.h - BLOCK_SIZE
            or pt.y < 0
        ):
            return True

        # Hit itself
        if pt in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        """Renders all game objects (snake, food, score) on screen."""
        self.display.fill(COLOR_BLACK)

        # Draw snake body
        for pt in self.snake:
            # Draw outer segment
            pygame.draw.rect(
                self.display,
                COLOR_BLUE_DARK,
                pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE),
            )
            # Draw inner segment for better visualization
            pygame.draw.rect(
                self.display,
                COLOR_BLUE_LIGHT,
                pygame.Rect(pt.x + 4, pt.y + 4, 12, 12),
            )

        # Draw food
        pygame.draw.rect(
            self.display,
            COLOR_RED,
            pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE),
        )

        # Draw score text
        text = font.render(f"Score: {self.score}", True, COLOR_WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        """Updates the direction and coordinates of the snake head based on action."""
        # Clockwise directions: [RIGHT, DOWN, LEFT, UP]
        clock_wise = [
            Direction.RIGHT,
            Direction.DOWN,
            Direction.LEFT,
            Direction.UP,
        ]
        idx = clock_wise.index(self.direction)

        if pygame.math.Vector2(action).length() == 0:
            # Fallback if action is an empty vector [0, 0, 0]
            new_dir = self.direction
        elif action[0] == 1:
            new_dir = clock_wise[idx]  # Go straight (no change)
        elif action[1] == 1:
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # Turn right (clockwise)
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # Turn left (counter-clockwise)

        self.direction = new_dir

        x = self.head.x
        y = self.head.y

        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)