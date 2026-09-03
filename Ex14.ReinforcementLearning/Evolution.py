from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

# Representation of all possible actions
ACTIONS = {
    0: "left",
    1: "right",
    2: "up",
    3: "down"
}

"""
Evolution is a simple custom Gymnasium environment to test and explain Q-learning.
There are 2 distinct state representations:
    - The agent's position (all_states = False)
    - The agent's position and whether or not he has leveled up (all_states = True)
In both instances, the agent must avoid the enemy in the middle of the map and can earn rewards for either consuming a rare candy or a thunderstone.
The agent starts in the bottom left corner and can move left, right, up, or down.
The agent receives a reward of 1 for consuming the rare candy, 5 for consuming the thunderstone, and -10 for running into the enemy.
It also receives a reward of -1 for trying to move into an outer wall.
The episode ends when the agent consumes the thunderstone or runs into the enemy.
As typical for a Gymnasium environment, the environment can be rendered and set to a specific frame rate,
although this will slow down the training process and is only to visualize the agent's behavior in real time.
"""
class Evolution(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10000}

    # Initialize the environment with all necessary variables
    def __init__(self, render_mode=None, all_states=False, fps=10000):
        self.size = 3   # size of the grid (size x size); tba: make this a parameter

        self.action_space = spaces.Discrete(4)  # 4 possible actions: left, right, up, down
        self.state_multiplier = 2 if all_states else 1  # 1 if only agent position is used, 2 if agent position and level status is used
        self.observation_space = spaces.Discrete(self.size*self.size*self.state_multiplier)   # number of possible states

        self.window_size = 512

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.metadata["render_fps"] = fps

        self.window = None
        self.clock = None

        # load textures for rendering
        #if self.render_mode is not None:
        assets_path = Path(__file__).resolve().parent / "assets"
        self.textures = {
                "background": pygame.image.load(assets_path / "grass.png"),
                "agents": [pygame.image.load(assets_path / "pikachu.png"), pygame.image.load(assets_path / "pikachu2.png"), \
                           pygame.image.load(assets_path / "pikachu3.png"), pygame.image.load(assets_path / "raichu.png")],
                "enemy": pygame.image.load(assets_path / "groudon.png"),
                "candy": pygame.image.load(assets_path / "candy.png"),
                "thunderstone": pygame.image.load(assets_path / "thunderstone.png"),
                "TM": pygame.image.load(assets_path / "TM.png"),
            }   

        # Set up pygame window and clock if rendering is enabled
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Evolution")
            self.textures["background"] = self.textures["background"].convert()
            self.textures["enemy"] = self.textures["enemy"].convert_alpha()
            self.textures["candy"] = self.textures["candy"].convert_alpha()
            self.textures["thunderstone"] = self.textures["thunderstone"].convert_alpha()
            self.textures["TM"] = self.textures["TM"].convert_alpha()
            self.textures["agents"][0] = self.textures["agents"][0].convert_alpha()
            self.textures["agents"][1] = self.textures["agents"][1].convert_alpha()
            self.textures["agents"][2] = self.textures["agents"][2].convert_alpha()

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        self.reset()    # reset environment to initial state

    def reset(self, *, seed=None, options=None):
        """
        state is a 2D array like [row, column] of shape (size, size)
        0 = empty, 1 = agent, 2 = small reward, 3 = high reward, -2 = enemy, high negative reward
        This is only the internal representation, the agent only knows its own position and the rewards
        """
        super().reset(seed=seed)

        self.state = np.zeros((self.size, self.size))   # initialize a grid of size x size with all zeros
        self.state[self.size-1, 0] = 1  # set agent position to bottom left corner
        self.state[self.size-1, self.size-1] = 3    # set thunderstone position to bottom right corner
        self.state[0, 0] = 2    # set candy position to top left corner
        self.state[self.size//2, self.size//2] = -2   # set enemy position to middle of grid
        self.agent_pos = [self.size-1, 0]   # save the agent's current position

        self.leveled = False    # whether or not the agent has leveled up
        self.evolved = False    # whether or not the agent has evolved
        self.agent_frame = 0    # which texture to use for the agent (0=pikachu, 1=pikachu2, 2=raichu)

        self.render()   # render the initial state

        return self.get_agent_pos(), {}     # return the agent's position as the observation

    def step(self, action):
        reward = 0  # reward for the current step
        done = False    # whether or not the episode is over
        old_pos = self.agent_pos.copy()    # save the agent's position before moving

        """
        Update the agent's position based on the action
        actions: 0 = left, 1 = right, 2 = up, 3 = down
        """
        if action == 0:
            self.agent_pos[1] = max(self.agent_pos[1]-1, 0)
        elif action == 1:
            self.agent_pos[1] = min(self.agent_pos[1]+1, self.size-1)
        elif action == 2:
            self.agent_pos[0] = max(self.agent_pos[0]-1, 0)
        elif action == 3:
            self.agent_pos[0] = min(self.agent_pos[0]+1, self.size-1)
        else:
            raise ValueError("Invalid action {}".format(action))    # raise error if action is invalid

        # check if the agent tried to move into an outer wall and give a reward of -1 if so
        if self.agent_pos[0] == old_pos[0] and self.agent_pos[1] == old_pos[1]:
            reward = -1
            return self.get_agent_pos(), reward, done, False, {}

        # update the internal state representation based on the agent's movement
        self.state[self.agent_pos[0], self.agent_pos[1]] += 1
        self.state[old_pos[0], old_pos[1]] -= 1
        
        # check if the agent walked into the enemy and end the episode if so
        if self.state[self.agent_pos[0], self.agent_pos[1]] == -1:
            reward = -10
            done = True
        # check if agent walked into the rare candy, give a reward of 1 and consume the candy if so
        elif self.state[self.agent_pos[0], self.agent_pos[1]] == 3:
            self.state[self.agent_pos[0], self.agent_pos[1]] = 1
            reward = 1
            # set the agent's level status to true to disable rendering of the candy and enable the additional state representations if all_states = True
            self.leveled = True
            self.agent_frame = 1   # change the agent's texture to pikachu2
        # check if agent walked into the thunderstone, give a reward of 5 and end the episode if so
        elif self.state[self.agent_pos[0], self.agent_pos[1]] == 4:
            reward = 5
            done = True
            self.evolved = True    # set the agent's evolution status to true to disable rendering of the thunderstone
            self.agent_frame = 3    # change the agent's texture to raichu
        else:
            reward = 0  # no reward if the agent walked onto an empty field

        self.render()   # render the updated state

        return self.get_agent_pos(), reward, done, False, {}   # return the agent's position as the observation as well as the reward and done status

    # render the current state if necessary
    def render(self):
        if self.render_mode is not None:
            return self._render_frame()

    # close the pygame window if necessary
    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
            self.clock = None

    """
    Returns the agent's position as the observation
    For the agent to be able to index the q-table with the observation, the 2D position is converted
    to an integer between 0 and size*size*(state_multiplier-1) depending on the level status, where size
    is the edge length of the grid.
    This is done by multiplying the row index by the size, adding the column index and also adding
    size*size if the agent has leveled up and additional state representations are enabled.
    """
    def get_agent_pos(self):
        return (self.agent_pos[1] + self.agent_pos[0]*self.size + self.size*self.size*(self.state_multiplier - 1)*self.leveled)

    def _render_frame(self):
        canvas = pygame.Surface((self.window_size, self.window_size))   # create a new surface to draw on

        square_size = self.window_size / self.size  # size of each square in the grid

        # draw background
        canvas.blit(self.textures["background"], (0, 0))

        # draw rare candy if not consumed yet
        if not self.leveled:
            canvas.blit(self.textures["candy"], (0, 0))

        # draw thunderstone if not consumed yet
        if not self.evolved:
            canvas.blit(self.textures["thunderstone"], ((self.size-1)*square_size, (self.size-1)*square_size))

        # draw agent with correct texture in correct position (column, row)
        canvas.blit(self.textures["agents"][self.agent_frame], (self.agent_pos[1]*square_size, self.agent_pos[0]*square_size))

        # draw enemy in the middle of the grid with respect to the square size
        canvas.blit(self.textures["enemy"], (self.size//2*square_size, self.size//2*square_size))

        # either draw the canvas to the window or return the canvas as an array depending on the render mode
        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])    # limit the frame rate
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))



    # Helper function for visualization
    def _get_frame_canvas(self, state):
        self.reset()
        if self.window is None:
            pygame.init()
            pygame.font.init()

        if state == 2:
            self.step(2)
            self.step(2)

        canvas = pygame.Surface((self.window_size, self.window_size))
        square_size = self.window_size / self.size
        canvas.blit(self.textures["background"], (0, 0))
        if not self.leveled:
            canvas.blit(self.textures["candy"], (0, 0))
        if not self.evolved:
            canvas.blit(self.textures["thunderstone"], ((self.size-1)*square_size, (self.size-1)*square_size))
        canvas.blit(self.textures["agents"][self.agent_frame], (self.agent_pos[1]*square_size, self.agent_pos[0]*square_size))
        canvas.blit(self.textures["enemy"], (self.size//2*square_size, self.size//2*square_size))

        return canvas, square_size

    # Helper function for visualization
    def _draw_text_with_outline(self, font, string, canvas, x, y):
        text = font.render(string, True, (255, 255, 255))
        canvas.blit(text, (x-1, y-1))
        canvas.blit(text, (x-1, y+1))
        canvas.blit(text, (x+1, y-1))
        canvas.blit(text, (x+1, y+1))
        text = font.render(string, True, (0, 0, 0))
        canvas.blit(text, (x, y))

    # Helper function for visualization
    def _get_q_frame_info(self, q_table, state):
        if state > self.state_multiplier:
            raise ValueError("Invalid state {}, environment was initialized with {}".format(state, self.state_multiplier))
        canvas, square_size = self._get_frame_canvas(state)

        bold_font = pygame.font.match_font("Arial", bold=True)
        font = pygame.font.Font(bold_font, 18)
        for row in range(self.size):
            for col in range(self.size):
                self._draw_text_with_outline(font, "Left: {:.4f}".format(q_table[row*self.size + col + self.size*self.size*(state-1), 0]), canvas, \
                                             col*square_size + square_size/5., row*square_size + square_size * (1./6.))
                self._draw_text_with_outline(font, "Right: {:.4f}".format(q_table[row*self.size + col + self.size*self.size*(state-1), 1]), canvas, \
                                             col*square_size + square_size/5., row*square_size + square_size * (2./6.))
                self._draw_text_with_outline(font, "Up: {:.4f}".format(q_table[row*self.size + col + self.size*self.size*(state-1), 2]), canvas, \
                                             col*square_size + square_size/5., row*square_size + square_size * (3./6.))
                self._draw_text_with_outline(font, "Down: {:.4f}".format(q_table[row*self.size + col + self.size*self.size*(state-1), 3]), canvas, \
                                             col*square_size + square_size/5., row*square_size + square_size * (4./6.))

        return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))




"""
This could be an done as an additional exercise for you, if you are interested:
-------------------------------------------------------------------------------
The goal is to replace the rare candy with a consumable TM that enables the agent to defeat the enemy.
For more understandable results, the enemy should be moved to the top right corner. This is to see the
difference between the agent's behavior with the different state representations, because the agent only
focuses on the thunderstone with all_states = False.
Defeating the enemy should give a higher reward than consuming the thunderstone, so when the agent can
differentiate if it has consumed the TM or not, it should be able to focus on the higher reward.
Those results are to be investigated by analyzing the q-table with and without the additional state representations.
It is up to you to decide if you want to alter the original environment or create a new one based on it like
it is done here. The exact reward values are also up to you, but the ones here show very clear results.
"""
class Evolutionv2(Evolution):
    def reset(self, *, seed=None, options=None):
        self.tm = False     # whether or not the agent has consumed the TM; the same as the level status in the original environment
        self.enemy = True   # whether or not the enemy is still alive for rendering purposes
        obs, info = super().reset(seed=seed, options=options)   # call the original reset function to initialize the environment
        self.state[self.size//2, self.size//2] = 0   # remove enemy from middle of grid
        self.state[0, self.size-1] = -2   # set enemy position to top right corner
        self.render()
        return obs, info  # the observation can be returned directly because it is only the agent's position with respect to the TM status

    def step(self, action):
        reward = 0  # reward for the current step
        done = False    # whether or not the episode is over
        old_pos = self.agent_pos.copy()    # save the agent's position before moving

        """
        Update the agent's position based on the action
        actions: 0 = left, 1 = right, 2 = up, 3 = down
        """
        if action == 0:
            self.agent_pos[1] = max(self.agent_pos[1]-1, 0)
        elif action == 1:
            self.agent_pos[1] = min(self.agent_pos[1]+1, self.size-1)
        elif action == 2:
            self.agent_pos[0] = max(self.agent_pos[0]-1, 0)
        elif action == 3:
            self.agent_pos[0] = min(self.agent_pos[0]+1, self.size-1)
        else:
            raise ValueError("Invalid action {}".format(action))    # raise error if action is invalid

        # check if the agent tried to move into an outer wall and give a reward of -1 if so
        if self.agent_pos[0] == old_pos[0] and self.agent_pos[1] == old_pos[1]:
            reward = -1
            return self.get_agent_pos(), reward, done, False, {}

        # update the internal state representation based on the agent's movement
        self.state[self.agent_pos[0], self.agent_pos[1]] += 1
        self.state[old_pos[0], old_pos[1]] -= 1
        
        # check if the agent walked into the enemy and end the episode if so
        if self.state[self.agent_pos[0], self.agent_pos[1]] == -1:
            reward = -10 if not self.tm else 6  # give a reward of -10 if the agent has not consumed the TM, otherwise give a reward of 6
            self.enemy = False if self.tm else True  # disable rendering of the enemy if the agent has consumed the TM
            done = True
        # check if agent walked into the TM, give a reward of 1 and consume the TM if so
        elif self.state[self.agent_pos[0], self.agent_pos[1]] == 3:
            self.state[self.agent_pos[0], self.agent_pos[1]] = 1
            reward = 1
            # set the agent's TM status to true to disable rendering of the TM and enable the additional state representations if all_states = True
            self.tm = True
            self.agent_frame = 2   # change the agent's texture to pikachu3
        # check if agent walked into the thunderstone, give a reward of 5 and end the episode if so
        elif self.state[self.agent_pos[0], self.agent_pos[1]] == 4:
            reward = 5
            done = True
            self.evolved = True    # set the agent's evolution status to true to disable rendering of the thunderstone
            self.agent_frame = 3    # change the agent's texture to raichu
        else:
            reward = 0  # no reward if the agent walked onto an empty field

        self.render()   # render the updated state

        return self.get_agent_pos(), reward, done, False, {}   # return the agent's position as the observation as well as the reward and done status

    def _render_frame(self):
        canvas = pygame.Surface((self.window_size, self.window_size))   # create a new surface to draw on

        square_size = self.window_size / self.size  # size of each square in the grid

        # draw background
        canvas.blit(self.textures["background"], (0, 0))

        # draw TM if not consumed yet
        if not self.tm:
            canvas.blit(self.textures["TM"], (0, 0))

        # draw thunderstone if not consumed yet
        if not self.evolved:
            canvas.blit(self.textures["thunderstone"], ((self.size-1)*square_size, (self.size-1)*square_size))

        # draw agent with correct texture in correct position (column, row)
        canvas.blit(self.textures["agents"][self.agent_frame], (self.agent_pos[1]*square_size, self.agent_pos[0]*square_size))

        # draw enemy in the top right corner of the grid with respect to the square size
        if self.enemy:
            canvas.blit(self.textures["enemy"], ((self.size-1)*square_size, 0))

        # either draw the canvas to the window or return the canvas as an array depending on the render mode
        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])    # limit the frame rate
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    # Returns the agent's position as the observation; same as in the original environment except that it's the TM status instead of the level status
    def get_agent_pos(self):
        return (self.agent_pos[1] + self.agent_pos[0]*self.size + self.size*self.size*(self.state_multiplier - 1)*self.tm)

    # Helper function for visualization
    def _get_frame_canvas(self, state):
        self.reset()
        if self.window is None:
            pygame.init()
            pygame.font.init()

        if state == 2:
            self.step(2)
            self.step(2)

        canvas = pygame.Surface((self.window_size, self.window_size))
        square_size = self.window_size / self.size
        canvas.blit(self.textures["background"], (0, 0))
        if not self.tm:
            canvas.blit(self.textures["TM"], (0, 0))
        if not self.evolved:
            canvas.blit(self.textures["thunderstone"], ((self.size-1)*square_size, (self.size-1)*square_size))
        canvas.blit(self.textures["agents"][self.agent_frame], (self.agent_pos[1]*square_size, self.agent_pos[0]*square_size))
        if self.enemy:
            canvas.blit(self.textures["enemy"], ((self.size-1)*square_size, 0))

        return canvas, square_size
