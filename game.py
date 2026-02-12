import arcade
import numpy as np

from env import GameEnv, GameEnvConfig


class GamePlayWindow(arcade.Window):
    def __init__(self, config: GameEnvConfig):
        # Initialize window with Env dimensions
        super().__init__(config.width, config.height, config.name)

        # Setup environment
        self.env = GameEnv(config, window=self)

        # Action State: [Move, Turn]
        # Move: 0=Stop, 1=Forward
        # Turn: 0=None, 1=Left, 2=Right
        self.current_action = np.array([0, 0], dtype=np.int32)

        # Keys pressed set for smoother handling
        self.keys = set()

        # Reset environment
        self.obs, self.info = self.env.reset()
        self.total_reward = 0.0

        # UI Text Objects initialized later or simple debug draw
        self.text_info = arcade.Text(
            "", 10, self.env.height - 20, arcade.color.WHITE, 12, anchor_y="top"
        )

    def on_key_press(self, key, modifiers):
        self.keys.add(key)
        self.update_action()

    def on_key_release(self, key, modifiers):
        if key in self.keys:
            self.keys.remove(key)
        self.update_action()

    def update_action(self):
        move = 0
        turn = 0

        if arcade.key.UP in self.keys:
            move = 1

        if arcade.key.LEFT in self.keys:
            turn = 1
        elif arcade.key.RIGHT in self.keys:
            turn = 2

        self.current_action = np.array([move, turn], dtype=np.int32)

    def on_update(self, delta_time):
        # Step environment
        self.obs, reward, terminated, truncated, self.info = self.env.step(
            self.current_action
        )
        self.total_reward += reward

        if terminated or truncated:
            reason = self.info["reason"] if "reason" in self.info else "Truncated"
            print(f"Game Over! Return: {self.total_reward} | Reason: {reason}")
            self.obs, self.info = self.env.reset()
            self.total_reward = 0.0

        # Update debug text
        energy = self.obs["energy"][0]
        self.text_info.text = (
            f"Action: {self.current_action} | Energy: {energy:.1f} | Return: {self.total_reward:.2f}"
        )

    def on_draw(self):
        # Env handles clearing and drawing logic
        # Just draw the UI Overlay
        self.text_info.draw()


def main():
    # Create the window which drives the env
    GamePlayWindow(config=GameEnvConfig(render_mode="human_play"))

    # Run the game loop
    arcade.run()


if __name__ == "__main__":
    main()
