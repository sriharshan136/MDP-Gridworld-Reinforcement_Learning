import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.animation import FFMpegWriter


class GridWorldVisualizer:
    def __init__(self, world, file_path, title, artist, unit=1):
        self.world = world
        self.num_rows = world.num_rows
        self.num_cols = world.num_cols
        self.unit = unit

        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 8))
        self.ax = self.plot_gridworld(self.ax)
        self.ax.set_title(f"Grid-World (MDP): {artist}")

        metadata = dict(title=title, artist=artist)
        self.writer = FFMpegWriter(fps=60, metadata=metadata, bitrate=-1)
        self.writer.setup(self.fig, file_path, 200)

        # Store a reference to the policy arrow plot and value text
        self.policy_arrow_plots = []
        self.value_texts = []

        # Set up the text object for subtitles
        self.sub_text_obj = self.fig.text(0.5, 0.05, '', ha='center', fontsize=12)

    def plot_gridworld(self, ax, unit=1):
        plt.rcParams['animation.ffmpeg_path'] = 'data/ffmpeg.exe'

        ax.set_aspect('equal')
        ax.axis('off')

        # Draw grid lines
        for i in range(self.num_cols + 1):
            if i == 0 or i == self.num_cols:
                ax.plot([i * unit, i * unit], [0, self.num_rows * unit],
                        color='black')
            else:
                ax.plot([i * unit, i * unit], [0, self.num_rows * unit],
                        alpha=0.7, color='grey', linestyle='dashed')
        for i in range(self.num_rows + 1):
            if i == 0 or i == self.num_rows:
                ax.plot([0, self.num_cols * unit], [i * unit, i * unit],
                        color='black')
            else:
                ax.plot([0, self.num_cols * unit], [i * unit, i * unit],
                        alpha=0.7, color='grey', linestyle='dashed')

        # Draw Start, Wall, Trap, and Goal States.
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                y = (self.num_rows - 1 - i) * unit
                x = j * unit
                s = self.world.get_state_from_pos((i, j))

                # Draw walls, goal, and start states
                if self.world.map[i, j] == 3:
                    rect = patches.Rectangle((x, y), unit, unit, edgecolor='none', facecolor='black')
                    ax.add_patch(rect)
                elif self.world.map[i, j] == 2:
                    rect = patches.Rectangle((x, y), unit, unit, edgecolor='none', facecolor='red')
                    ax.add_patch(rect)
                elif self.world.map[i, j] == 1:
                    rect = patches.Rectangle((x, y), unit, unit, edgecolor='none', facecolor='green')
                    ax.add_patch(rect)
                elif self.world.start_pos is not None:
                    start_s = self.world.get_state_from_pos(self.world.start_pos)
                    if start_s == s:
                        rect = patches.Rectangle((x, y), unit, unit, edgecolor='none', facecolor='yellow')
                        ax.add_patch(rect)

        # Adjust subplot parameters to control the layout
        plt.subplots_adjust(left=0.0, right=1.0, top=0.94, bottom=0.06)

        return ax

    def plot_policy_values(self, policy, values, sub_text=None):
        # Remove previous policy arrow plots
        for plot in self.policy_arrow_plots:
            plot.remove()
        self.policy_arrow_plots = []
        for plot in self.value_texts:
            plot.remove()
        self.value_texts = []

        for i in range(self.num_rows):
            for j in range(self.num_cols):
                y = (self.num_rows - 1 - i) * self.unit
                x = j * self.unit
                s = self.world.get_state_from_pos((i, j))

                # Display state values
                if values is not None and self.world.map[i, j] != 3:
                    if values[s] is None:
                        continue
                    text = self.ax.text(x + 0.5 * self.unit, y + 0.5 * self.unit, f'{values[s]:.4f}',
                                        horizontalalignment='center', verticalalignment='center', fontsize=9)
                    self.value_texts.append(text)

                # Display policy arrows
                if policy is not None and self.world.map[i, j] == 0:
                    a = policy[s]
                    if a is None:
                        continue
                    symbol = ['^', '>', 'v', '<']
                    plot = self.ax.plot([x + 0.5 * self.unit], [y + 0.5 * self.unit], marker=symbol[a], alpha=0.5,
                                        linestyle='none', markersize=15, color='#1f77b4')[0]
                    self.policy_arrow_plots.append(plot)

        # Update subtitle text
        self.sub_text_obj.set_text(sub_text)

        self.writer.grab_frame()

    def finalize_video(self):
        self.writer.finish()
        plt.close(self.fig)
