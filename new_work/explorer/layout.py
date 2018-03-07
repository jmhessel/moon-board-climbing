import tkinter as tk
from PIL import ImageTk
from PIL import Image

import os
import sys
import_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/types'
sys.path.append(import_path)
from climbset import Climbset


def format_image(climb_image):
    climb_image = climb_image.crop((0, 0, 11, 18))
    wpercent = (300 / float(climb_image.size[0]))
    hsize = int((float(climb_image.size[1]) * float(wpercent)))
    climb_image = climb_image.resize((300, hsize), Image.ANTIALIAS)
    return climb_image


class ClimbsetNavigator:

    def __init__(self, climbset):
        self.app_root = tk.Tk()
        # Create the variables for managing the display
        self.climb_num = 0
        self.overlay_visible = True
        self.climbset = climbset

        # Bind keypress events
        self.app_root.bind('<Left>', self.left_event)
        self.app_root.bind('<Right>', self.right_event)
        self.app_root.bind('<Escape>', self.close_window)
        self.app_root.bind('<Delete>', self.delete_event)

        # Create the display elements
        self.top_label = tk.Label(self.app_root, text="Image 1 of {}".format(len(self.climbset.climbs)))
        self.left_button = tk.Button(self.app_root, text="<-", command=self.last_image)
        self.right_button = tk.Button(self.app_root, text="->", command=self.next_image)
        self.toggle_button = tk.Button(self.app_root, text="Toggle", command=self.toggle_overlay)
        self.delete_button = tk.Button(self.app_root, text="Delete", command=self.delete_current)
        self.save_button = tk.Button(self.app_root, text="Save Climbs", command=self.save_all)
        self.main_image = tk.Label(self.app_root)

        # Manage the layout
        self.top_label.grid(column=0, row=0, columnspan=5, padx=10, pady=10)

        self.left_button.grid(column=0, row=1, padx=10, pady=10)
        self.right_button.grid(column=4, row=1, padx=10, pady=10)

        self.toggle_button.grid(column=0, row=2, columnspan=2, padx=10, pady=10)
        self.delete_button.grid(column=2, row=2, padx=10, pady=10)
        self.save_button.grid(column=3, row=2, columnspan=2, padx=10, pady=10)

        self.main_image.grid(column=1, row=1, columnspan=3, padx=10, pady=10)

        # Manage the initial state of buttons
        self.left_button.config(state=tk.DISABLED)
        if len(self.climbset.climbs) == 1:
            self.right_button.config(state=tk.DISABLED)
            self.delete_button.config(state=tk.DISABLED)

        self.set_image_from_index()

    def left_event(self, event):
        self.last_image()

    def right_event(self, event):
        self.next_image()

    def close_window(self,event):
        self.app_root.withdraw()

    def delete_event(self,event):
        self.delete_current()

    def save_all(self):
        from pathlib import Path
        import pickle
        save_dir = str(Path().resolve().parent) + '/data/climbsets/'
        save_name = 'climbs.pkl'
        with open(save_dir + save_name, 'wb') as handle:
            pickle.dump(self.climbset, handle)

    def delete_current(self):
        if len(self.climbset.climbs) > 1:
            self.climbset.climbs.pop(self.climb_num)
            if self.climb_num == 0:
                self.process_image_change()
            else:
                self.last_image()

        if len(self.climbset.climbs) <= 1:
            self.delete_button.config(state=tk.DISABLED)

    def toggle_overlay(self):
        self.overlay_visible = not self.overlay_visible

    def set_image_from_index(self):
        # Change the image which is displayed to match the index
        image = self.climbset.climbs[self.climb_num].as_image()
        image = format_image(image)
        image.save('current.png')
        self.img = ImageTk.PhotoImage(file='current.png')
        self.main_image.configure(image=self.img)

    def run(self):
        # Show the app
        self.app_root.mainloop()

    def next_image(self):
        # Move to next image
        if self.climb_num < len(self.climbset.climbs) - 1:
            self.climb_num += 1
            self.process_image_change()

    def last_image(self):
        # Move to previous image
        if self.climb_num > 0:
            self.climb_num += -1
            self.process_image_change()

    def process_image_change(self):
        self.set_image_from_index()
        self.update_view_state()

    def update_view_state(self):
        # Check that the title at the top and the left and right buttons are in the correct state
        self.top_label.configure(text='Image {} of {}'.format(self.climb_num + 1, len(self.climbset.climbs)))

        if self.climb_num == 0:
            self.left_button.config(state=tk.DISABLED)
        elif self.climb_num >= 1:
            self.left_button.config(state=tk.NORMAL)

        if self.climb_num == len(self.climbset.climbs) - 1:
            self.right_button.config(state=tk.DISABLED)
        elif self.climb_num <= len(self.climbset.climbs):
            self.right_button.config(state=tk.NORMAL)

if __name__ == '__main__':
    example_no_string = ['ChDlHnGjEr', 'JbIeDhDjCmEoBr', 'FeHhJhHkEjEmEnIhEoDqEr']
    loaded_climbset = Climbset(example_no_string, 'sample')

    app = ClimbsetNavigator(loaded_climbset)
    app.run()
