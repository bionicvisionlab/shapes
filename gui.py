import tkinter as tk
from tkinter import filedialog
from pulse2percept.implants import ArgusII

from shapes import load_shapes

class VisualPerceptsGUI:
    def __init__(self, master):
        self.percepts_filepath = None
        self.master = master
        master.title("Visual Percepts GUI")
        self.width = master.winfo_width()

        # Create left panel for Setup and Implant
        self.left_panel = tk.Frame(master, bg="white", width=self.width//4)
        self.left_panel.pack(side="left", fill="both", expand=True)

        # Create header and load button in Setup panel
        setup_header = tk.Label(self.left_panel, text="Setup", font=("Arial", 14), bg="white")
        setup_header.pack(padx=10, pady=10)

        load_button = tk.Button(self.left_panel, text="Load", command=self.load_file)
        load_button.pack(padx=10, pady=10)

        self.implant = ArgusII()

        self.implant_panel = tk.Canvas(self.left_panel, width=self.width//4, height=400, bg="white", highlightthickness=0)
        self.implant_panel.pack(padx=10, pady=10, side='bottom')

        # Create header and implant representation in Implant panel
        implant_header = tk.Label(self.left_panel, text="Implant", font=("Arial", 14), bg="white")
        implant_header.pack(side='bottom', padx=10, pady=10)

        # Create right panel for Percepts
        self.right_panel = tk.Frame(master, bg="white")
        self.right_panel.pack(side="right", fill="both", expand=True)

        # Create header for Percepts panel
        percepts_header = tk.Label(self.right_panel, text="Percepts", font=("Arial", 14), bg="white")
        percepts_header.pack(padx=10, pady=10)

        # Create larger canvas for Percepts panel
        self.percepts_panel = tk.Canvas(self.right_panel, width=800, height=800, bg="white")
        self.percepts_panel.pack(padx=10, pady=10)

        # Create implant representation on the left panel
        self.create_implant_representation()

        # Set resize callback
        master.bind("<Configure>", self.on_resize)

    def load_file(self):
        # Open a file dialog to select a file to load
        file_path = filedialog.askopenfilename()

        if file_path:
            self.percepts_filepath = file_path
            self.load_percepts()
    
    def load_percepts(self):
        if self.percepts_filepath is None:
            return
        self.dataset = load_shapes(self.percepts_filepath)
        print("loaded ", len(self.dataset), " percepts")
        

    def on_resize(self, event):
        width = self.master.winfo_width()
        if width != self.width:
            self.left_panel.configure(width=width//4)
            self.implant_panel.configure(width=width//4)
            self.percepts_panel.configure(width=width * 3 / 4)
            self.width = width

    def create_implant_representation(self):
        # Get the implant and electrode information
        panel = self.implant_panel
        implant = self.implant
        electrodes = implant.electrodes
        electrode_names = implant.electrode_names

        # Find minimum x and y values
        min_x = min([electrodes[e].x for e in electrode_names])
        min_y = min([electrodes[e].y for e in electrode_names]) 
        max_x = max([electrodes[e].x for e in electrode_names]) 
        max_y = max([electrodes[e].y for e in electrode_names]) 

        resize = max ([max_x - min_x, max_y - min_y])

        # Create a canvas to draw on
        canvas_width = 500
        canvas_height = 500
        canvas = tk.Canvas(panel, width=canvas_width, height=canvas_height, bg='white', highlightthickness=0)
        canvas.pack(expand=tk.YES, fill=tk.BOTH)

        # Draw each electrode as a white circle with black border
        for e in electrode_names:
            r = 15
            x = (electrodes[e].x - min_x) / resize * 450 + 25 
            y = (electrodes[e].y - min_y) / resize * 450 + r
            
            circle = canvas.create_oval(x-r, y-r, x+r, y+r, outline='black', fill='white')

            # Bind the blue color on hover event to the electrode circle
            canvas.tag_bind(circle, '<Enter>', lambda event, circle=circle: canvas.itemconfig(circle, fill='blue'))
            canvas.tag_bind(circle, '<Leave>', lambda event, circle=circle: canvas.itemconfig(circle, fill='white'))

        # Add a title to the panel
        # label = tk.Label(panel, text="Implant")
        # label.pack(side=tk.TOP, padx=10, pady=10)

root = tk.Tk()
gui = VisualPerceptsGUI(root)
root.mainloop()
