import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from pulse2percept.implants import ArgusII
from pulse2percept.models import AxonMapModel
import numpy as np
from shapes import load_shapes, subject_params, model_from_params

class VisualPerceptsGUI:
    def __init__(self, master):
        self.percepts_filepath = None
        self.loaded_percepts = False
        self.dataset = None
        self.current_dataset = None
        self.subject_combobox = None
        self.implant = ArgusII()
        self.model = AxonMapModel().build()
        self.master = master
        self.valid_electrodes = []
        self.selected_electrodes = set()

        master.title("Visual Percepts GUI")
        self.width = master.winfo_width()

        # Create left panel for Setup and Implant
        self.left_panel = tk.Frame(master, bg="white", width=self.width//4)
        self.left_panel.pack(side="left", fill="both", expand=True)

        # Create header and load button in Setup panel
        setup_header = tk.Label(self.left_panel, text="Setup", font=("Arial", 14), bg="white")
        setup_header.pack(padx=10, pady=10)

        load_button = tk.Button(self.left_panel, text="Load Dataset", command=self.load_file)
        load_button.pack(padx=10, pady=10)


        self.subject_stim_frame = tk.Frame(self.left_panel, bg="white")
        self.subject_stim_frame.pack(side="top", fill="both", expand=True)
        self.subject_frame = None
        self.display_subjects()
        self.stim_classes_frame = None
        self.display_stim_classes()

        # diplay stimulus parameter options
        self.stim_params = ['freq', 'amp1', 'pdur']
        self.stim_params_names = ['Frequency', 'Amplitude', 'Pulse Duration']
        self.stim_param_options = {s : [] for s in self.stim_params}
        self.stim_params_frame = tk.Frame(self.left_panel, bg='white')
        self.stim_params_frame.pack(side="top", fill="both", expand=True)
        self.stim_param_specific_frames = {s : None for s in self.stim_params}
        self.stim_param_vars = {s : {} for s in self.stim_params}
        self.stim_param_checkbuttons = {s : None for s in self.stim_params}
        for stim_param, stim_param_name in zip(self.stim_params, self.stim_params_names):
            self.diplay_stim_param(stim_param, stim_param_name)
        
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


    def display_subjects(self):
        # Create label and combobox for selecting subject
        if self.subject_frame is None:
            self.subject_frame = ttk.LabelFrame(self.subject_stim_frame, text="Subject", padding=(10, 10, 10, 0), height=200)
            self.subject_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        if not self.loaded_percepts:
            self.subjects = ["None"]
        else:
            self.subjects = list(self.dataset['subject'].unique())

        if not hasattr(self, 'subject') or self.subject.get() == 'None':
            self.subject = tk.StringVar(value=self.subjects[0])

        if self.subject_combobox is None:
            self.subject_combobox = ttk.Combobox(self.subject_frame, values=self.subjects, textvariable=self.subject)
            self.subject_combobox.pack(padx=10, pady=10)
            self.subject_combobox.bind("<<ComboboxSelected>>", self.update_current_dataset)
        else:
            self.subject_combobox['values'] = self.subjects
            self.subject_combobox['textvariable'] = self.subject
            if self.loaded_percepts:
                self.update_current_dataset(None, refresh=False)
    
    def display_stim_classes(self):
        if not self.loaded_percepts:
            self.stim_classes = []
        else:
            self.stim_classes = list(self.dataset['stim_class'].unique())
            # self.stim_classes = [i for i in self.stim_classes if 'Step' not in i and 'CDL' not in i]

        if self.stim_classes_frame is None:
            self.stim_classes_frame = ttk.LabelFrame(self.subject_stim_frame, text="Stim Class", padding=(10, 10, 10, 0), height=200)
            self.stim_classes_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        if not hasattr(self, 'stim_class_vars') or (len(self.stim_class_vars) == 0 and
                                                    self.loaded_percepts):
            # Create a tk.BooleanVar for each stim class
            self.stim_class_vars = {}
            for stim_class in self.stim_classes:
                var = tk.BooleanVar(value=True)
                self.stim_class_vars[stim_class] = var

            # Create a ttk.Checkbutton for each stim class
            self.stim_class_checkbuttons = {}
            for stim_class in self.stim_classes:
                checkbutton = ttk.Checkbutton(self.stim_classes_frame, text=f"{stim_class}", 
                                              variable=self.stim_class_vars[stim_class], command=self.update_current_dataset_no_event)
                checkbutton.pack(side="top", anchor="w", padx=10, pady=(5, 0))
                self.stim_class_checkbuttons[stim_class] = checkbutton

            if self.loaded_percepts:
                self.update_current_dataset(None, refresh=False)

        if self.loaded_percepts:
            for stim_class in self.stim_classes:
                num = len(self.current_dataset[self.current_dataset['stim_class'] == stim_class])
                self.stim_class_checkbuttons[stim_class].configure(text=f"{stim_class} ({num})")

    def diplay_stim_param(self, stim_param, name):
        if not self.loaded_percepts:
            self.stim_param_options[stim_param] = []
        else:
            self.stim_param_options[stim_param] = sorted(list(self.dataset[stim_param].unique()))

        if self.stim_param_specific_frames[stim_param] is None:
            self.stim_param_specific_frames[stim_param] = ttk.LabelFrame(self.stim_params_frame, text=name, padding=(10, 10, 10, 0), height=200)
            self.stim_param_specific_frames[stim_param].pack(side="left", fill="both", expand=True, padx=10, pady=10)

        if len(self.stim_param_vars[stim_param]) == 0 and self.loaded_percepts:
            # Create a tk.BooleanVar for each stim class
            self.stim_param_vars[stim_param] = {}
            for stim_param_option in self.stim_param_options[stim_param]:
                var = tk.BooleanVar(value=True)
                self.stim_param_vars[stim_param][stim_param_option] = var

            # Create a ttk.Checkbutton for each stim class
            self.stim_param_checkbuttons[stim_param] = {}
            for stim_param_option in self.stim_param_options[stim_param]:
                checkbutton = ttk.Checkbutton(self.stim_param_specific_frames[stim_param], text=f"{stim_param_option : .2f}", 
                                              variable=self.stim_param_vars[stim_param][stim_param_option], command=self.update_current_dataset_no_event)
                checkbutton.pack(side="top", anchor="w", padx=10, pady=(5, 0))
                self.stim_param_checkbuttons[stim_param][stim_param_option] = checkbutton

            if self.loaded_percepts:
                self.update_current_dataset(None, refresh=False)
        if self.loaded_percepts:
            for stim_param_option in self.stim_param_options[stim_param]:
                if self.stim_param_vars[stim_param][stim_param_option].get():
                    num = len(self.current_dataset[self.current_dataset[stim_param] == stim_param_option])
                else:
                    # select instead for how many there would be if this option was enabled.
                    # jk thats way too hard
                    num = "?"
                self.stim_param_checkbuttons[stim_param][stim_param_option].configure(text=f"{stim_param_option : .2f} ({num})")

    # overload for when there is no event
    def update_current_dataset_no_event(self, refresh=True):
        return self.update_current_dataset(None, refresh=refresh)

    def update_current_dataset(self, event, refresh=True):
        if self.dataset is None:
            print('Need to load a dataset first')
            return
        self.current_dataset = self.dataset

        #select by subject
        subject = self.subject.get()
        if subject is not None and subject != 'None':
            self.current_dataset = self.current_dataset[self.current_dataset['subject'] == subject]
            # update the implant representation
            if subject in subject_params:
                self.implant, self.model = model_from_params(subject_params[subject], biphasic=False)
        
        # select by stim class
        selected_stim_classes = [i for i in self.stim_class_vars.keys() if self.stim_class_vars[i].get()]
        if selected_stim_classes != ['None']:
            self.current_dataset = self.current_dataset[self.current_dataset['stim_class'].isin(selected_stim_classes)]

        # select by each of the stimulus parameters
        for stim_param in self.stim_params:
            if len(self.stim_param_options[stim_param]) > 0:
                selected_stim_param_options = [i for i in self.stim_param_vars[stim_param].keys() if self.stim_param_vars[stim_param][i].get()]
                if selected_stim_param_options != ['None']:
                    self.current_dataset = self.current_dataset[self.current_dataset[stim_param].isin(selected_stim_param_options)]

        # select by selected electrodes
        if len(self.selected_electrodes) > 0:
            def contains(row, elec):
                return elec in row['electrodes']
            for elec in self.selected_electrodes:
                self.current_dataset = self.current_dataset[self.current_dataset.apply(lambda row : contains(row, elec), axis=1)]

        # update valid electrodes
        self.valid_electrodes = list(self.current_dataset['electrode1'].unique())
        if refresh:
            self.refresh_options()


    def load_file(self):
        # Open a file dialog to select a file to load
        file_path = filedialog.askopenfilename()

        if file_path:
            self.percepts_filepath = file_path
            self.load_percepts()
    
    def load_percepts(self):
        if self.percepts_filepath is None:
            return
        try:
            self.dataset = load_shapes(self.percepts_filepath, stim_class=None, implant="ArgusII", combine=True)
            self.current_dataset = self.dataset
            print("loaded ", len(self.dataset), " percepts")
        except:
            print("Could not load percepts, is this the right file?")
            return
        self.loaded_percepts = True
        self.refresh_options()

    def refresh_options(self):
        self.display_subjects()
        self.display_stim_classes()
        for stim_param, name in zip(self.stim_params, self.stim_params_names):
            self.diplay_stim_param(stim_param, name)
        self.create_implant_representation()
        

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

        # Destroy the previous canvas if it exists
        if hasattr(self, 'implant_canvas'):
            self.implant_canvas.destroy()

        # Create a canvas to draw on
        canvas_width = 450
        canvas_height = 450
        r = 15
        self.implant_canvas = tk.Canvas(panel, width=canvas_width, height=canvas_height, bg='white', highlightthickness=0)
        self.implant_canvas.pack(expand=tk.YES, fill=tk.BOTH)

        # Draw each axon bundle as a curved line
        axon_bundles = self.model.grow_axon_bundles(n_bundles=100, prune=False)
        for bundle in axon_bundles[:25] + axon_bundles[-25:]:
            curve_points = []
            for i in range(bundle.shape[0]):
                # transform to scene coordinates
                x = (bundle[i, 0] - min_x) / resize * 400 + 25
                y = (bundle[i, 1] - min_y) / resize * 400 + r
                curve_points.append(x)
                curve_points.append(y)
            if len(curve_points) >= 4:
                self.implant_canvas.create_line(curve_points, smooth=True, width=2, fill='gray75')


        # Draw each electrode as a white circle with black border
        for e in electrode_names:
            # transform to scene coordinates
            x = (electrodes[e].x - min_x) / resize * 400 + 25 
            y = (electrodes[e].y - min_y) / resize * 400 + r
            
            if e in self.selected_electrodes:
                fill = 'green'
            elif self.valid_electrodes is None or e in self.valid_electrodes:
                fill = 'white'
            else:
                fill = 'gray'
            circle = self.implant_canvas.create_oval(x-r, y-r, x+r, y+r, outline='black', fill=fill)
            text = self.implant_canvas.create_text(x, y, text=e, font=("Arial", 10))

            # Bind the green color on hover event to the electrode circle and text
            def set_color_green(event, circle=circle, text=text):
                self.implant_canvas.itemconfig(circle, fill='green')
            
            def set_color_white(event, circle=circle, text=text):
                self.implant_canvas.itemconfig(circle, fill='white')

            def call_electrode_selected(event, circle=circle, text=text, e=e):
                if e in self.selected_electrodes:
                    self.electrode_deselected(e, circle)
                else:
                    self.electrode_selected(e, circle)

            if (self.valid_electrodes is None or e in self.valid_electrodes):
                if e not in self.selected_electrodes:
                    self.implant_canvas.tag_bind(circle, '<Enter>', set_color_green)
                    self.implant_canvas.tag_bind(circle, '<Leave>', set_color_white)
                    self.implant_canvas.tag_bind(text, '<Enter>', set_color_green)
                    self.implant_canvas.tag_bind(text, '<Leave>', set_color_white)
                self.implant_canvas.tag_bind(circle, '<Button-1>', call_electrode_selected)
                self.implant_canvas.tag_bind(text, '<Button-1>', call_electrode_selected)

    def electrode_selected(self, e, circle):
        print("calling select")
        self.selected_electrodes.add(e)
        self.update_current_dataset_no_event()
        # self.implant_canvas.itemconfig(circle, fill='green')
    
    def electrode_deselected(self, e, circle):
        print('calling deselect')
        self.selected_electrodes.remove(e)
        self.update_current_dataset_no_event()
        # self.implant_canvas.itemconfig(circle, fill='white')

root = tk.Tk()
gui = VisualPerceptsGUI(root)
root.mainloop()
