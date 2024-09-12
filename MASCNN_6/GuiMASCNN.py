import tkinter as tk
from tkinter import ttk, scrolledtext
import subprocess
from threading import Thread

def run_simulation():
    # Gray out the window and display the working icon
    run_button.config(state=tk.DISABLED)
    clear_button.config(state=tk.DISABLED)
    root.config(cursor="wait")
    root.update_idletasks()

    # Run the simulation in a separate thread to keep the GUI responsive
    simulation_thread = Thread(target=run_simulation_thread)
    simulation_thread.start()

def run_simulation_thread():
    # Collecting the values from the GUI inputs
    gpu = '--gpu' if gpu_var.get() else ''
    processors = processors_var.get()
    epochs = epochs_var.get()
    batch_size = batch_size_var.get()
    lr = lr_var.get()
    patience = patience_var.get()

    # Get the latency values
    try:
        latency_lower = int(latency_lower_var.get())
        latency_upper = int(latency_upper_var.get())

        # Validate latency values
        if latency_lower <= 0 or latency_upper <= 0:
            raise ValueError("Latency values must be positive integers.")
        if latency_lower >= latency_upper:
            raise ValueError("Lower latency must be less than upper latency.")

        latency = f"--latency {latency_lower},{latency_upper}"
    except ValueError as e:
        output_text.insert(tk.END, f"Error: {str(e)}\n")
        run_button.config(state=tk.NORMAL)
        clear_button.config(state=tk.NORMAL)
        root.config(cursor="")
        return

    # Constructing the command
    command = f"python mainMASCNN.py {gpu} --processors {processors} --epochs {epochs} --batch_size {batch_size} --lr {lr} --patience {patience} {latency}"
    command_display_var.set(command)

    # Running the command and capturing the output
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        output, error = process.communicate()

        if process.returncode == 0:
            output_text.insert(tk.END, output.decode())
        else:
            output_text.insert(tk.END, error.decode())
    except Exception as e:
        output_text.insert(tk.END, f"Error: {str(e)}")

    # Update progress bar (Simulated, adjust as needed)
    progress_var.set(100)

    # Restore the window and stop the working icon
    run_button.config(state=tk.NORMAL)
    clear_button.config(state=tk.NORMAL)
    root.config(cursor="")

def clear_output():
    output_text.delete(1.0, tk.END)
    progress_var.set(0)

# Main Window Setup
root = tk.Tk()
root.title("DLMP Simulator")

# Set the window size to 600px wide and adjust height proportionally
window_width = 600
window_height = int(root.winfo_screenheight() * 0.65)
root.geometry(f"{window_width}x{window_height}")

# Adjust the canvas to fill the window
canvas = tk.Canvas(root, bg='#FFCCCC')  # Set canvas background to light red
scroll_y = tk.Scrollbar(root, orient="vertical", command=canvas.yview)

# Adjust the width of the frame to fill the available space
frame = tk.Frame(canvas, bg='#FFCCCC')
frame.grid_columnconfigure(0, weight=1)  # Make the column expandable
frame.grid_columnconfigure(1, weight=1)  # Make the second column expandable

# Information Labels
info_text = (
    "Toronto Metropolitan University\n"
    "Distributed Systems and Multimedia Processing\n"
    "Director: Dr. Abdolreza Abhari\n"
    "Developer: Jorge Lopez\n"
    "Department of Computer Science"
)
info_label = ttk.Label(frame, text=info_text, background='#FFCCCC', font=("Arial", 10, "bold"), anchor='center')
info_label.grid(row=0, column=0, columnspan=2, pady=10, padx=10, sticky='ew')  # Centered text with expand

# Input Fields and Configuration
gpu_var = tk.BooleanVar()
gpu_check = tk.Checkbutton(frame, text="Use GPU", variable=gpu_var, bg='#FFCCCC')
gpu_check.grid(row=1, column=0, columnspan=2, sticky='w', padx=10, pady=10)

# Sliders and Fields organized in two columns
ttk.Label(frame, text="Processors:", background='#FFCCCC').grid(row=2, column=0, sticky='w', padx=10)
processors_var = tk.IntVar(value=2)
processors_scale = tk.Scale(frame, from_=1, to=8, variable=processors_var, orient="horizontal")
processors_scale.grid(row=2, column=1, padx=10, pady=10, sticky='ew')

ttk.Label(frame, text="Epochs:", background='#FFCCCC').grid(row=3, column=0, sticky='w', padx=10)
epochs_var = tk.IntVar(value=20)
epochs_scale = tk.Scale(frame, from_=1, to=100, variable=epochs_var, orient="horizontal")
epochs_scale.grid(row=3, column=1, padx=10, pady=10, sticky='ew')

ttk.Label(frame, text="Batch Size:", background='#FFCCCC').grid(row=4, column=0, sticky='w', padx=10)
batch_size_var = tk.IntVar(value=32)
batch_size_scale = tk.Scale(frame, from_=2, to=4096, resolution=2, variable=batch_size_var, orient="horizontal")
batch_size_scale.grid(row=4, column=1, padx=10, pady=10, sticky='ew')

ttk.Label(frame, text="Learning Rate:", background='#FFCCCC').grid(row=5, column=0, sticky='w', padx=10)
lr_var = tk.StringVar(value="0.009")
lr_entry = ttk.Entry(frame, textvariable=lr_var)
lr_entry.grid(row=5, column=1, padx=10, sticky='ew')

ttk.Label(frame, text="Patience:", background='#FFCCCC').grid(row=6, column=0, sticky='w', padx=10)
patience_var = tk.IntVar(value=5)
patience_scale = tk.Scale(frame, from_=1, to=50, variable=patience_var, orient="horizontal")
patience_scale.grid(row=6, column=1, padx=10, pady=10, sticky='ew')

# Latency Input Fields in two columns
ttk.Label(frame, text="Latency Lower (ms):", background='#FFCCCC').grid(row=7, column=0, sticky='w', padx=10)
latency_lower_var = tk.StringVar(value="10")
latency_lower_entry = ttk.Entry(frame, textvariable=latency_lower_var)
latency_lower_entry.grid(row=7, column=1, padx=10, pady=10, sticky='ew')

ttk.Label(frame, text="Latency Upper (ms):", background='#FFCCCC').grid(row=8, column=0, sticky='w', padx=10)
latency_upper_var = tk.StringVar(value="100")
latency_upper_entry = ttk.Entry(frame, textvariable=latency_upper_var)
latency_upper_entry.grid(row=8, column=1, padx=10, pady=10, sticky='ew')

# Action Buttons - Separate row for buttons
run_button = ttk.Button(frame, text="Run Simulation", command=run_simulation)
run_button.grid(row=9, column=0, padx=10, pady=10, sticky='ew')

clear_button = ttk.Button(frame, text="Clear Output", command=clear_output)
clear_button.grid(row=9, column=1, padx=10, pady=10, sticky='ew')

# Command Display and Progress Bar
command_display_var = tk.StringVar()
command_label = ttk.Label(frame, textvariable=command_display_var, wraplength=500, background='#FFCCCC')
command_label.grid(row=10, column=0, columnspan=2, pady=10, padx=10, sticky='ew')

progress_var = tk.DoubleVar(value=0)
progress_bar = ttk.Progressbar(frame, maximum=100, variable=progress_var)
progress_bar.grid(row=11, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

# Output Area (back at the bottom, matching the window width)
output_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=10)  # Adjusted height for larger output window
output_text.pack(side="bottom", fill="both", padx=10, pady=10)

# Packing and Scrolling Configuration for the main frame (inputs on top)
canvas.create_window(0, 0, anchor='nw', window=frame)
canvas.update_idletasks()
canvas.configure(scrollregion=canvas.bbox('all'), yscrollcommand=scroll_y.set)

canvas.pack(fill='both', expand=True, side='top')
scroll_y.pack(fill='y', side='right')

# Run the GUI event loop
root.mainloop()
