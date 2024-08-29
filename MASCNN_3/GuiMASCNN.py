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

    # Constructing the command
    command = f"python mainMASCNN.py {gpu} --processors {processors} --epochs {epochs} --batch_size {batch_size} --lr {lr}"
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

# Main Window
root = tk.Tk()
root.title("DSMP Deep Learning Simulator")

# Resize the window by 20% longer
root.geometry("780x720")  # Original size was 780x600

# Change background color to light red
root.configure(bg='#FFCCCC')

# Information Labels
info_text = (
    "Toronto Metropolitan University\n"
    "Director: Dr. Abdolreza Abhari\n"
    "Jorge Lopez PhD. student\n"
    "Department of Computer Science"
)
info_label = ttk.Label(root, text=info_text, background='#FFCCCC', font=("Arial", 10, "bold"), anchor='center')
info_label.grid(row=0, column=0, columnspan=2, pady=10, padx=10, sticky='we')  # Centered text

# Input Fields
gpu_var = tk.BooleanVar()
gpu_check = tk.Checkbutton(root, text="Use GPU", variable=gpu_var, bg='#FFCCCC')
gpu_check.grid(row=1, column=0, sticky='w', padx=10, pady=10)

ttk.Label(root, text="Number of Processors:", background='#FFCCCC').grid(row=2, column=0, sticky='w', padx=10)
processors_var = tk.IntVar(value=2)
processors_menu = ttk.Combobox(root, textvariable=processors_var, values=[1, 2, 4, 8])
processors_menu.grid(row=2, column=1, padx=10)

ttk.Label(root, text="Number of Epochs:", background='#FFCCCC').grid(row=3, column=0, sticky='w', padx=10)
epochs_var = tk.IntVar(value=20)
epochs_spinbox = tk.Spinbox(root, from_=1, to=100, textvariable=epochs_var)
epochs_spinbox.grid(row=3, column=1, padx=10)

ttk.Label(root, text="Batch Size:", background='#FFCCCC').grid(row=4, column=0, sticky='w', padx=10)
batch_size_var = tk.IntVar(value=36)
batch_size_spinbox = tk.Spinbox(root, from_=1, to=256, textvariable=batch_size_var)
batch_size_spinbox.grid(row=4, column=1, padx=10)

ttk.Label(root, text="Learning Rate:", background='#FFCCCC').grid(row=5, column=0, sticky='w', padx=10)
lr_var = tk.StringVar(value="0.009")
lr_entry = ttk.Entry(root, textvariable=lr_var)
lr_entry.grid(row=5, column=1, padx=10)

# Command Display
command_display_var = tk.StringVar()
command_label = ttk.Label(root, textvariable=command_display_var, wraplength=500, background='#FFCCCC')
command_label.grid(row=6, column=0, columnspan=2, pady=10, padx=10)

# Output Area (40% longer)
output_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=15)  # Increased height for larger output window
output_text.grid(row=7, column=0, columnspan=2, padx=10, pady=10)

# Progress Bar
progress_var = tk.DoubleVar(value=0)
progress_bar = ttk.Progressbar(root, maximum=100, variable=progress_var)
progress_bar.grid(row=8, column=0, columnspan=2, padx=10, pady=10, sticky="we")

# Action Buttons
run_button = ttk.Button(root, text="Run Simulation", command=run_simulation)
run_button.grid(row=9, column=0, padx=10, pady=10)

clear_button = ttk.Button(root, text="Clear Output", command=clear_output)
clear_button.grid(row=9, column=1, padx=10, pady=10)

# Run the GUI event loop
root.mainloop()
