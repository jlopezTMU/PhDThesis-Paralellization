import tkinter as tk
from tkinter import ttk, scrolledtext
import subprocess

def run_simulation():
    # Collecting the values from the GUI inputs
    gpu = '--gpu' if gpu_var.get() else ''
    processors = processors_var.get()
    folds = folds_var.get()
    epochs = epochs_var.get()
    batch_size = batch_size_var.get()
    lr = lr_var.get()

    # Constructing the command
    command = f"python mainMASCNN.py {gpu} --processors {processors} --folds {folds} --epochs {epochs} --batch_size {batch_size} --lr {lr}"
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

def clear_output():
    output_text.delete(1.0, tk.END)
    progress_var.set(0)

# Main Window
root = tk.Tk()
root.title("DSMP Deep Learning Simulator")
root.geometry("600x400")

# Input Fields
gpu_var = tk.BooleanVar()
gpu_check = tk.Checkbutton(root, text="Use GPU", variable=gpu_var)
gpu_check.grid(row=0, column=0, sticky='w', padx=10, pady=10)

ttk.Label(root, text="Number of Processors:").grid(row=1, column=0, sticky='w', padx=10)
processors_var = tk.IntVar(value=2)
processors_menu = ttk.Combobox(root, textvariable=processors_var, values=[1, 2, 4, 8])
processors_menu.grid(row=1, column=1, padx=10)

ttk.Label(root, text="Number of Folds:").grid(row=2, column=0, sticky='w', padx=10)
folds_var = tk.IntVar(value=10)
folds_spinbox = tk.Spinbox(root, from_=1, to=20, textvariable=folds_var)
folds_spinbox.grid(row=2, column=1, padx=10)

ttk.Label(root, text="Number of Epochs:").grid(row=3, column=0, sticky='w', padx=10)
epochs_var = tk.IntVar(value=20)
epochs_spinbox = tk.Spinbox(root, from_=1, to=100, textvariable=epochs_var)
epochs_spinbox.grid(row=3, column=1, padx=10)

ttk.Label(root, text="Batch Size:").grid(row=4, column=0, sticky='w', padx=10)
batch_size_var = tk.IntVar(value=36)
batch_size_spinbox = tk.Spinbox(root, from_=1, to=256, textvariable=batch_size_var)
batch_size_spinbox.grid(row=4, column=1, padx=10)

ttk.Label(root, text="Learning Rate:").grid(row=5, column=0, sticky='w', padx=10)
lr_var = tk.StringVar(value="0.009")
lr_entry = ttk.Entry(root, textvariable=lr_var)
lr_entry.grid(row=5, column=1, padx=10)

# Command Display
command_display_var = tk.StringVar()
command_label = ttk.Label(root, textvariable=command_display_var, wraplength=500)
command_label.grid(row=6, column=0, columnspan=2, pady=10, padx=10)

# Output Area
output_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=10)
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
