import tkinter as tk
from tkinter import ttk, scrolledtext
import subprocess
import threading

# Function to run the simulation
def run_simulation():
    run_button.config(state=tk.DISABLED)
    root.config(cursor="watch")
    output_area.insert(tk.END, "Simulation is running...\n")
    output_area.see(tk.END)

    use_gpu = " --gpu" if gpu_var.get() else ""
    processors = processors_var.get()
    epochs = epochs_var.get()
    batch_size = batch_size_var.get()
    learning_rate = lr_var.get()
    latency = latency_var.get() or "1,2"
    patience = patience_var.get()

    cmd = f"python mainMASCNN.py{use_gpu} --processors {processors} --epochs {epochs} --batch_size {batch_size} --lr {learning_rate} --latency {latency} --patience {patience}"

    def run_process():
        try:
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                output_area.insert(tk.END, line)
                output_area.see(tk.END)
            process.wait()
        except Exception as e:
            output_area.insert(tk.END, f"Error: {str(e)}\n")
        finally:
            run_button.config(state=tk.NORMAL)
            root.config(cursor="")

    threading.Thread(target=run_process, daemon=True).start()

# Function to clear the output
def clear_output():
    output_area.delete('1.0', tk.END)

# Create the main window
root = tk.Tk()
root.title("Deep Learning Simulator (DLMP)")
root.configure(bg="#FFEEEE")

# Scrollable main frame
main_canvas = tk.Canvas(root, bg="#FFEEEE", highlightthickness=0)
main_scrollbar_y = ttk.Scrollbar(root, orient="vertical", command=main_canvas.yview)
main_scrollbar_x = ttk.Scrollbar(root, orient="horizontal", command=main_canvas.xview)
scrollable_frame = tk.Frame(main_canvas, bg="#FFEEEE")

scrollable_frame.bind(
    "<Configure>",
    lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
)
main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
main_canvas.configure(yscrollcommand=main_scrollbar_y.set, xscrollcommand=main_scrollbar_x.set)

main_canvas.pack(side="left", fill="both", expand=True)
main_scrollbar_y.pack(side="right", fill="y")
main_scrollbar_x.pack(side="bottom", fill="x")

# Titles
tk.Label(scrollable_frame, text="Deep Learning Simulator (DLMP)", bg="#FFEEEE", fg="red", font=("Arial", 16, "bold")).pack(pady=10)
tk.Label(scrollable_frame, text="Simulation Input Configuration", bg="#FFEEEE", fg="red", font=("Arial", 12, "underline")).pack()

# Input fields
input_frame = tk.Frame(scrollable_frame, bg="#FFEEEE")
input_frame.pack(pady=10, padx=10)

gpu_var = tk.BooleanVar()
gpu_check = tk.Checkbutton(input_frame, text="Use GPU?", variable=gpu_var, bg="#FFEEEE", fg="red")
gpu_check.grid(row=0, column=0, columnspan=2, sticky="w", pady=5)

def add_input(label, widget, row):
    tk.Label(input_frame, text=label, bg="#FFEEEE", fg="red").grid(row=row, column=0, sticky="w", padx=5, pady=5)
    widget.grid(row=row, column=1, sticky="we", padx=5, pady=5)

processors_var = tk.IntVar(value=1)
add_input("Number of Processors:", tk.Scale(input_frame, from_=1, to=16, orient="horizontal", variable=processors_var), 1)

epochs_var = tk.IntVar(value=16)
add_input("Number of Epochs:", tk.Scale(input_frame, from_=2, to=256, orient="horizontal", variable=epochs_var), 2)

batch_size_var = tk.IntVar(value=32)
add_input("Batch Size:", tk.Scale(input_frame, from_=2, to=256, orient="horizontal", variable=batch_size_var), 3)

lr_var = tk.StringVar(value="0.02")
add_input("Learning Rate:", tk.Entry(input_frame, textvariable=lr_var), 4)

latency_var = tk.StringVar(value="1,2")
add_input("Latency (Min, Max):", tk.Entry(input_frame, textvariable=latency_var), 5)

patience_var = tk.IntVar(value=3)
add_input("Patience:", tk.Scale(input_frame, from_=1, to=10, orient="horizontal", variable=patience_var), 6)

# Buttons
buttons_frame = tk.Frame(scrollable_frame, bg="#FFEEEE")
buttons_frame.pack(pady=10)

run_button = tk.Button(buttons_frame, text="RUN SIMULATION", command=run_simulation, bg="red", fg="white", width=20)
run_button.grid(row=0, column=0, padx=10)

clear_button = tk.Button(buttons_frame, text="CLEAR OUTPUT", command=clear_output, bg="red", fg="white", width=20)
clear_button.grid(row=0, column=1, padx=10)

# Output Section Title
tk.Label(scrollable_frame, text="Output of the Program", bg="#FFEEEE", fg="red", font=("Arial", 12, "underline")).pack()

# Scrollable Output Area
output_frame = tk.Frame(scrollable_frame, bg="#FFEEEE")
output_frame.pack(fill="both", expand=True, padx=10, pady=5)

output_area = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, width=90, height=20, bg="white", fg="black", borderwidth=1, relief="solid")
output_area.pack(fill="both", expand=True)

# Run main loop
root.mainloop()
