#!/usr/bin/env python3
"""
Preprocessing program for SERS data
This module handles data preprocessing operations including:
- SG (Savitzky-Golay) filtering for spectral smoothing
- AirPLS baseline correction
- Batch processing of train and test datasets
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import signal
from scipy.sparse import diags, eye, csc_matrix
from scipy.sparse.linalg import spsolve
import os

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# Algorithm section
def SG(data, window_length, polyorder):
    return signal.savgol_filter(data, window_length, polyorder)

def WhittakerSmooth(x, w, lambda_, differences=1):
    X = np.matrix(x)
    m = X.size
    E = eye(m, format='csc')
    for i in range(differences):
        E = E[1:] - E[:-1]
    W = diags(w, 0, shape=(m, m))
    A = csc_matrix(W + (lambda_ * E.T * E))
    B = csc_matrix(W * X.T)
    background = spsolve(A, B)
    return np.array(background)

def airPLS(x, lambda_=1e8, porder=3, itermax=15):
    m = x.shape[0]
    w = np.ones(m)
    for i in range(1, itermax + 1):
        z = WhittakerSmooth(x, w, lambda_, porder)
        d = x - z
        dssn = np.abs(d[d < 0].sum())
        if (dssn < 0.001 * (abs(x)).sum() or i == itermax):
            if (i == itermax): print('WARING max iteration reached!')
            break
        w[d >= 0] = 0
        w[d < 0] = np.exp(i * np.abs(d[d < 0]) / dssn)
        w[0] = np.exp(i * (d[d < 0]).max() / dssn)
        w[-1] = w[0]
    return z

def AirPLS(data, lambda_, porder):
    data_process = data.copy()
    x = data.iloc[0].values  # First row as reference
    
    for i in range(1, data.shape[0]):  
        y = data.iloc[i].values
        merge1 = np.row_stack((y, x))  # Stack current row with first row
        sg = SG(merge1, 7, 3)[0]  # Apply SG filter, take first row result
        merge2 = np.row_stack((sg, x))  # Stack SG result with first row
        data_AirPLS = merge2.copy()
        
        # Apply AirPLS to the two stacked rows separately
        for j in range(merge2.shape[0]):
            data_AirPLS[j] = merge2[j] - airPLS(merge2[j], lambda_=lambda_, porder=porder)
        
        data_process.iloc[i] = data_AirPLS[0]  # Take first row as final result
    
    return data_process

class SpectralProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Raman Spectroscopy Processing Program")
        self.root.geometry("1000x1000")
        
        # Storage variables
        self.folder_path = tk.StringVar()
        self.window_length = tk.IntVar(value=7)
        self.polyorder = tk.IntVar(value=3)
        self.lambda_val = tk.DoubleVar(value=1e6)
        self.porder_val = tk.IntVar(value=3)
        self.prefix_name = tk.StringVar(value="t")
        self.save_folder_path = tk.StringVar()
        
        self.data = None
        self.merge_data = None
        self.all_data_processed = None
        self.folders = []
        self.current_preview_index = 0
        
        # Create main frame
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Show first page
        self.show_intro_page()
    
    def clear_frame(self):
        """Clear all content in the current frame"""
        for widget in self.main_frame.winfo_children():
            widget.destroy()
    
    def show_intro_page(self):
        """First page: Program introduction"""
        self.clear_frame()
        
        title_label = ttk.Label(self.main_frame, text="Spectroscopy Processing Program", font=("Arial", 16, "bold"))
        title_label.pack(pady=20)
        
        intro_text = """Welcome to Spectroscopy Processing Program! (LXT 2025.12 Updated)

This program provides the following features:
1. SG Filtering - Use Savitzky-Golay filter to smooth spectroscopy data
2. AirPLS Baseline Correction - Use Adaptive Iterative Reweighted Penalized Least Squares for baseline correction

Applicable data format:
Single spectrum CSV files saved by spectroscopy software.

Processing workflow:
1. Select relevant folders; spectra should be organized in the following format:
    Root Folder/
         ├── Folder 1/
         │     ├── Data File 1.csv
         │     ├── Data File 2.csv
         │     └── ...
         ├── Folder 2/
         │     ├── Data File 1.csv
         │     ├── Data File 2.csv
         │     └── ...
         └── ...
2. Configure SG filter parameters and preview results
3. Configure AirPLS parameters and preview results
4. Confirm parameters and batch process all files
5. Save processed data

Note: The first column of spectra is Raman shift values, the second column is intensity values.

Click the "Start" button to continue...
        """
        
        intro_label = ttk.Label(self.main_frame, text=intro_text, justify=tk.LEFT)
        intro_label.pack(pady=20, padx=20)
        
        start_button = ttk.Button(self.main_frame, text="Start", command=self.show_folder_selection_page)
        start_button.pack(pady=10)
    
    def show_folder_selection_page(self):
        """Second page: Folder selection"""
        self.clear_frame()
        
        title_label = ttk.Label(self.main_frame, text="Select Data Folder", font=("Arial", 14, "bold"))
        title_label.pack(pady=20)
        
        instruction_label = ttk.Label(self.main_frame, text="Please select a folder containing data files:")
        instruction_label.pack(pady=10)
        
        # Folder selection frame
        folder_frame = ttk.Frame(self.main_frame)
        folder_frame.pack(pady=10, fill=tk.X, padx=50)
        
        folder_entry = ttk.Entry(folder_frame, textvariable=self.folder_path, width=50)
        folder_entry.pack(side=tk.LEFT, padx=(0, 10), fill=tk.X, expand=True)
        
        browse_button = ttk.Button(folder_frame, text="Browse", command=self.browse_folder)
        browse_button.pack(side=tk.LEFT)
        
        # Button frame
        button_frame = ttk.Frame(self.main_frame)
        button_frame.pack(pady=20)
        
        back_button = ttk.Button(button_frame, text="Previous", command=self.show_intro_page)
        back_button.pack(side=tk.LEFT, padx=10)
        
        next_button = ttk.Button(button_frame, text="Next", command=self.load_data_for_preview)
        next_button.pack(side=tk.LEFT, padx=10)
    
    def browse_folder(self):
        """Browse folders"""
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.folder_path.set(folder_selected)
    
    def load_data_for_preview(self):
        """Load data for preview"""
        if not self.folder_path.get():
            messagebox.showerror("Error", "Please select a folder first!")
            return
        
        try:
            # Get first folder in directory
            all_folders = [f for f in os.listdir(self.folder_path.get()) if os.path.isdir(os.path.join(self.folder_path.get(), f))]
            if not all_folders:
                messagebox.showerror("Error", "No subfolders found in the selected folder!")
                return
            
            self.folders = all_folders
            
            # Load first file for preview
            first_file_path = os.path.join(self.folder_path.get(), self.folders[0])
            all_files = os.listdir(first_file_path)
            csv_files = [f for f in all_files if f.lower().endswith('.csv')]
            first_file_path = os.path.join(first_file_path, csv_files[0])
            self.data = pd.read_csv(first_file_path, sep=',', skiprows=[0], names=['Raman Shift', 'Intensity'], encoding='GBK')           
            
            if self.data.shape[1] <= 1:
                messagebox.showerror("Error", "Files need at least 2 columns of data for processing!")
                return
            
            messagebox.showinfo("Success", f"Found {len(self.folders)} folders, loaded spectrum {self.current_preview_index+1} from the first folder for preview.")
            
            # Go to SG filter page
            self.show_sg_filter_page()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading data: {str(e)}")
    
    def show_sg_filter_page(self):
        """Third page: SG filter parameter settings and preview"""
        self.clear_frame()
        
        title_label = ttk.Label(self.main_frame, text="SG Filter Parameter Settings", font=("Arial", 14, "bold"))
        title_label.pack(pady=20)
        
        # Info label
        info_label = ttk.Label(self.main_frame, text=f"Current preview: Spectrum {self.current_preview_index + 1} from {self.folders[0]}", 
                              foreground="blue")
        info_label.pack(pady=5)
        
        # Parameter input frame
        param_frame = ttk.LabelFrame(self.main_frame, text="SG Filter Parameters")
        param_frame.pack(pady=10, padx=50, fill=tk.X)
        
        # Window length
        ttk.Label(param_frame, text="Window length (window_length):").grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
        window_spinbox = ttk.Spinbox(param_frame, from_=3, to=51, textvariable=self.window_length, width=10)
        window_spinbox.grid(row=0, column=1, padx=10, pady=10)
        
        # Polynomial order
        ttk.Label(param_frame, text="Polynomial order (polyorder):").grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)
        poly_spinbox = ttk.Spinbox(param_frame, from_=1, to=10, textvariable=self.polyorder, width=10)
        poly_spinbox.grid(row=1, column=1, padx=10, pady=10)
        
        # Button frame
        preview_frame = ttk.Frame(self.main_frame)
        preview_frame.pack(pady=10)
        
        preview_button = ttk.Button(preview_frame, text="Preview SG Effect", command=self.preview_sg_filter)
        preview_button.pack(side=tk.LEFT, padx=5)
        
        confirm_sg_button = ttk.Button(preview_frame, text="Confirm SG Parameters and Continue", command=self.confirm_sg_parameters)
        confirm_sg_button.pack(side=tk.LEFT, padx=5)
        
        # Chart frame
        self.chart_frame = ttk.Frame(self.main_frame)
        self.chart_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # Navigation button frame
        nav_frame = ttk.Frame(self.main_frame)
        nav_frame.pack(pady=10)
        
        back_button = ttk.Button(nav_frame, text="Previous", command=self.show_folder_selection_page)
        back_button.pack(side=tk.LEFT, padx=10)
    
    def preview_sg_filter(self):
        try:
            # Clear previous chart
            for widget in self.chart_frame.winfo_children():
                widget.destroy()
            
            # Process according to logic: first column as reference, second column stacked with first
            x = self.data.iloc[:, 0].values  # First column as reference
            y = self.data.iloc[:, self.current_preview_index+1].values  # Current preview column
            
            # Stack and apply SG filter
            merge1 = np.row_stack((y, x))
            window_length = self.window_length.get()
            polyorder = self.polyorder.get()
            
            # Ensure window length is odd
            if window_length % 2 == 0:
                window_length += 1
                self.window_length.set(window_length)
            
            sg_result = SG(merge1, window_length, polyorder)
            sg_filtered = sg_result[0]  # Take first row as SG filter result
            
            # Create chart
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(x[100:130], y[100:130], label='Original Spectrum', alpha=0.7)
            ax.plot(x[100:130], sg_filtered[100:130], label='After SG Filter', alpha=0.8)
            ax.set_title('SG Filter Effect Comparison')
            ax.set_xlabel('Raman Shift')
            ax.set_ylabel('Intensity')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Display chart in tkinter
            canvas = FigureCanvasTkAgg(fig, self.chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error generating preview: {str(e)}")
    
    def confirm_sg_parameters(self):
        """Confirm SG parameters and proceed to AirPLS page"""
        if not hasattr(self, 'chart_frame') or not self.chart_frame.winfo_children():
            messagebox.showwarning("Warning", "Please preview the effect first before confirming parameters!")
            return

        self.show_airpls_page()
    
    def show_airpls_page(self):
        """Fourth page: AirPLS parameter settings and preview"""
        self.clear_frame()
        
        title_label = ttk.Label(self.main_frame, text="AirPLS Baseline Correction", font=("Arial", 14, "bold"))
        title_label.pack(pady=20)
        
        # Info label
        info_label = ttk.Label(self.main_frame, text=f"Current preview: Spectrum {self.current_preview_index + 1} from {self.folders[0]} (Complete processing workflow)", 
                              foreground="blue")
        info_label.pack(pady=5)

        # Parameter input frame
        param_frame = ttk.LabelFrame(self.main_frame, text="AirPLS Parameters")
        param_frame.pack(pady=10, padx=50, fill=tk.X)
        
        # Lambda parameter
        ttk.Label(param_frame, text="Lambda value:").grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
        lambda_entry = ttk.Entry(param_frame, textvariable=self.lambda_val, width=15)
        lambda_entry.grid(row=0, column=1, padx=10, pady=10)
        
        # Porder parameter
        ttk.Label(param_frame, text="Porder value:").grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)
        porder_spinbox = ttk.Spinbox(param_frame, from_=1, to=10, textvariable=self.porder_val, width=10)
        porder_spinbox.grid(row=1, column=1, padx=10, pady=10)
        
        # Button frame
        preview_frame = ttk.Frame(self.main_frame)
        preview_frame.pack(pady=10)
        
        preview_button = ttk.Button(preview_frame, text="Preview Complete Effect", command=self.preview_complete_processing)
        preview_button.pack(side=tk.LEFT, padx=5)
        
        confirm_airpls_button = ttk.Button(preview_frame, text="Confirm All Parameters and Process Files", command=self.process_all_files)
        confirm_airpls_button.pack(side=tk.LEFT, padx=5)
        
        # Chart frame
        self.chart_frame_airpls = ttk.Frame(self.main_frame)
        self.chart_frame_airpls.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # Navigation button frame
        nav_frame = ttk.Frame(self.main_frame)
        nav_frame.pack(pady=10)
        
        back_button = ttk.Button(nav_frame, text="Previous", command=self.show_sg_filter_page)
        back_button.pack(side=tk.LEFT, padx=10)
    
    def preview_complete_processing(self):
        """Preview the complete processing workflow effect"""
        try:
            # Clear previous chart
            for widget in self.chart_frame_airpls.winfo_children():
                widget.destroy()
            
            # Follow complete processing logic
            x = self.data.iloc[:, 0].values  # First column as reference
            y = self.data.iloc[:, self.current_preview_index+1].values  # Current preview column
            
            # SG filter step
            merge1 = np.row_stack((y, x))
            window_length = self.window_length.get()
            polyorder = self.polyorder.get()
            
            # Ensure window length is odd
            if window_length % 2 == 0:
                window_length += 1
            
            sg_result = SG(merge1, window_length, polyorder)
            sg_filtered = sg_result[0]  # SG filter result
            
            # AirPLS step
            merge2 = np.row_stack((sg_filtered, x))
            lambda_val = self.lambda_val.get()
            porder_val = self.porder_val.get()
            
            # Apply AirPLS to the two stacked rows separately
            data_AirPLS = merge2.copy()
            for j in range(merge2.shape[0]):
                data_AirPLS[j] = merge2[j] - airPLS(merge2[j], lambda_=lambda_val, porder=porder_val)
            
            final_result = data_AirPLS[0]  # Final result
            
            # Create chart
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Left: SG filter effect
            ax1.plot(x, y, label='Original Spectrum', alpha=0.7)
            ax1.plot(x, sg_filtered, label='After SG Filter', alpha=0.8)
            ax1.set_title('Spectrum After SG Filter')
            ax1.set_xlabel("Raman Shift")
            ax1.set_ylabel('Intensity')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Right: Complete processing effect
            ax2.plot(x, y, label='Original Spectrum', alpha=0.7)
            ax2.plot(x, y-final_result, label="Baseline", alpha=0.7)
            ax2.plot(x, final_result, label='Spectrum After Baseline Correction', alpha=0.8)
            ax2.set_title('Complete Processing Effect (SG + AirPLS)')
            ax2.set_xlabel('Raman Shift')
            ax2.set_ylabel('Intensity')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Display chart in tkinter
            canvas = FigureCanvasTkAgg(fig, self.chart_frame_airpls)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error generating preview: {str(e)}")
    
    def process_all_files(self):
        """Process all files according to the logic"""
        if not hasattr(self, 'chart_frame_airpls') or not self.chart_frame_airpls.winfo_children():
            messagebox.showwarning("Warning", "Please preview the effect first before processing files!")
            return
        
        try:
            # Show processing progress
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Processing")
            progress_window.geometry("300x100")
            progress_window.transient(self.root)
            progress_window.grab_set()
            
            ttk.Label(progress_window, text="Processing files, please wait...").pack(pady=10)
            progress = ttk.Progressbar(progress_window, mode='indeterminate')
            progress.pack(pady=10, padx=20, fill=tk.X)
            progress.start()
            
            self.root.update()
            
            # Process all files
            self.all_data_processed = {}
            self.merge_data = pd.DataFrame()
            window_length = self.window_length.get()
            polyorder = self.polyorder.get()
            lambda_val = self.lambda_val.get()
            porder_val = self.porder_val.get()
            
            # Ensure window length is odd
            if window_length % 2 == 0:
                window_length += 1
            
            for folder in self.folders:
                folder_path = os.path.join(self.folder_path.get(), folder)
                processed_files = {}
                calculate_mean_spec = pd.DataFrame()
                for file in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file)
                    data = pd.read_csv(file_path, sep=',', skiprows=[0], names=['Raman Shift', 'Intensity'], encoding='GBK')       
                
                    # Process according to logic
                    data_process = data.copy()
                    x = data.iloc[:, 0].values
                    if self.merge_data.empty:
                        self.merge_data['Raman Shift'] = x
                    
                    y = data.iloc[:, 1].values
                    merge1 = np.row_stack((y, x))
                    sg = SG(merge1, window_length, polyorder)[0]
                    merge2 = np.row_stack((sg, x))
                        
                    # Apply AirPLS to the two stacked rows separately
                    data_AirPLS = merge2.copy()
                    for j in range(merge2.shape[0]):
                        data_AirPLS[j] = merge2[j] - airPLS(merge2[j], lambda_=lambda_val, porder=porder_val)

                    processed_data = pd.DataFrame({'Raman Shift': x, 'Processed Intensity': data_AirPLS[0]})
                    processed_files[file] = processed_data
                    calculate_mean_spec = pd.concat([calculate_mean_spec, processed_data['Processed Intensity']], axis=1)
                
                mean_spec = calculate_mean_spec.mean(axis=1)
                self.merge_data[folder] = mean_spec.values
                self.all_data_processed[folder] = processed_files
            
            progress.stop()
            progress_window.destroy()
            
            messagebox.showinfo("Success", f"Completed processing of all {len(self.folders)} folders!")

            # Go to save page
            self.show_save_page()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error processing files: {str(e)}")
    
    def show_save_page(self):
        """Fifth page: Save settings"""
        self.clear_frame()
        
        title_label = ttk.Label(self.main_frame, text="Save Processing Results", font=("Arial", 14, "bold"))
        title_label.pack(pady=20)
        
        # Prefix name input
        prefix_frame = ttk.Frame(self.main_frame)
        prefix_frame.pack(pady=10, padx=50, fill=tk.X)
        
        ttk.Label(prefix_frame, text="File prefix:").pack(side=tk.LEFT)
        prefix_entry = ttk.Entry(prefix_frame, textvariable=self.prefix_name, width=20)
        prefix_entry.pack(side=tk.LEFT, padx=(10, 0))
        
        # Save folder selection
        save_folder_frame = ttk.Frame(self.main_frame)
        save_folder_frame.pack(pady=10, padx=50, fill=tk.X)
        
        ttk.Label(save_folder_frame, text="Save folder:").pack(side=tk.LEFT)
        save_folder_entry = ttk.Entry(save_folder_frame, textvariable=self.save_folder_path, width=40)
        save_folder_entry.pack(side=tk.LEFT, padx=(10, 10), fill=tk.X, expand=True)
        
        browse_save_button = ttk.Button(save_folder_frame, text="Browse", command=self.browse_save_folder)
        browse_save_button.pack(side=tk.LEFT)
        
        # Button frame
        button_frame = ttk.Frame(self.main_frame)
        button_frame.pack(pady=20)
        
        back_button = ttk.Button(button_frame, text="Previous", command=self.show_airpls_page)
        back_button.pack(side=tk.LEFT, padx=10)
        
        save_button = ttk.Button(button_frame, text="Save Results", command=self.save_results)
        save_button.pack(side=tk.LEFT, padx=10)
    
    def browse_save_folder(self):
        """Browse save folder"""
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.save_folder_path.set(folder_selected)
    
    def save_results(self):
        """Save processing results"""
        if not self.save_folder_path.get():
            messagebox.showerror("Error", "Please select a save folder!")
            return
        
        if not self.prefix_name.get():
            messagebox.showerror("Error", "Please enter a file prefix!")
            return
        
        try:
            # Save processed data
            for i, (folder, processed_data) in enumerate(self.all_data_processed.items()):
                # Create subfolder named after folder
                folder_save_path = os.path.join(self.save_folder_path.get(), folder)
                os.makedirs(folder_save_path, exist_ok=True)

                for file_name, data in processed_data.items():
                    csv_file = file_name
                    save_path = os.path.join(folder_save_path, f"{self.prefix_name.get()}{os.path.splitext(csv_file)[0]}.csv")
                    data.to_csv(save_path, index=False)
            
            messagebox.showinfo("Success", f"Processing complete! {len(self.all_data_processed)} folders saved to: {self.save_folder_path.get()}")

            # Save all_data_processed as a merged file
            merged_save_path = os.path.join(self.save_folder_path.get(), f"{self.prefix_name.get()}_merged_results.csv")
            self.merge_data.to_csv(merged_save_path, index=False)

            
        except Exception as e:
            messagebox.showerror("Error", f"Error saving files: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SpectralProcessorApp(root)
    root.mainloop()