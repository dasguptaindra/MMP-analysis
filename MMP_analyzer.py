# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdMMPA import FragmentMol  # Use RDKit's FragmentMol directly
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk
import numpy as np
from itertools import combinations
from operator import itemgetter
from io import BytesIO
import base64
import requests
import sys
import os
import threading
import warnings
import re
warnings.filterwarnings('ignore')

class MMPAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Matched Molecular Pairs Analyzer")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2b2b2b')
        
        # Variables
        self.df = None
        self.delta_df = None
        self.mmp_df = None
        self.current_idx = 0
        self.min_transform_occurrence = 5
        
        # Create GUI
        self.create_widgets()
        
    def setup_styles(self):
        """Configure custom styles for the GUI"""
        self.colors = {
            'bg': '#2b2b2b',
            'fg': '#ffffff',
            'accent': '#4CAF50',
            'secondary': '#2196F3',
            'warning': '#FF9800',
            'error': '#F44336',
            'card': '#3c3f41',
            'text': '#bbbbbb'
        }
        
        # Configure ttk styles
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure styles for different widgets
        style.configure('Title.TLabel', 
                       background=self.colors['bg'],
                       foreground=self.colors['accent'],
                       font=('Arial', 16, 'bold'))
        
        style.configure('Header.TLabel',
                       background=self.colors['card'],
                       foreground=self.colors['fg'],
                       font=('Arial', 12, 'bold'))
        
        style.configure('Custom.TButton',
                       background=self.colors['secondary'],
                       foreground=self.colors['fg'],
                       borderwidth=1,
                       focuscolor='none')
        
        style.map('Custom.TButton',
                 background=[('active', self.colors['accent'])])
        
        style.configure('Custom.TEntry',
                       fieldbackground=self.colors['card'],
                       foreground=self.colors['fg'],
                       borderwidth=1)
        
        style.configure('Custom.TCombobox',
                       fieldbackground=self.colors['card'],
                       foreground=self.colors['fg'],
                       background=self.colors['card'])
        
        # Configure progress bar style
        style.configure("green.Horizontal.TProgressbar",
                       background=self.colors['accent'],
                       troughcolor=self.colors['card'])
        
    def create_widgets(self):
        """Create all GUI widgets"""
        # Setup styles first
        self.setup_styles()
        
        # Create main container
        main_container = tk.Frame(self.root, bg=self.colors['bg'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_frame = tk.Frame(main_container, bg=self.colors['bg'])
        title_frame.pack(fill=tk.X, pady=(0, 20))
        
        title_label = tk.Label(title_frame, 
                              text="Matched Molecular Pairs Analyzer",
                              font=('Arial', 24, 'bold'),
                              bg=self.colors['bg'],
                              fg=self.colors['accent'])
        title_label.pack()
        
        subtitle_label = tk.Label(title_frame,
                                 text="Analyze SAR using matched molecular pairs",
                                 font=('Arial', 10),
                                 bg=self.colors['bg'],
                                 fg=self.colors['text'])
        subtitle_label.pack()
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.setup_data_tab()
        self.setup_analysis_tab()
        self.setup_results_tab()
        self.setup_visualization_tab()
        self.setup_compounds_tab()  # New tab for compound details
        
        # Status bar
        self.status_bar = tk.Label(self.root, 
                                  text="Ready",
                                  bd=1, 
                                  relief=tk.SUNKEN, 
                                  anchor=tk.W,
                                  bg=self.colors['card'],
                                  fg=self.colors['fg'])
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def setup_data_tab(self):
        """Setup data loading and preprocessing tab"""
        data_tab = tk.Frame(self.notebook, bg=self.colors['bg'])
        self.notebook.add(data_tab, text="Data Input")
        
        # Control frame
        control_frame = tk.Frame(data_tab, bg=self.colors['bg'])
        control_frame.pack(fill=tk.X, padx=20, pady=20)
        
        # Load data section
        load_frame = tk.LabelFrame(control_frame, 
                                  text="Load Data",
                                  bg=self.colors['card'],
                                  fg=self.colors['fg'],
                                  font=('Arial', 12, 'bold'),
                                  padx=15,
                                  pady=15)
        load_frame.pack(fill=tk.X, pady=(0, 20))
        
        # File upload (only upload button, no URL input)
        file_frame = tk.Frame(load_frame, bg=self.colors['card'])
        file_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(file_frame,
                  text="Upload CSV File",
                  command=self.load_csv_file,
                  style='Custom.TButton').pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(file_frame,
                  text="Export Results",
                  command=self.export_results,
                  style='Custom.TButton').pack(side=tk.LEFT)
        
        # Data format information
        format_frame = tk.Frame(load_frame, bg=self.colors['card'])
        format_frame.pack(fill=tk.X, pady=10)
        
        format_label = tk.Label(format_frame,
                               text="Required CSV format: SMILES, Name/ID, pIC50",
                               bg=self.colors['card'],
                               fg=self.colors['warning'],
                               font=('Arial', 9, 'italic'))
        format_label.pack()
        
        example_label = tk.Label(format_frame,
                                text="Example:\nCC(=O)Oc1ccccc1C(=O)O,aspirin,4.2",
                                bg=self.colors['card'],
                                fg=self.colors['text'],
                                font=('Courier', 8))
        example_label.pack()
        
        # Data info frame
        info_frame = tk.LabelFrame(control_frame,
                                  text="Data Information",
                                  bg=self.colors['card'],
                                  fg=self.colors['fg'],
                                  font=('Arial', 12, 'bold'),
                                  padx=15,
                                  pady=15)
        info_frame.pack(fill=tk.X)
        
        # Text widget for data info
        self.data_info_text = scrolledtext.ScrolledText(info_frame,
                                                       height=10,
                                                       bg=self.colors['card'],
                                                       fg=self.colors['fg'],
                                                       insertbackground=self.colors['fg'])
        self.data_info_text.pack(fill=tk.BOTH, expand=True)
        
        # Preprocessing options
        options_frame = tk.LabelFrame(control_frame,
                                     text="Preprocessing Options",
                                     bg=self.colors['card'],
                                     fg=self.colors['fg'],
                                     font=('Arial', 12, 'bold'),
                                     padx=15,
                                     pady=15)
        options_frame.pack(fill=tk.X, pady=(20, 0))
        
        # Minimum transform occurrence
        tk.Label(options_frame,
                text="Minimum Transform Occurrence:",
                bg=self.colors['card'],
                fg=self.colors['fg']).grid(row=0, column=0, padx=(0, 10))
        
        self.min_occurrence_var = tk.IntVar(value=5)
        min_occurrence_spin = ttk.Spinbox(options_frame,
                                         from_=1,
                                         to=50,
                                         textvariable=self.min_occurrence_var,
                                         width=10)
        min_occurrence_spin.grid(row=0, column=1, padx=(0, 20))
        
        # Run analysis button
        ttk.Button(options_frame,
                  text="Run MMP Analysis",
                  command=self.run_analysis,
                  style='Custom.TButton').grid(row=0, column=2, padx=20)
        
    def setup_analysis_tab(self):
        """Setup analysis tab"""
        analysis_tab = tk.Frame(self.notebook, bg=self.colors['bg'])
        self.notebook.add(analysis_tab, text="MMP Analysis")
        
        # Control frame
        control_frame = tk.Frame(analysis_tab, bg=self.colors['bg'])
        control_frame.pack(fill=tk.X, padx=20, pady=20)
        
        # MMP selection
        selection_frame = tk.LabelFrame(control_frame,
                                       text="Select MMP Transform",
                                       bg=self.colors['card'],
                                       fg=self.colors['fg'],
                                       font=('Arial', 12, 'bold'),
                                       padx=15,
                                       pady=15)
        selection_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Transform list
        list_frame = tk.Frame(selection_frame, bg=self.colors['card'])
        list_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Listbox for transforms
        self.transform_listbox = tk.Listbox(list_frame,
                                           bg=self.colors['card'],
                                           fg=self.colors['fg'],
                                           selectbackground=self.colors['secondary'],
                                           selectforeground=self.colors['fg'],
                                           font=('Courier', 10))
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.transform_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.transform_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.transform_listbox.yview)
        
        # Bind selection event
        self.transform_listbox.bind('<<ListboxSelect>>', self.on_transform_select)
        
        # Sort options
        sort_frame = tk.Frame(selection_frame, bg=self.colors['card'])
        sort_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(sort_frame,
                text="Sort by:",
                bg=self.colors['card'],
                fg=self.colors['fg']).pack(side=tk.LEFT, padx=(0, 10))
        
        self.sort_var = tk.StringVar(value="mean_delta")
        sort_combo = ttk.Combobox(sort_frame,
                                 textvariable=self.sort_var,
                                 values=["mean_delta", "Count", "idx"],
                                 state="readonly",
                                 width=15)
        sort_combo.pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Button(sort_frame,
                  text="Sort",
                  command=self.sort_transforms,
                  style='Custom.TButton').pack(side=tk.LEFT)
        
        # Transform visualization
        viz_frame = tk.LabelFrame(control_frame,
                                 text="Transform Visualization",
                                 bg=self.colors['card'],
                                 fg=self.colors['fg'],
                                 font=('Arial', 12, 'bold'),
                                 padx=15,
                                 pady=15)
        viz_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas for molecule drawing
        self.transform_canvas = tk.Canvas(viz_frame,
                                         bg=self.colors['card'],
                                         highlightthickness=0)
        self.transform_canvas.pack(fill=tk.BOTH, expand=True, pady=10)
        
    def setup_results_tab(self):
        """Setup results display tab"""
        results_tab = tk.Frame(self.notebook, bg=self.colors['bg'])
        self.notebook.add(results_tab, text="Results")
        
        # Create text widget for results
        self.results_text = scrolledtext.ScrolledText(results_tab,
                                                     bg=self.colors['card'],
                                                     fg=self.colors['fg'],
                                                     insertbackground=self.colors['fg'],
                                                     font=('Courier', 10))
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
    def setup_visualization_tab(self):
        """Setup visualization tab"""
        viz_tab = tk.Frame(self.notebook, bg=self.colors['bg'])
        self.notebook.add(viz_tab, text="Visualization")
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(10, 6), dpi=100, facecolor=self.colors['card'])
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(self.colors['card'])
        
        # Customize colors
        self.ax.spines['bottom'].set_color(self.colors['fg'])
        self.ax.spines['top'].set_color(self.colors['fg'])
        self.ax.spines['right'].set_color(self.colors['fg'])
        self.ax.spines['left'].set_color(self.colors['fg'])
        self.ax.tick_params(axis='x', colors=self.colors['fg'])
        self.ax.tick_params(axis='y', colors=self.colors['fg'])
        self.ax.xaxis.label.set_color(self.colors['fg'])
        self.ax.yaxis.label.set_color(self.colors['fg'])
        self.ax.title.set_color(self.colors['accent'])
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, viz_tab)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Control frame
        control_frame = tk.Frame(viz_tab, bg=self.colors['bg'])
        control_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        # Visualization type
        tk.Label(control_frame,
                text="Plot Type:",
                bg=self.colors['bg'],
                fg=self.colors['fg']).pack(side=tk.LEFT, padx=(0, 10))
        
        self.plot_type_var = tk.StringVar(value="stripplot")
        plot_combo = ttk.Combobox(control_frame,
                                 textvariable=self.plot_type_var,
                                 values=["stripplot", "histogram", "boxplot"],
                                 state="readonly",
                                 width=15)
        plot_combo.pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Button(control_frame,
                  text="Update Plot",
                  command=self.update_plot,
                  style='Custom.TButton').pack(side=tk.LEFT)
        
    def setup_compounds_tab(self):
        """Setup tab to display compounds associated with transformations"""
        compounds_tab = tk.Frame(self.notebook, bg=self.colors['bg'])
        self.notebook.add(compounds_tab, text="Compound Details")
        
        # Create a paned window to split between structure view and text details
        paned_window = ttk.PanedWindow(compounds_tab, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left frame for molecule structure display
        structure_frame = tk.Frame(paned_window, bg=self.colors['card'])
        paned_window.add(structure_frame, weight=1)
        
        # Canvas for molecule structure
        self.molecule_canvas = tk.Canvas(structure_frame,
                                        bg=self.colors['card'],
                                        highlightthickness=0)
        self.molecule_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Label for molecule info
        self.molecule_info_label = tk.Label(structure_frame,
                                           bg=self.colors['card'],
                                           fg=self.colors['accent'],
                                           font=('Arial', 10, 'bold'))
        self.molecule_info_label.pack(pady=5)
        
        # Right frame for compound details text
        text_frame = tk.Frame(paned_window, bg=self.colors['bg'])
        paned_window.add(text_frame, weight=2)
        
        # Create text widget for compound details
        self.compounds_text = scrolledtext.ScrolledText(text_frame,
                                                       bg=self.colors['card'],
                                                       fg=self.colors['fg'],
                                                       insertbackground=self.colors['fg'],
                                                       font=('Courier', 9))
        self.compounds_text.pack(fill=tk.BOTH, expand=True)
        
        # Bind click event to display molecule when compound name is clicked
        self.compounds_text.tag_configure("compound", foreground=self.colors['secondary'])
        self.compounds_text.tag_bind("compound", "<Button-1>", self.on_compound_click)
        
        # Button to refresh compound details
        button_frame = tk.Frame(compounds_tab, bg=self.colors['bg'])
        button_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        ttk.Button(button_frame,
                  text="Show Compounds for Selected Transform",
                  command=self.show_compounds_for_transform,
                  style='Custom.TButton').pack()
        
    def update_status(self, message):
        """Update status bar"""
        self.status_bar.config(text=message)
        self.root.update_idletasks()
        
    def is_valid_smiles(self, smiles_string):
        """Check if a string is a valid SMILES"""
        if pd.isna(smiles_string):
            return False
        
        # Check if it looks like a SMILES (contains organic symbols and brackets)
        smiles_str = str(smiles_string)
        
        # Common non-SMILES patterns (like CHEMBL IDs)
        if re.match(r'^CHEMBL\d+$', smiles_str, re.IGNORECASE):
            return False
        if re.match(r'^[A-Za-z]+\d+$', smiles_str):
            return False
            
        # Check if it can be parsed as SMILES
        mol = Chem.MolFromSmiles(smiles_str)
        return mol is not None
    
    def detect_columns(self, df):
        """Detect which columns contain SMILES, names, and values"""
        # Try to find SMILES column
        smiles_col = None
        for col in df.columns:
            # Check first 10 non-null values
            sample = df[col].dropna().head(10)
            if len(sample) > 0:
                valid_count = sum(1 for s in sample if self.is_valid_smiles(s))
                if valid_count / len(sample) > 0.7:  # If >70% are valid SMILES
                    smiles_col = col
                    break
        
        # If no SMILES column found, try common column names
        if smiles_col is None:
            for name in ['SMILES', 'smiles', 'Smiles', 'smi', 'structure', 'canonical_smiles']:
                if name in df.columns:
                    smiles_col = name
                    break
        
        # Find numeric column (pIC50 or similar)
        numeric_col = None
        for col in df.columns:
            if col != smiles_col and pd.api.types.is_numeric_dtype(df[col]):
                numeric_col = col
                break
        
        # Find name/ID column
        name_col = None
        for col in df.columns:
            if col not in [smiles_col, numeric_col]:
                name_col = col
                break
        
        return smiles_col, name_col, numeric_col
    
    def load_csv_file(self):
        """Load data from CSV file"""
        try:
            file_path = filedialog.askopenfilename(
                title="Select CSV file",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            if file_path:
                self.update_status("Loading data from file...")
                
                # Try different delimiters and encodings
                delimiters = [',', ';', '\t', '|']
                encodings = ['utf-8', 'latin-1', 'cp1252']
                df = None
                
                for encoding in encodings:
                    for delimiter in delimiters:
                        try:
                            df = pd.read_csv(file_path, delimiter=delimiter, encoding=encoding)
                            if len(df.columns) >= 3:  # Need at least 3 columns
                                self.update_status(f"Loaded with encoding: {encoding}, delimiter: '{delimiter}'")
                                break
                        except:
                            continue
                    if df is not None and len(df.columns) >= 3:
                        break
                
                if df is None or len(df.columns) < 3:
                    # Last try without specifying delimiter (let pandas guess)
                    try:
                        df = pd.read_csv(file_path)
                    except:
                        try:
                            df = pd.read_csv(file_path, encoding='utf-8')
                        except:
                            messagebox.showerror("Error", "Could not read CSV file.")
                            return
                        
                if df is None:
                    messagebox.showerror("Error", "Could not read CSV file.")
                    return
                    
                self.df = df
                
                # Detect columns
                smiles_col, name_col, numeric_col = self.detect_columns(self.df)
                
                if smiles_col is None:
                    messagebox.showerror("Error", 
                        "Could not detect SMILES column. Please ensure your CSV contains SMILES strings.")
                    self.df = None
                    return
                    
                if numeric_col is None:
                    messagebox.showerror("Error", 
                        "Could not detect numeric activity column. Please ensure your CSV contains numeric values.")
                    self.df = None
                    return
                
                # Select and rename columns
                columns_to_keep = []
                if smiles_col:
                    columns_to_keep.append(smiles_col)
                if name_col:
                    columns_to_keep.append(name_col)
                if numeric_col:
                    columns_to_keep.append(numeric_col)
                
                self.df = self.df[columns_to_keep].copy()
                
                # Rename columns
                new_column_names = {}
                if smiles_col:
                    new_column_names[smiles_col] = 'SMILES'
                if name_col:
                    new_column_names[name_col] = 'Name'
                else:
                    self.df['Name'] = [f"Compound_{i+1}" for i in range(len(self.df))]
                if numeric_col:
                    new_column_names[numeric_col] = 'pIC50'
                
                self.df.rename(columns=new_column_names, inplace=True)
                
                # Ensure we have the right columns in the right order
                self.df = self.df[['SMILES', 'Name', 'pIC50']]
                
                # Check for missing values
                if self.df['SMILES'].isnull().any() or self.df['pIC50'].isnull().any():
                    messagebox.showwarning("Warning", "Some rows have missing data. These will be removed.")
                    self.df = self.df.dropna(subset=['SMILES', 'pIC50'])
                
                self.preprocess_data()
                self.update_status(f"Data loaded: {len(self.df)} molecules")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")
            self.update_status("Error loading data")
            self.df = None
    
    def preprocess_data(self):
        """Preprocess the loaded data"""
        if self.df is None:
            return
            
        # Add RDKit molecules
        self.update_status("Processing molecules...")
        
        # Filter out invalid SMILES
        valid_mask = self.df['SMILES'].apply(self.is_valid_smiles)
        invalid_count = (~valid_mask).sum()
        
        if invalid_count > 0:
            invalid_samples = self.df[~valid_mask]['SMILES'].head(5).tolist()
            messagebox.showwarning("Warning", 
                f"{invalid_count} invalid SMILES strings found. They will be removed.\n"
                f"Examples: {invalid_samples[:3]}")
            self.df = self.df[valid_mask].copy()
        
        # Convert SMILES to molecules
        self.df['mol'] = self.df['SMILES'].apply(lambda x: Chem.MolFromSmiles(str(x)))
        
        # Remove salts, counterions, etc.
        self.df['mol'] = self.df['mol'].apply(self.get_largest_fragment)
        
        # Update data info
        info_text = f"Dataset Information:\n"
        info_text += f"Number of molecules: {len(self.df)}\n"
        info_text += f"pIC50 range: {self.df['pIC50'].min():.2f} - {self.df['pIC50'].max():.2f}\n"
        info_text += f"Mean pIC50: {self.df['pIC50'].mean():.2f}\n"
        info_text += f"Standard deviation: {self.df['pIC50'].std():.2f}\n\n"
        info_text += "First few molecules:\n"
        
        for i, row in self.df.head(5).iterrows():
            info_text += f"{row['Name']}: pIC50={row['pIC50']:.2f}\n"
        
        self.data_info_text.delete(1.0, tk.END)
        self.data_info_text.insert(1.0, info_text)
    
    def get_largest_fragment(self, mol):
        """Get the largest fragment from a molecule"""
        if mol is None:
            return None
        try:
            frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
            if frags:
                return max(frags, key=lambda x: x.GetNumAtoms())
            return mol
        except:
            return mol
    
    def remove_map_nums(self, mol):
        """Remove atom map numbers from a molecule"""
        if mol is None:
            return mol
        try:
            for atm in mol.GetAtoms():
                atm.SetAtomMapNum(0)
        except:
            pass
        return mol
    
    def sort_fragments(self, mol):
        """Sort fragments by number of atoms"""
        if mol is None:
            return []
        try:
            frag_list = list(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True))
            [self.remove_map_nums(x) for x in frag_list]
            frag_num_atoms_list = [(x.GetNumAtoms(), x) for x in frag_list]
            frag_num_atoms_list.sort(key=lambda x: x[0], reverse=True)
            return [x[1] for x in frag_num_atoms_list]
        except:
            return []
    
    def fragment_mol(self, mol, maxCuts=1):
        """Fragment molecule using RDKit's MMPA - Consistent with original notebook"""
        try:
            # Use RDKit's FragmentMol directly (same as in the original notebook)
            return FragmentMol(mol, maxCuts=maxCuts)
        except Exception as e:
            print(f"Error fragmenting molecule: {e}")
            return []
    
    def run_analysis(self):
        """Run the MMP analysis"""
        if self.df is None:
            messagebox.showwarning("Warning", "Please load data first")
            return
            
        # Run in separate thread to avoid freezing GUI
        thread = threading.Thread(target=self._run_analysis_thread)
        thread.daemon = True
        thread.start()
    
    def _run_analysis_thread(self):
        """Run analysis in separate thread"""
        try:
            self.min_transform_occurrence = self.min_occurrence_var.get()
            
            # Show progress window
            self.progress_window = tk.Toplevel(self.root)
            self.progress_window.title("Analysis Progress")
            self.progress_window.geometry("400x150")
            self.progress_window.configure(bg=self.colors['bg'])
            self.progress_window.transient(self.root)
            self.progress_window.grab_set()
            
            # Center the progress window
            self.progress_window.update_idletasks()
            x = self.root.winfo_x() + (self.root.winfo_width() - self.progress_window.winfo_width()) // 2
            y = self.root.winfo_y() + (self.root.winfo_height() - self.progress_window.winfo_height()) // 2
            self.progress_window.geometry(f"+{x}+{y}")
            
            tk.Label(self.progress_window,
                    text="Running MMP Analysis...",
                    font=('Arial', 12, 'bold'),
                    bg=self.colors['bg'],
                    fg=self.colors['accent']).pack(pady=20)
            
            # Create progress bar without custom style
            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(self.progress_window,
                                          variable=progress_var,
                                          maximum=100,
                                          mode='determinate')
            progress_bar.pack(fill=tk.X, padx=20, pady=10)
            
            status_label = tk.Label(self.progress_window,
                                   text="Initializing...",
                                   bg=self.colors['bg'],
                                   fg=self.colors['fg'])
            status_label.pack()
            
            def update_progress(step, total, message):
                progress = (step / total) * 100
                progress_var.set(progress)
                status_label.config(text=message)
                self.progress_window.update()
            
            # Step 1: Decompose molecules - Consistent with original notebook
            update_progress(1, 5, "Decomposing molecules...")
            row_list = []
            total_mols = len(self.df)
            
            for i, (smiles, name, pIC50, mol) in enumerate(self.df.values):
                if mol is None:
                    continue
                    
                # Use FragmentMol directly as in original notebook
                frag_list = self.fragment_mol(mol, maxCuts=1)
                for _, frag_mol in frag_list:
                    pair_list = self.sort_fragments(frag_mol)
                    if len(pair_list) == 2:
                        # Consistent with original notebook: SMILES, Core, R_group, Name, pIC50
                        tmp_list = [smiles, Chem.MolToSmiles(pair_list[0]), 
                                   Chem.MolToSmiles(pair_list[1]), name, pIC50]
                        row_list.append(tmp_list)
                
                if i % 10 == 0 or i == total_mols - 1:
                    update_progress(1, 5, f"Processing molecule {i+1}/{total_mols}")
            
            if not row_list:
                raise ValueError("No valid molecular pairs found. Check your input data.")
            
            # Consistent column names with original notebook
            row_df = pd.DataFrame(row_list, columns=["SMILES", "Core", "R_group", "Name", "pIC50"])
            
            # Step 2: Collect pairs with same scaffold - Consistent with original notebook
            update_progress(2, 5, "Collecting molecular pairs...")
            delta_list = []
            
            core_groups = list(row_df.groupby("Core"))
            total_cores = len(core_groups)
            
            for core_idx, (k, v) in enumerate(core_groups):
                if len(v) > 2:
                    for a, b in combinations(range(0, len(v)), 2):
                        reagent_a = v.iloc[a]
                        reagent_b = v.iloc[b]
                        if reagent_a.SMILES == reagent_b.SMILES:
                            continue
                        reagent_a, reagent_b = sorted([reagent_a, reagent_b], key=lambda x: x.SMILES)
                        delta = reagent_b.pIC50 - reagent_a.pIC50
                        # Consistent with original notebook column order
                        delta_list.append(list(reagent_a.values) + list(reagent_b.values) +
                                         [f"{reagent_a.R_group.replace('*','*-')}>>{reagent_b.R_group.replace('*','*-')}", delta])
                
                if core_idx % 10 == 0 or core_idx == total_cores - 1:
                    update_progress(2, 5, f"Processing core {core_idx+1}/{total_cores}")
            
            if not delta_list:
                raise ValueError("No delta values calculated. Check your input data.")
            
            # Consistent column names with original notebook
            cols = ["SMILES_1", "Core_1", "R_group_1", "Name_1", "pIC50_1",
                   "SMILES_2", "Core_2", "Rgroup_1", "Name_2", "pIC50_2",
                   "Transform", "Delta"]
            self.delta_df = pd.DataFrame(delta_list, columns=cols)
            
            # Step 3: Collect frequently occurring pairs
            update_progress(3, 5, "Analyzing transforms...")
            mmp_list = []
            transform_groups = list(self.delta_df.groupby("Transform"))
            
            for transform_idx, (k, v) in enumerate(transform_groups):
                if len(v) > self.min_transform_occurrence:
                    # Collect compound names for this transform
                    compound_names = list(set(v['Name_1'].tolist() + v['Name_2'].tolist()))
                    mmp_list.append([k, len(v), v.Delta.values, compound_names])
                
                if transform_idx % 10 == 0 or transform_idx == len(transform_groups) - 1:
                    update_progress(3, 5, f"Processing transform {transform_idx+1}/{len(transform_groups)}")
            
            if not mmp_list:
                raise ValueError(f"No transforms found with occurrence > {self.min_transform_occurrence}. Try lowering the minimum occurrence.")
            
            self.mmp_df = pd.DataFrame(mmp_list, columns=["Transform", "Count", "Deltas", "Compounds"])
            self.mmp_df['idx'] = range(0, len(self.mmp_df))
            self.mmp_df['mean_delta'] = [x.mean() for x in self.mmp_df.Deltas]
            
            # Add reaction molecules
            update_progress(4, 5, "Creating reaction visualizations...")
            self.mmp_df['rxn_mol'] = self.mmp_df.Transform.apply(
                lambda x: AllChem.ReactionFromSmarts(x, useSmiles=True))
            
            # Create transform dictionary
            transform_dict = dict([(a, b) for a, b in self.mmp_df[["Transform", "idx"]].values])
            self.delta_df['idx'] = [transform_dict.get(x) for x in self.delta_df.Transform]
            
            # Update transform listbox
            self.update_transform_list()
            
            update_progress(5, 5, "Analysis complete!")
            
            # Close progress window after delay
            self.progress_window.after(1000, self.progress_window.destroy)
            
            # Show results
            self.show_results_summary()
            self.update_status(f"Analysis complete. Found {len(self.mmp_df)} transforms.")
            
        except Exception as e:
            if hasattr(self, 'progress_window') and self.progress_window.winfo_exists():
                self.progress_window.destroy()
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            self.update_status("Analysis failed")
    
    def update_transform_list(self):
        """Update the transform listbox"""
        if self.mmp_df is None:
            return
            
        self.transform_listbox.delete(0, tk.END)
        
        # Sort by mean delta
        sorted_df = self.mmp_df.sort_values("mean_delta", ascending=True)
        
        for idx, row in sorted_df.iterrows():
            display_text = f"{row['idx']:3d} | {row['Transform'][:40]:40s} | Count: {row['Count']:3d} | Mean Delta: {row['mean_delta']:6.2f}"
            self.transform_listbox.insert(tk.END, display_text)
            # Store index in listbox
            self.transform_listbox.itemconfig(tk.END, {'bg': self.get_color_for_delta(row['mean_delta'])})
    
    def get_color_for_delta(self, delta):
        """Get color based on delta value"""
        if delta < -2:
            return '#FF6B6B'  # Red for strong negative effect
        elif delta < -1:
            return '#FFA726'  # Orange for moderate negative
        elif delta < 1:
            return '#42A5F5'  # Blue for neutral/mild
        elif delta < 2:
            return '#66BB6A'  # Green for moderate positive
        else:
            return '#4CAF50'  # Dark green for strong positive
    
    def on_transform_select(self, event):
        """Handle transform selection"""
        selection = self.transform_listbox.curselection()
        if not selection:
            return
            
        selected_text = self.transform_listbox.get(selection[0])
        # Extract index from text (first 3 characters)
        try:
            idx = int(selected_text.split('|')[0].strip())
            self.current_idx = idx
            self.show_transform(idx)
            self.update_plot()
        except:
            pass
    
    def show_transform(self, idx):
        """Display the selected transform"""
        if self.mmp_df is None or idx >= len(self.mmp_df):
            return
            
        row = self.mmp_df.iloc[idx]
        
        # Clear canvas
        self.transform_canvas.delete("all")
        
        # Draw reaction
        try:
            rxn = row['rxn_mol']
            img = Draw.ReactionToImage(rxn, subImgSize=(200, 150))
            
            # Convert PIL Image to PhotoImage
            photo = ImageTk.PhotoImage(img)
            
            # Display on canvas
            self.transform_canvas.create_image(200, 75, image=photo)
            self.transform_canvas.image = photo  # Keep reference
            
            # Add text
            self.transform_canvas.create_text(200, 160,
                                             text=f"Transform {idx}: {row['Transform']}",
                                             fill=self.colors['fg'],
                                             font=('Arial', 10))
            
            self.transform_canvas.create_text(200, 180,
                                             text=f"Count: {row['Count']}, Mean Delta pIC50: {row['mean_delta']:.2f}",
                                             fill=self.colors['accent'],
                                             font=('Arial', 10, 'bold'))
            
        except Exception as e:
            self.transform_canvas.create_text(200, 75,
                                             text=f"Error drawing transform: {str(e)}",
                                             fill=self.colors['error'],
                                             font=('Arial', 10))
    
    def update_plot(self):
        """Update the visualization plot"""
        if self.mmp_df is None or self.current_idx >= len(self.mmp_df):
            return
            
        self.ax.clear()
        row = self.mmp_df.iloc[self.current_idx]
        deltas = row['Deltas']
        
        plot_type = self.plot_type_var.get()
        
        if plot_type == "stripplot":
            sns.stripplot(x=deltas, ax=self.ax, color=self.colors['secondary'], size=8)
            self.ax.axvline(0, ls="--", color=self.colors['error'], alpha=0.7)
            self.ax.axvline(row['mean_delta'], ls="--", color=self.colors['accent'], alpha=0.7)
            self.ax.set_xlabel("Delta pIC50", color=self.colors['fg'])
            self.ax.set_title(f"Transform {self.current_idx}: {row['Transform'][:30]}...", 
                            color=self.colors['accent'])
            
        elif plot_type == "histogram":
            self.ax.hist(deltas, bins=20, color=self.colors['secondary'], edgecolor=self.colors['fg'], alpha=0.7)
            self.ax.axvline(0, ls="--", color=self.colors['error'], alpha=0.7)
            self.ax.axvline(row['mean_delta'], ls="--", color=self.colors['accent'], alpha=0.7)
            self.ax.set_xlabel("Delta pIC50", color=self.colors['fg'])
            self.ax.set_ylabel("Frequency", color=self.colors['fg'])
            self.ax.set_title(f"Distribution for Transform {self.current_idx}", 
                            color=self.colors['accent'])
            
        elif plot_type == "boxplot":
            sns.boxplot(x=deltas, ax=self.ax, color=self.colors['secondary'])
            self.ax.axvline(0, ls="--", color=self.colors['error'], alpha=0.7)
            self.ax.set_xlabel("Delta pIC50", color=self.colors['fg'])
            self.ax.set_title(f"Boxplot for Transform {self.current_idx}", 
                            color=self.colors['accent'])
        
        self.ax.set_xlim(-5, 5)
        self.ax.grid(True, alpha=0.3, color=self.colors['fg'])
        
        # Update statistics text
        stats_text = f"Statistics:\n"
        stats_text += f"Mean: {np.mean(deltas):.3f}\n"
        stats_text += f"Std: {np.std(deltas):.3f}\n"
        stats_text += f"Min: {np.min(deltas):.3f}\n"
        stats_text += f"Max: {np.max(deltas):.3f}\n"
        stats_text += f"Count: {len(deltas)}"
        
        self.ax.text(0.02, 0.98, stats_text,
                    transform=self.ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor=self.colors['card'], alpha=0.8),
                    fontsize=10,
                    color=self.colors['fg'])
        
        self.canvas.draw()
    
    def show_results_summary(self):
        """Display analysis results summary"""
        if self.mmp_df is None:
            return
            
        self.results_text.delete(1.0, tk.END)
        
        summary = "=" * 80 + "\n"
        summary += "MATCHED MOLECULAR PAIRS ANALYSIS RESULTS\n"
        summary += "=" * 80 + "\n\n"
        
        summary += f"Dataset Statistics:\n"
        summary += f"Total molecules: {len(self.df)}\n"
        summary += f"Total molecular pairs analyzed: {len(self.delta_df)}\n"
        summary += f"Unique transforms found: {len(self.mmp_df)}\n"
        summary += f"Minimum transform occurrence: {self.min_transform_occurrence}\n\n"
        
        summary += "=" * 80 + "\n"
        summary += "TOP 20 TRANSFORMS (sorted by absolute mean Delta pIC50)\n"
        summary += "=" * 80 + "\n\n"
        
        # Sort by absolute mean delta
        sorted_df = self.mmp_df.copy()
        sorted_df['abs_mean'] = abs(sorted_df['mean_delta'])
        sorted_df = sorted_df.sort_values('abs_mean', ascending=False).head(20)
        
        for idx, row in sorted_df.iterrows():
            summary += f"Transform {row['idx']}:\n"
            summary += f"  Reaction: {row['Transform']}\n"
            summary += f"  Count: {row['Count']}\n"
            summary += f"  Mean Delta pIC50: {row['mean_delta']:.3f}\n"
            summary += f"  Delta pIC50 range: {min(row['Deltas']):.3f} to {max(row['Deltas']):.3f}\n"
            # Show first few compound names
            if row['Compounds']:
                compounds = row['Compounds'][:5]  # Show first 5
                summary += f"  Compounds: {', '.join(compounds)}"
                if len(row['Compounds']) > 5:
                    summary += f" ... and {len(row['Compounds']) - 5} more"
                summary += "\n"
            summary += "-" * 60 + "\n"
        
        self.results_text.insert(1.0, summary)
    
    def on_compound_click(self, event):
        """Handle click on compound name in text widget"""
        # Get the character position of the click
        index = self.compounds_text.index(f"@{event.x},{event.y}")
        
        # Get the tag names at this position
        tags = self.compounds_text.tag_names(index)
        
        # If a compound tag is present, extract the compound name
        for tag in tags:
            if tag.startswith("compound_"):
                compound_name = tag.replace("compound_", "")
                self.display_molecule_structure(compound_name)
                break
    
    def display_molecule_structure(self, compound_name):
        """Display molecule structure in the structure panel"""
        # Clear canvas
        self.molecule_canvas.delete("all")
        
        # Find the compound in the dataset
        compound_data = self.df[self.df['Name'] == compound_name]
        
        if compound_data.empty:
            # Try to find in delta_df
            compound_data_1 = self.delta_df[self.delta_df['Name_1'] == compound_name]
            compound_data_2 = self.delta_df[self.delta_df['Name_2'] == compound_name]
            
            if not compound_data_1.empty:
                smiles = compound_data_1.iloc[0]['SMILES_1']
                pIC50 = compound_data_1.iloc[0]['pIC50_1']
            elif not compound_data_2.empty:
                smiles = compound_data_2.iloc[0]['SMILES_2']
                pIC50 = compound_data_2.iloc[0]['pIC50_2']
            else:
                self.molecule_info_label.config(text=f"Compound not found: {compound_name}")
                return
        else:
            smiles = compound_data.iloc[0]['SMILES']
            pIC50 = compound_data.iloc[0]['pIC50']
        
        try:
            # Create molecule
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("Invalid SMILES")
            
            # Draw molecule
            img = Draw.MolToImage(mol, size=(300, 300))
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img)
            
            # Display on canvas
            self.molecule_canvas.create_image(150, 150, image=photo)
            self.molecule_canvas.image = photo  # Keep reference
            
            # Update info label
            info_text = f"Name: {compound_name}\n"
            info_text += f"pIC50: {pIC50:.2f}\n"
            info_text += f"SMILES: {smiles[:50]}..." if len(smiles) > 50 else f"SMILES: {smiles}"
            self.molecule_info_label.config(text=info_text)
            
        except Exception as e:
            self.molecule_canvas.create_text(150, 150,
                                            text=f"Error drawing molecule: {str(e)}",
                                            fill=self.colors['error'],
                                            font=('Arial', 10))
            self.molecule_info_label.config(text=f"Error displaying: {compound_name}")
    
    def show_compounds_for_transform(self):
        """Show all compounds associated with the selected transform"""
        if self.mmp_df is None or self.current_idx >= len(self.mmp_df):
            messagebox.showwarning("Warning", "No transform selected or data not loaded")
            return
            
        row = self.mmp_df.iloc[self.current_idx]
        
        self.compounds_text.delete(1.0, tk.END)
        
        details = "=" * 80 + "\n"
        details += f"COMPOUNDS FOR TRANSFORM {self.current_idx}\n"
        details += "=" * 80 + "\n\n"
        
        details += f"Transform: {row['Transform']}\n"
        details += f"Mean Delta pIC50: {row['mean_delta']:.3f}\n"
        details += f"Number of occurrences: {row['Count']}\n"
        details += f"Total unique compounds: {len(row['Compounds'])}\n\n"
        
        # Get detailed information for each compound
        if row['Compounds']:
            # Get all pairs for this transform
            transform_pairs = self.delta_df[self.delta_df['idx'] == self.current_idx]
            
            details += "All Compound Pairs for this Transform:\n"
            details += "-" * 80 + "\n"
            
            for _, pair in transform_pairs.iterrows():
                # Make compound names clickable
                details += f"Pair: "
                
                # Add first compound as clickable
                tag_name_1 = f"compound_{pair['Name_1']}"
                self.compounds_text.insert(tk.END, pair['Name_1'], (tag_name_1, "compound"))
                self.compounds_text.insert(tk.END, " -> ")
                
                # Add second compound as clickable
                tag_name_2 = f"compound_{pair['Name_2']}"
                self.compounds_text.insert(tk.END, pair['Name_2'], (tag_name_2, "compound"))
                self.compounds_text.insert(tk.END, "\n")
                
                details += f"  Delta pIC50: {pair['Delta']:.3f}\n"
                details += f"  pIC50 1: {pair['pIC50_1']:.2f}, pIC50 2: {pair['pIC50_2']:.2f}\n"
                details += "-" * 40 + "\n"
            
            # Insert the details text
            self.compounds_text.insert(1.0, details)
            
            # Add all unique compounds section
            self.compounds_text.insert(tk.END, "\nAll Unique Compounds Involved:\n")
            self.compounds_text.insert(tk.END, "-" * 80 + "\n")
            
            # Get unique compounds and their pIC50 values
            all_compounds = set(row['Compounds'])
            compound_data = {}
            
            for name in all_compounds:
                # Find the compound in original data
                compound_row = self.df[self.df['Name'] == name]
                if not compound_row.empty:
                    compound_data[name] = {
                        'pIC50': compound_row.iloc[0]['pIC50'],
                        'SMILES': compound_row.iloc[0]['SMILES'][:100]
                    }
            
            # Sort compounds by pIC50
            sorted_compounds = sorted(compound_data.items(), key=lambda x: x[1]['pIC50'], reverse=True)
            
            for name, data in sorted_compounds:
                # Make compound name clickable
                tag_name = f"compound_{name}"
                self.compounds_text.insert(tk.END, f"{name}:\n", (tag_name, "compound"))
                self.compounds_text.insert(tk.END, f"  pIC50: {data['pIC50']:.2f}\n")
                self.compounds_text.insert(tk.END, f"  SMILES: {data['SMILES']}...\n\n")
        else:
            self.compounds_text.insert(1.0, details + "No compound data available.\n")
        
        # Configure the compound tag style
        self.compounds_text.tag_configure("compound", foreground=self.colors['secondary'], underline=1)
    
    def sort_transforms(self):
        """Sort transforms based on selected criterion"""
        if self.mmp_df is None:
            return
            
        sort_by = self.sort_var.get()
        ascending = sort_by != "mean_delta"  # Sort mean_delta in descending order by default
        
        if sort_by == "mean_delta":
            # Sort by absolute value for mean_delta
            self.mmp_df['abs_mean'] = abs(self.mmp_df['mean_delta'])
            self.mmp_df = self.mmp_df.sort_values('abs_mean', ascending=False)
        else:
            self.mmp_df = self.mmp_df.sort_values(sort_by, ascending=ascending)
        
        self.update_transform_list()
        self.update_status(f"Sorted by {sort_by}")
    
    def export_results(self):
        """Export results to CSV files with UTF-8 encoding"""
        if self.mmp_df is None:
            messagebox.showwarning("Warning", "No results to export")
            return
            
        try:
            # Ask for directory
            directory = filedialog.askdirectory(title="Select export directory")
            if not directory:
                return
            
            # Export MMP results with compound names - using UTF-8 encoding
            mmp_export = self.mmp_df.copy()
            mmp_export['Compounds'] = mmp_export['Compounds'].apply(lambda x: ';'.join(x) if x else '')
            mmp_file = os.path.join(directory, "mmp_results.csv")
            mmp_export.to_csv(mmp_file, index=False, encoding='utf-8')
            
            # Export delta pairs - using UTF-8 encoding
            delta_file = os.path.join(directory, "mmp_pairs.csv")
            self.delta_df.to_csv(delta_file, index=False, encoding='utf-8')
            
            # Export dataset info - using UTF-8 encoding
            info_file = os.path.join(directory, "dataset_info.txt")
            with open(info_file, 'w', encoding='utf-8') as f:
                f.write(self.data_info_text.get(1.0, tk.END))
            
            # Export compound details for each transform - using UTF-8 encoding
            details_file = os.path.join(directory, "transform_details.txt")
            with open(details_file, 'w', encoding='utf-8') as f:
                f.write(self.results_text.get(1.0, tk.END))
            
            messagebox.showinfo("Success", 
                              f"Results exported successfully:\n{mmp_file}\n{delta_file}\n{info_file}\n{details_file}")
            self.update_status("Results exported")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export results: {str(e)}")

def main():
    """Main function to run the application"""
    # Check for required packages
    try:
        import pandas as pd
        from rdkit import Chem
        from rdkit.Chem.rdMMPA import FragmentMol
        from PIL import Image, ImageTk
    except ImportError as e:
        print(f"Error: Required package not found: {e}")
        print("Please install required packages:")
        print("pip install pandas rdkit-pypi seaborn matplotlib pillow")
        return
    
    root = tk.Tk()
    app = MMPAnalyzerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
