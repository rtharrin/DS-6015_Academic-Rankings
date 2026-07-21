# Interactive Main Script
"""
TODO: Add a description of what this script does and how to use it.
"""

#imports
import pandas as pd
import os
import numpy as np
import plotly.express as px
import ipywidgets as widgets
from IPython.display import display
import logging
import datetime
import ipywidgets as widgets
from IPython.display import display
from IPython import get_ipython
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox



LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "file_handler": {
            "class": "logging.FileHandler",
            "formatter": "standard",
            "filename": f"../logs/interactive_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log",
            "mode": "a",
        },
        "console_handler": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
        },
    },
    "loggers": {
        "__main__": {
            "handlers": ["file_handler", "console_handler"],
            "level": logging.INFO,
            "propagate": False,
        },
    },
}


logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)
try:
    # Works when running as a saved script (.py)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Works when running in an interactive notebook
    BASE_DIR = os.getcwd()
# Combines the script folder with your relative path safely
IN_PATH = os.path.normpath(os.path.join(BASE_DIR, "../data/interactive/in/"))
OUT_PATH = os.path.normpath(os.path.join(BASE_DIR, "../data/interactive/out/"))
ARCHIVE_PATH = os.path.normpath(os.path.join(BASE_DIR, "../data/interactive/in/archive/"))


# Functions
def load_data(file_path=IN_PATH):
    """
    Load all csvs in the specified directory into a single pandas DataFrame and return it uncleaned.
    """

    logger.info(f"load_data: Loading data from {file_path}")

    try:
        all_files = [os.path.join(file_path, f) for f in os.listdir(file_path) if f.endswith('.csv')]
        df_list = [pd.read_csv(f) for f in all_files]
        df = pd.concat(df_list, ignore_index=True)
        logger.info(f"Loaded {len(all_files)} files from {file_path}.")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error



def archive_raw_data(in_path=IN_PATH, archive_path=ARCHIVE_PATH):
    """
    Move the raw data to the archive directory.
    """
    try:
        if not os.path.exists(in_path):
            logger.warning(f"Input path {in_path} does not exist. Skipping archive.")
            return
        
        in_files = [os.path.join(in_path, f) for f in os.listdir(in_path) if f.endswith('.csv')]
        archive_files = [os.path.join(archive_path, f) for f in os.listdir(in_path) if f.endswith('.csv')]

        for in_file, archive_file in zip(in_files, archive_files):
            if os.path.exists(archive_file):
                logger.warning(f"File {archive_file} already exists in the archive. Skipping.")
                continue
            os.rename(in_file, archive_file)
        
    
        logger.info(f"Archived raw data from {in_path} to {archive_path}.")

    except Exception as e:
        logger.error(f"Error during archiving raw data: {e}")
        raise

def preprocess_data(df, export = False):
    """
    Preprocess the DataFrame by cleaning and transforming the data.
    Assumes Academic insight data is in a long format and pivots it to a wide format.
    If column names in the index, pivot, or value columns change this function will need to be updated.

    Parameters:
    df (pd.DataFrame): The input DataFrame to preprocess.
    export (bool): If True, export the cleaned DataFrame to a CSV file in the OUT_PATH directory.
    """
    logger.info("preprocess_data: Starting preprocessing of data.")
    # Make sure df isn't empty
    if df.empty:
        logger.warning("The DataFrame is empty. No preprocessing will be done.")
        return df

    #set index columns
    index_columns = ['Issue year', 'Name', 'IPEDS ID', 'School ID']
    piv_columns = ['Metric description']
    val_columns = ['Value']

    try:
        #make sure required columns exist
        for col in index_columns + piv_columns + val_columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' is missing from the DataFrame.")
                return df
        
        #remove duplicates, pivot values and rename columns
        df = df.drop_duplicates()

        df = df.pivot_table(
            index=index_columns, 
            columns=piv_columns,
            values=val_columns,
            aggfunc='first').reset_index()

        df = df.rename(columns=lambda x: str(x).strip().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_').lower())
        df.columns = [' '.join([str(part) for part in col if str(part) != '']).strip() for col in df.columns]
        df.columns = df.columns.str.replace('value ', '').str.lower()

        logger.info("Preprocessing completed successfully.")

        if export:
            export_path = os.path.join(OUT_PATH, f"interactive_cleaned_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")
            df.to_csv(export_path, index=False)
            logger.info(f"Preprocessed data exported to {export_path}")


        return df
    except Exception as e:
        logger.error(f"Error during preprocess_data: {e}")
        return df  # Return the original DataFrame on error
    

## Working the best, keep this

def build_plot(df_plot, school=None, metrics=None, show_widget_controls=True):
    """
    Build a plot from a cleaned dataframe.
    Supports multiple school selections.
    Line color is based on School Name, and Line Style is based on Metric.
    Supports both Jupyter Notebook widgets and standalone Tkinter widgets for user interaction.
    Also supports a static plot if no widget controls are desired.
    """
    logger.info("build_plot: Starting to build the plot.")
    try:
        df_plot = df_plot.copy()
        if "issue_year" not in df_plot.columns or "name" not in df_plot.columns:
            raise ValueError("Expected columns 'issue_year' and 'name'.")
        
        if df_plot["issue_year"].dtype == object:
            df_plot["issue_year"] = pd.to_numeric(df_plot["issue_year"], errors="coerce")
            
        exclude_cols = {"issue_year", "name", "ipeds_id", "school_id"}
        
        if metrics is None:
            metrics = sorted(
                [col for col in df_plot.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df_plot[col])]
            )
        if not metrics:
            raise ValueError("No numeric metric columns available for plotting.")
            
        schools = sorted([str(s).strip() for s in df_plot["name"].dropna().unique() if str(s).strip()])
        if not schools:
            raise ValueError("No school names available for plotting.")
            
        # Default to the first school if none provided as an explicit list/string
        if school is None:
            selected_schools = [schools[0]]
        elif isinstance(school, str):
            selected_schools = [school]
        else:
            selected_schools = school

        # --- DETECT ENVIRONMENT ---
        is_notebook = False
        try:
            if get_ipython() is not None:
                is_notebook = True
        except Exception:
            pass

        # --- OPTION A: JUPYTER NOTEBOOK WIDGETS ---
        if show_widget_controls and is_notebook:
            try:             
                checkboxes = [
                    widgets.Checkbox(value=(i < min(2, len(metrics))), description=metric, indent=False)
                    for i, metric in enumerate(metrics)
                ]
                metric_box = widgets.VBox(checkboxes, layout=widgets.Layout(max_height="200px", overflow="auto"))
                
                # Changed Dropdown to SelectMultiple for notebooks
                school_selector = widgets.SelectMultiple(
                    options=schools,
                    value=tuple(selected_schools),
                    description="Schools:",
                    layout=widgets.Layout(max_height="150px")
                )
                output = widgets.Output()

                def get_selected_metrics():
                    return [cb.description for cb in checkboxes if cb.value]

                def update_plot(_=None):
                    selected_metrics = get_selected_metrics()
                    chosen_schools = list(school_selector.value)
                    
                    with output:
                        output.clear_output(wait=True)
                        if not selected_metrics or not chosen_schools:
                            print("Select at least one school and one metric.")
                            return
                        
                        filtered = df_plot[df_plot["name"].astype(str).str.strip().isin(chosen_schools)].copy()
                        if filtered.empty:
                            print("No data available for chosen selections.")
                            return
                        
                        # Melt dataframe to long format for color vs dash plotting
                        df_melted = filtered.melt(
                            id_vars=["issue_year", "name"], 
                            value_vars=selected_metrics, 
                            var_name="Metric", 
                            value_name="Value"
                        )
                        
                        fig = px.line(
                            df_melted, x="issue_year", y="Value",
                            color="name", line_dash="Metric", markers=True,
                            title="Metric Trends Comparison",
                            labels={"issue_year": "Year", "name": "School Name"}
                        )
                        fig.update_layout(hovermode="x unified")
                        fig.update_xaxes(dtick=1)
                        fig.show()

                school_selector.observe(update_plot, names="value")
                for cb in checkboxes:
                    cb.observe(update_plot, names="value")
                
                control_panel = widgets.HBox([
                    widgets.VBox([school_selector, widgets.Label("Choose metrics:")], layout=widgets.Layout(width="260px")),
                    metric_box
                ])
                display(control_panel, output)
                update_plot()
                return None
            except Exception as e:
                logger.warning(f"Failed to load notebook widgets, falling back: {e}")

        # --- OPTION B: STANDALONE DESKTOP TKINTER WIDGETS ---
        elif show_widget_controls and not is_notebook:
            

            root = tk.Tk()
            root.title("Plot Controls (Multi-School)")
            root.geometry("450x600")

            # School Multi-Select Listbox with Scrollbar
            tk.Label(root, text="Select Schools (Hold Ctrl/Cmd to select multiple):", font=("Arial", 10, "bold")).pack(anchor="w", padx=10, pady=5)
            
            school_frame = tk.Frame(root)
            school_frame.pack(fill="x", padx=10, pady=5)
            
            school_scrollbar = ttk.Scrollbar(school_frame, orient="vertical")
            # selectmode="extended" allows click-and-drag or Ctrl/Cmd clicks
            school_listbox = tk.Listbox(school_frame, selectmode="extended", height=6, yscrollcommand=school_scrollbar.set)
            school_scrollbar.config(command=school_listbox.yview)
            
            for s in schools:
                school_listbox.insert(tk.END, s)
                if s in selected_schools:
                    school_listbox.selection_set(tk.END) # Default highlight
                    
            school_listbox.pack(side="left", fill="x", expand=True)
            school_scrollbar.pack(side="right", fill="y")

            # Metrics Checkboxes with a Scrollbar
            tk.Label(root, text="Select Metrics:", font=("Arial", 10, "bold")).pack(anchor="w", padx=10, pady=5)
            
            canvas_frame = tk.Frame(root)
            canvas_frame.pack(fill="both", expand=True, padx=10, pady=5)
            
            canvas = tk.Canvas(canvas_frame)
            scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
            scrollable_frame = tk.Frame(canvas)

            scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)

            checkbox_vars = {}
            for i, metric in enumerate(metrics):
                initial_val = True if i < min(2, len(metrics)) else False
                var = tk.BooleanVar(value=initial_val)
                checkbox_vars[metric] = var
                cb = tk.Checkbutton(scrollable_frame, text=metric, variable=var, anchor="w")
                cb.pack(fill="x", anchor="w", pady=2)

            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")

            # Execution logic
            def on_submit():
                # Extract all highlighted indices from the listbox
                chosen_schools = [school_listbox.get(idx) for idx in school_listbox.curselection()]
                chosen_metrics = [m for m, var in checkbox_vars.items() if var.get()]
                
                if not chosen_schools:
                    messagebox.showwarning("Warning", "Please select at least one school.")
                    return
                if not chosen_metrics:
                    messagebox.showwarning("Warning", "Please select at least one metric.")
                    return
                
                filtered = df_plot[df_plot["name"].astype(str).str.strip().isin(chosen_schools)].copy()
                if filtered.empty:
                    messagebox.showerror("Error", "No data available for chosen selections.")
                    return
                
                # Transform table from wide to long format for Plotly mapping
                df_melted = filtered.melt(
                    id_vars=["issue_year", "name"], 
                    value_vars=chosen_metrics, 
                    var_name="Metric", 
                    value_name="Value"
                )
                
                fig = px.line(
                    df_melted, x="issue_year", y="Value",
                    color="name", line_dash="Metric", markers=True,
                    title="Metric Trends Comparison",
                    labels={"issue_year": "Year", "name": "School Name"}
                )
                fig.update_layout(hovermode="x unified")
                fig.update_xaxes(dtick=1)
                fig.show()

            submit_btn = ttk.Button(root, text="Generate Plot", command=on_submit)
            submit_btn.pack(fill="x", padx=10, pady=15)

            root.mainloop()
            return None

        # --- OPTION C: STATIC PLOT (NO CONTROLS) ---
        filtered = df_plot[df_plot["name"].astype(str).str.strip() == selected_schools].copy()
        if filtered.empty:
            raise ValueError(f"No data available for {selected_schools}.")
            
        fig = px.line(
            filtered, x="issue_year", y=metrics, markers=True,
            title=f"Metric Trends for {selected_schools}", labels={"issue_year": "Year"}
        )
        fig.update_layout(hovermode="x unified", legend_title_text="Metrics")
        fig.update_xaxes(dtick=1)
        return fig

    except Exception as e:
        logger.error(f"Error during build_plot: {e}")
        raise

def main(export_cleaned=True, show_widget_controls=True):
    """
    Main function to load, preprocess, and plot the data.
    """
    try:
        # Load raw data
        df_raw = load_data(IN_PATH)
        if df_raw.empty:
            logger.warning("No data loaded. Exiting.")
            return
        
        # Preprocess data
        df_cleaned = preprocess_data(df_raw, export=True)
        
        # Build and display plot with widget controls
        build_plot(df_cleaned, show_widget_controls=True)

        #archive raw data after processing
        archive_raw_data(IN_PATH, ARCHIVE_PATH)

        logger.info("Data processing and plot generation complete.")

    except Exception as e:
        logger.error(f"Error in main: {e}")

    logger.info("Script execution finished.")


if __name__ == "__main__":
    main()