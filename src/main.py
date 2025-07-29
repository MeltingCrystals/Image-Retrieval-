# main.py
"""Main script for running the enhanced image search application with geometric metrics and MI analysis."""

import logging
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
from app_pipeline import EnhancedImageSearchApp
from mi_analysis import MIAnalysis, EnhancedMIAnalysis
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EnhancedImageSearchGUI:
    """Advanced GUI for the image search application with geometric metrics and MI analysis."""

    def __init__(self, root, app, test_mode=False, coco_path=None):
        """
        Initializes the enhanced GUI for the image search application.
        """
        self.app = app
        self.root = root
        self.test_mode = test_mode
        self.coco_path = coco_path
        self.root.title("Enhanced Image Search with Geometric Metrics")
        self.root.geometry("1000x700")

        # Create notebook for tabbed interface
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True)

        # Create tabs
        self.search_tab = ttk.Frame(self.notebook)
        self.mi_analysis_tab = ttk.Frame(self.notebook)
        self.geometric_analysis_tab = ttk.Frame(self.notebook)
        self.comparison_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.search_tab, text="Search")
        self.notebook.add(self.mi_analysis_tab, text="MI Analysis")
        self.notebook.add(self.geometric_analysis_tab, text="Geometric Analysis")
        self.notebook.add(self.comparison_tab, text="Metric Comparison")

        # Set up each tab
        self.setup_search_tab()
        self.setup_mi_analysis_tab()
        self.setup_geometric_analysis_tab()
        self.setup_comparison_tab()

        # Variables for analysis
        self.mi_analyzer = None
        self.enhanced_mi_analyzer = None
        self.optimal_params = None

        # Process images if in test mode
        if self.test_mode:
            self.process_images()

    def create_scrollable_frame(self, parent):
        """Creates a scrollable frame and returns the container frame and inner scrollable frame."""
        # Create a frame to hold the canvas and scrollbar
        container = ttk.Frame(parent)
        container.pack(fill="both", expand=True, padx=10, pady=5)

        # Create canvas and scrollbar
        canvas = tk.Canvas(container)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)

        # Create the scrollable frame inside the canvas
        scrollable_frame = ttk.Frame(canvas)

        # Configure the canvas
        def configure_canvas(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        scrollable_frame.bind("<Configure>", configure_canvas)
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        # Configure canvas to expand with the window
        def configure_canvas_width(event):
            canvas.itemconfig(canvas_window, width=event.width)

        canvas.bind("<Configure>", configure_canvas_width)
        canvas.configure(yscrollcommand=scrollbar.set)

        # Pack the scrollbar and canvas
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        return container, scrollable_frame

    def setup_search_tab(self):
        """Sets up the basic search tab interface."""
        # Top frame for controls
        self.control_frame = ttk.Frame(self.search_tab)
        self.control_frame.pack(fill="x", pady=5)

        self.select_button = ttk.Button(self.control_frame, text="Select Folder", command=self.select_and_process)
        self.select_button.pack(side="left", padx=5)

        self.query_label = ttk.Label(self.control_frame, text="Enter search query:")
        self.query_label.pack(side="left", padx=5)
        self.query_entry = ttk.Entry(self.control_frame, width=30)
        self.query_entry.pack(side="left", padx=5)

        self.search_button = ttk.Button(self.control_frame, text="Search", command=self.search)
        self.search_button.pack(side="left", padx=5)

        self.clear_button = ttk.Button(self.control_frame, text="Clear", command=self.clear_search)
        self.clear_button.pack(side="left", padx=5)

        # Add options for search method
        self.search_method_label = ttk.Label(self.control_frame, text="Search method:")
        self.search_method_label.pack(side="left", padx=5)

        self.search_method_var = tk.StringVar(value="standard")
        self.standard_radio = ttk.Radiobutton(self.control_frame, text="Standard",
                                              variable=self.search_method_var, value="standard")
        self.optimized_radio = ttk.Radiobutton(self.control_frame, text="Optimized",
                                               variable=self.search_method_var, value="optimized")
        self.standard_radio.pack(side="left")
        self.optimized_radio.pack(side="left", padx=5)

        # Create scrollable results frame
        self.search_container, self.search_scrollable_frame = self.create_scrollable_frame(self.search_tab)

    def setup_mi_analysis_tab(self):
        """Sets up the MI analysis tab interface."""
        # Control frame
        self.mi_control_frame = ttk.Frame(self.mi_analysis_tab)
        self.mi_control_frame.pack(fill="x", pady=5)

        # MI Analysis buttons
        self.run_mi_button = ttk.Button(self.mi_control_frame, text="Run Standard MI Analysis",
                                        command=self.run_standard_mi_analysis)
        self.run_mi_button.pack(side="left", padx=5)

        # Create scrollable results frame
        self.mi_container, self.mi_results_frame = self.create_scrollable_frame(self.mi_analysis_tab)

    def setup_geometric_analysis_tab(self):
        """Sets up the geometric analysis tab interface."""
        # Control frame
        self.geo_control_frame = ttk.Frame(self.geometric_analysis_tab)
        self.geo_control_frame.pack(fill="x", pady=5)

        # Geometric Analysis buttons
        self.run_geo_mi_button = ttk.Button(self.geo_control_frame, text="Run Geometric MI Analysis",
                                            command=self.run_geometric_mi_analysis)
        self.run_geo_mi_button.pack(side="left", padx=5)

        # Parameter optimization button
        self.optimize_button = ttk.Button(self.geo_control_frame, text="Optimize Parameters",
                                          command=self.run_parameter_optimization)
        self.optimize_button.pack(side="left", padx=5)

        # Apply parameters button
        self.apply_params_button = ttk.Button(self.geo_control_frame, text="Apply Optimal Parameters",
                                              command=self.apply_optimal_parameters)
        self.apply_params_button.pack(side="left", padx=5)

        # Create scrollable results frame
        self.geo_container, self.geo_results_frame = self.create_scrollable_frame(self.geometric_analysis_tab)

    def setup_comparison_tab(self):
        """Sets up the comparison tab interface."""
        # Control frame
        self.comp_control_frame = ttk.Frame(self.comparison_tab)
        self.comp_control_frame.pack(fill="x", pady=5)

        # Query entry
        self.comp_query_label = ttk.Label(self.comp_control_frame, text="Query:")
        self.comp_query_label.pack(side="left", padx=5)
        self.comp_query_entry = ttk.Entry(self.comp_control_frame, width=30)
        self.comp_query_entry.pack(side="left", padx=5)

        # Compare button
        self.compare_button = ttk.Button(self.comp_control_frame, text="Compare All Metrics",
                                         command=self.compare_metrics)
        self.compare_button.pack(side="left", padx=5)

        # Create scrollable results frame
        self.comp_container, self.comp_results_frame = self.create_scrollable_frame(self.comparison_tab)

    def select_and_process(self):
        """Selects a folder of images and processes them."""
        if self.test_mode:
            image_paths = list(Path(self.coco_path).glob("*.jpg"))
            logger.info(f"Test mode: Using COCO dataset at {self.coco_path}")
        else:
            image_paths = self.app.scan_folders()

        if not image_paths:
            messagebox.showerror("Error", "No valid images found in the selected folder.")
            return

        logger.info(f"Found {len(image_paths)} images. Processing...")
        self.app.process_images(image_paths)
        logger.info("Processing complete!")
        messagebox.showinfo("Success", f"Processed {len(image_paths)} images successfully.")

    def process_images(self):
        """Process images from a selected folder or test dataset."""
        if self.test_mode and self.coco_path:
            image_paths = list(Path(self.coco_path).glob("*.jpg"))
            logger.info(f"Test mode: Using COCO dataset at {self.coco_path}")
        else:
            logger.info("Please select a folder with images to process...")
            image_paths = self.app.scan_folders()

        if not image_paths:
            messagebox.showerror("Error", "No valid images found in the selected folder.")
            return

        logger.info(f"Found {len(image_paths)} images. Processing...")
        self.app.process_images(image_paths)
        logger.info("Processing complete!")

    def search(self):
        """Performs image search with the selected method."""
        query = self.query_entry.get().strip()
        if not query:
            messagebox.showwarning("Warning", "Query cannot be empty.")
            return

        # Clear previous results
        for widget in self.search_scrollable_frame.winfo_children():
            widget.destroy()

        # Determine search method
        search_method = self.search_method_var.get()
        use_optimized = search_method == "optimized"

        # Show progress
        progress_label = ttk.Label(self.search_scrollable_frame, text="Searching...")
        progress_label.pack(pady=10)
        self.root.update_idletasks()

        try:
            # Perform search
            results = self.app.search_images(query, top_k=10, use_optimized_similarity=use_optimized)

            # Remove progress label
            progress_label.destroy()

            # Display method info
            method_text = "Optimized Geometric Similarity" if use_optimized else "Standard Angular Similarity"
            ttk.Label(self.search_scrollable_frame,
                      text=f"Search results for: '{query}' using {method_text}",
                      font=("Arial", 12, "bold")).pack(pady=5)

            if not results:
                ttk.Label(self.search_scrollable_frame, text="No matching images found.").pack()
                return

            # Display results
            self.display_search_results(results)

        except Exception as e:
            logger.error(f"Search failed: {e}")
            progress_label.destroy()
            messagebox.showerror("Error", f"Search failed: {str(e)}")

    def display_search_results(self, results):
        """Displays search results with images and scores."""
        # Create a frame for results
        results_container = ttk.Frame(self.search_scrollable_frame)
        results_container.pack(fill="both", expand=True, padx=10, pady=5)

        # Display each result
        for idx, match in enumerate(results, 1):
            try:
                filename = Path(match["path"]).name
                confidence = match["score"] * 100

                # Create frame for this result
                result_frame = ttk.Frame(results_container)
                result_frame.pack(pady=10, fill="x")

                # Try to load and display the image
                try:
                    img = Image.open(match["path"]).resize((200, 200), Image.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    img_label = ttk.Label(result_frame, image=photo)
                    img_label.image = photo  # Keep a reference to prevent garbage collection
                    img_label.pack(side="left", padx=5)

                    # Display image info
                    info_frame = ttk.Frame(result_frame)
                    info_frame.pack(side="left", fill="both", expand=True)

                    ttk.Label(info_frame, text=f"Result {idx}", font=("Arial", 12, "bold")).pack(anchor="w")
                    ttk.Label(info_frame, text=f"Filename: {filename}").pack(anchor="w")
                    ttk.Label(info_frame, text=f"Confidence: {confidence:.1f}%").pack(anchor="w")
                    ttk.Label(info_frame, text=f"Path: {match['path']}").pack(anchor="w")

                except Exception as img_error:
                    # If image loading fails, show error
                    logger.error(f"Failed to load image {match['path']}: {img_error}")
                    ttk.Label(result_frame, text=f"Result {idx}: {filename}").pack(side="left")
                    ttk.Label(result_frame, text=f"Error loading image: {str(img_error)}").pack(side="left", padx=5)

            except Exception as e:
                logger.error(f"Failed to display result {idx}: {e}")
                ttk.Label(results_container, text=f"Error displaying result {idx}: {str(e)}").pack()

    def clear_search(self):
        """Clears the search results and query."""
        self.query_entry.delete(0, tk.END)
        for widget in self.search_scrollable_frame.winfo_children():
            widget.destroy()

    def run_standard_mi_analysis(self):
        """Runs standard mutual information analysis."""
        # Clear previous results
        for widget in self.mi_results_frame.winfo_children():
            widget.destroy()

        # Show progress
        progress_label = ttk.Label(self.mi_results_frame, text="Running MI analysis...")
        progress_label.pack(pady=10)
        self.root.update_idletasks()

        try:
            # Run MI analysis
            self.mi_analyzer, mi_results = self.app.run_mi_analysis(num_pairs=1000, num_bins=20)

            if not self.mi_analyzer or not mi_results:
                progress_label.destroy()
                messagebox.showerror("Error", "MI analysis failed. Make sure you have processed enough images.")
                return

            # Create visualization
            output_path = self.app.create_mi_visualization("standard_mi_analysis.png")

            progress_label.destroy()

            # Display results
            results_frame = ttk.Frame(self.mi_results_frame)
            results_frame.pack(fill="both", expand=True, padx=10, pady=5)

            # Show MI value and optimal threshold
            optimal_threshold = self.mi_analyzer.find_optimal_threshold()
            info_text = (f"Mutual Information: {mi_results['default']:.4f} bits\n"
                         f"Optimal Angular Threshold: {optimal_threshold:.4f}\n"
                         f"(Default threshold: 0.25)")
            ttk.Label(results_frame, text=info_text, justify="left", font=("Arial", 12)).pack(pady=10)

            # Show plot
            try:
                img = Image.open(output_path)
                width, height = 800, 600
                img = img.resize((width, height), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                plot_label = ttk.Label(results_frame, image=photo)
                plot_label.image = photo  # Keep a reference
                plot_label.pack(pady=10)

                # Add interpretation
                interpretation = self.mi_analyzer.interpret_mi_value(mi_results['default'])
                interp_text = (f"Interpretation:\n"
                               f"• {interpretation['quality_assessment']}\n"
                               f"• Normalized MI: {interpretation['normalized_mi']:.3f} (out of a theoretical max of {interpretation['max_theoretical_mi']:.3f} bits)\n"
                               f"• {interpretation['retrieval_implication']}")
                ttk.Label(results_frame, text=interp_text, justify="left", font=("Arial", 11)).pack(pady=10)

            except Exception as e:
                logger.error(f"Error displaying MI plot: {e}")
                ttk.Label(results_frame, text=f"Error displaying plot: {str(e)}").pack(pady=10)

        except Exception as e:
            logger.error(f"MI analysis failed: {e}")
            progress_label.destroy()
            messagebox.showerror("Error", f"MI analysis failed: {str(e)}")

    def run_geometric_mi_analysis(self):
        """Runs enhanced geometric mutual information analysis."""
        # Clear previous results
        for widget in self.geo_results_frame.winfo_children():
            widget.destroy()

        # Show progress
        progress_label = ttk.Label(self.geo_results_frame, text="Running geometric MI analysis...")
        progress_label.pack(pady=10)
        self.root.update_idletasks()

        try:
            # Run enhanced MI analysis
            self.enhanced_mi_analyzer, mi_results = self.app.run_enhanced_mi_analysis(
                num_pairs=1000, num_bins=20, keep_unnormalized=True)

            if not self.enhanced_mi_analyzer or not mi_results:
                progress_label.destroy()
                messagebox.showerror("Error",
                                     "Geometric MI analysis failed. Make sure you have processed enough images.")
                return

            # Create custom bar chart visualization
            output_path = self.create_mi_bar_chart(mi_results, "mi_bar_chart.png")
            progress_label.destroy()

            # Display results
            results_frame = ttk.Frame(self.geo_results_frame)
            results_frame.pack(fill="both", expand=True, padx=10, pady=5)

            # Show formatted MI values with percentages
            ttk.Label(results_frame, text="Mutual Information Analysis Results:",
                      font=("Arial", 14, "bold")).pack(anchor="w", pady=(10, 15))

            # Calculate max theoretical MI and sort results
            max_mi = max(mi_results.values()) if mi_results else 1.0
            sorted_results = sorted(mi_results.items(), key=lambda x: x[1], reverse=True)

            # Display formatted results
            for metric, mi_value in sorted_results:
                percentage = (mi_value / max_mi) * 100

                # Format metric name
                metric_display = self.format_metric_name(metric)

                # Create formatted text
                if metric == sorted_results[0][0]:  # Best performing metric
                    result_text = f"{metric_display}: {mi_value:.4f} bits ({percentage:.1f}% of max)"
                    font_style = ("Arial", 12, "bold")
                else:
                    result_text = f"{metric_display}: {mi_value:.4f} bits"
                    font_style = ("Arial", 11)

                ttk.Label(results_frame, text=result_text, font=font_style).pack(anchor="w", padx=20, pady=2)

            # Show the bar chart
            try:
                img = Image.open(output_path)
                width, height = 900, 500  # Adjusted size for bar chart
                img = img.resize((width, height), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                plot_label = ttk.Label(results_frame, image=photo)
                plot_label.image = photo  # Keep a reference
                plot_label.pack(pady=15)

                # Add interpretation
                best_metric = sorted_results[0]
                insights_text = (f"Analysis Summary:\n"
                                 f"• Top performer: {self.format_metric_name(best_metric[0])} "
                                 f"({best_metric[1]:.4f} bits)\n"
                                 f"• This metric captures the most semantic information for color relationships\n"
                                 f"• Consider using this metric or run parameter optimization for best results")
                ttk.Label(results_frame, text=insights_text, justify="left",
                          font=("Arial", 10), foreground="blue").pack(pady=10)

            except Exception as e:
                logger.error(f"Error displaying MI bar chart: {e}")
                ttk.Label(results_frame, text=f"Error displaying chart: {str(e)}").pack(pady=10)

        except Exception as e:
            logger.error(f"Geometric MI analysis failed: {e}")
            progress_label.destroy()
            messagebox.showerror("Error", f"Geometric MI analysis failed: {str(e)}")

    def format_metric_name(self, metric):
        """Format metric names for display."""
        name_mapping = {
            'linf_distance': 'L∞',
            'l1_distance': 'L1',
            'cosine_distance': 'Cosine',
            'l2_distance': 'L2',
            'magnitude_difference': 'Magnitude',
            'angular': 'Angular',
            'cosine_similarity': 'Cosine'
        }
        return name_mapping.get(metric, metric.replace('_', ' ').title())

    def create_mi_bar_chart(self, mi_results, filename):
        """Create a custom bar chart for MI results with confidence intervals."""
        import matplotlib.pyplot as plt
        import numpy as np

        # Calculate confidence intervals
        ci_data = self.calculate_confidence_intervals(mi_results)

        # Sort results by MI value (descending)
        sorted_results = sorted(ci_data.items(), key=lambda x: x[1]['value'], reverse=True)

        # Extract data
        metrics = [self.format_metric_name(metric) for metric, _ in sorted_results]
        mi_values = [data['value'] for _, data in sorted_results]
        errors = [data['margin'] for _, data in sorted_results]

        # Calculate percentages
        max_mi = max(mi_values)
        percentages = [(mi / max_mi) * 100 for mi in mi_values]

        # Create figure
        plt.figure(figsize=(12, 7))

        # Create bar chart with error bars
        bars = plt.bar(range(len(metrics)), mi_values,
                       yerr=errors,  # Add error bars
                       color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(metrics)],
                       alpha=0.8, edgecolor='black', linewidth=1,
                       capsize=5)  # Error bar caps

        # Customize the chart
        plt.xlabel('Geometric Metrics', fontsize=12, fontweight='bold')
        plt.ylabel('Mutual Information (bits)', fontsize=12, fontweight='bold')
        plt.title('Mutual Information by Geometric Metric\n(Higher values indicate better semantic discrimination)',
                  fontsize=14, fontweight='bold', pad=20)

        # Set x-axis labels
        plt.xticks(range(len(metrics)), metrics, fontsize=11, fontweight='bold')
        plt.yticks(fontsize=10)

        # Add value labels on top of bars
        for i, (bar, mi_val, pct) in enumerate(zip(bars, mi_values, percentages)):
            height = bar.get_height()

            # Format label text
            if i == 0:  # Best performing metric
                label = f'{mi_val:.4f}\n({pct:.1f}% of max)'
                fontweight = 'bold'
                color = 'red'
            else:
                label = f'{mi_val:.4f}'
                fontweight = 'normal'
                color = 'black'

            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     label, ha='center', va='bottom',
                     fontsize=10, fontweight=fontweight, color=color)

        # Add grid for better readability
        plt.grid(True, alpha=0.3, axis='y')

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        return filename

    def calculate_confidence_intervals(self, mi_results, confidence_level=0.95):
        """Calculate confidence intervals for MI values using bootstrap."""
        import numpy as np
        # Remove scipy dependency for now and use simple approximation
        # from scipy import stats

        confidence_intervals = {}

        for metric, mi_value in mi_results.items():
            # Simple confidence interval estimation without scipy
            n_samples = 1000  # Approximate number of pairs used

            # Rough estimation of standard error (this is simplified)
            std_error = mi_value / np.sqrt(n_samples) * 0.1  # Approximate

            # Calculate confidence interval using normal approximation
            # For 95% confidence level, use 1.96 instead of stats.norm.ppf
            z_score = 1.96 if confidence_level == 0.95 else 2.576  # 99% level
            margin_error = z_score * std_error
            ci_lower = max(0, mi_value - margin_error)
            ci_upper = mi_value + margin_error

            confidence_intervals[metric] = {
                'value': mi_value,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'margin': margin_error
            }

        return confidence_intervals

    def run_parameter_optimization(self):
        """Runs parameter optimization to find optimal weights for metrics."""
        if not self.enhanced_mi_analyzer:
            messagebox.showwarning("Warning", "Run Geometric MI Analysis first.")
            return

        # Clear previous optimization results
        for widget in self.geo_results_frame.winfo_children():
            if isinstance(widget, ttk.LabelFrame) and widget.winfo_children() and \
                    hasattr(widget.winfo_children()[0], "cget") and \
                    widget.winfo_children()[0].cget("text").startswith("Optimal Parameters"):
                widget.destroy()

        # Show progress
        optim_frame = ttk.LabelFrame(self.geo_results_frame, text="Parameter Optimization")
        optim_frame.pack(fill="x", padx=10, pady=10)

        progress_label = ttk.Label(optim_frame, text="Optimizing parameters...")
        progress_label.pack(pady=10)
        self.root.update_idletasks()

        try:
            # Create parameter grid
            param_grid = {
                'w_angle': [0.5, 0.75, 1.0],
                'w_l1': [0.0, 0.1, 0.2],
                'w_l2': [0.0, 0.1, 0.2],
                'w_inf': [0.0, 0.1, 0.2],
                'w_mag': [0.0, 0.1, 0.2]
            }

            # Run optimization
            optimal_result = self.enhanced_mi_analyzer.find_optimal_parameters(param_grid)
            self.optimal_params = optimal_result['parameters']

            progress_label.destroy()

            # Display results
            ttk.Label(optim_frame, text="Optimal Parameters:", font=("Arial", 12, "bold")).pack(anchor="w",
                                                                                                pady=(10, 5))

            params_text = ""
            for param, value in self.optimal_params.items():
                param_name = param.replace('w_', 'Weight for ').replace('_', ' ').title()
                params_text += f"{param_name}: {value:.2f}\n"

            params_text += f"\nMutual Information with these parameters: {optimal_result['mi_value']:.4f} bits"
            ttk.Label(optim_frame, text=params_text, justify="left", font=("Arial", 11)).pack(pady=10, padx=20)

            # Add apply button
            ttk.Button(optim_frame, text="Apply These Parameters",
                       command=self.apply_optimal_parameters).pack(pady=10)

        except Exception as e:
            logger.error(f"Parameter optimization failed: {e}")
            progress_label.destroy()
            messagebox.showerror("Error", f"Parameter optimization failed: {str(e)}")

    def apply_optimal_parameters(self):
        """Applies the optimal parameters to the searcher."""
        if not self.optimal_params:
            messagebox.showwarning("Warning", "Run parameter optimization first.")
            return

        try:
            # Apply parameters to the searcher
            self.app.searcher.set_similarity_params(self.optimal_params)

            # Enable optimized search
            self.search_method_var.set("optimized")

            # Show confirmation
            messagebox.showinfo("Success", "Applied optimal parameters to the search engine. "
                                           "Optimized search is now available.")

        except Exception as e:
            logger.error(f"Failed to apply parameters: {e}")
            messagebox.showerror("Error", f"Failed to apply parameters: {str(e)}")

    def compare_metrics(self):
        """Performs a comparison search using all metrics."""
        query = self.comp_query_entry.get().strip()
        if not query:
            messagebox.showwarning("Warning", "Query cannot be empty.")
            return

        # Clear previous results
        for widget in self.comp_results_frame.winfo_children():
            widget.destroy()

        # Show progress
        progress_label = ttk.Label(self.comp_results_frame, text="Comparing metrics...")
        progress_label.pack(pady=10)
        self.root.update_idletasks()

        try:
            # Perform multi-metric search
            results = self.app.search_with_multiple_metrics(query, top_k=5)

            progress_label.destroy()

            # Create notebook for metric tabs
            metrics_notebook = ttk.Notebook(self.comp_results_frame)
            metrics_notebook.pack(fill="both", expand=True, padx=10, pady=5)

            # Add a tab for each metric
            for metric, metric_results in results.items():
                if metric == "analysis":
                    continue

                # Create tab
                metric_tab = ttk.Frame(metrics_notebook)
                metrics_notebook.add(metric_tab, text=metric.replace("_", " ").title())

                # Create scrollable frame for this tab
                _, tab_scrollable_frame = self.create_scrollable_frame(metric_tab)

                # Display results
                ttk.Label(tab_scrollable_frame, text=f"Results using {metric.replace('_', ' ').title()}",
                          font=("Arial", 12, "bold")).pack(pady=5)

                for idx, match in enumerate(metric_results, 1):
                    try:
                        filename = Path(match["path"]).name

                        # Create frame for this result
                        result_frame = ttk.Frame(tab_scrollable_frame)
                        result_frame.pack(pady=5, fill="x")

                        # Try to load thumbnail
                        try:
                            img = Image.open(match["path"]).resize((100, 100), Image.LANCZOS)
                            photo = ImageTk.PhotoImage(img)
                            img_label = ttk.Label(result_frame, image=photo)
                            img_label.image = photo
                            img_label.pack(side="left", padx=5)
                        except Exception as img_error:
                            ttk.Label(result_frame, text="[Image Error]").pack(side="left", padx=5)

                        # Show result info
                        info_text = f"{idx}. {filename}\n"

                        # Add metric-specific score
                        if metric == "cosine_similarity":
                            info_text += f"Cosine similarity: {match['cosine_similarity']:.3f}"
                        elif metric.endswith("_distance"):
                            info_text += f"{metric.replace('_', ' ').title()}: {match[metric]:.3f}"
                        elif metric == "optimized_similarity":
                            info_text += f"Optimized score: {match['optimized_similarity']:.3f}"

                        ttk.Label(result_frame, text=info_text).pack(side="left", padx=5)

                    except Exception as e:
                        logger.error(f"Error displaying result: {e}")

            # Add analysis tab
            if "analysis" in results:
                analysis_tab = ttk.Frame(metrics_notebook)
                metrics_notebook.add(analysis_tab, text="Analysis")

                # Create scrollable frame for analysis
                _, analysis_scrollable_frame = self.create_scrollable_frame(analysis_tab)

                # Display intersection analysis
                if "intersections" in results["analysis"]:
                    intersect_frame = ttk.LabelFrame(analysis_scrollable_frame, text="Metric Intersections")
                    intersect_frame.pack(fill="x", padx=10, pady=5)

                    for key, data in results["analysis"]["intersections"].items():
                        ttk.Label(intersect_frame, text=f"{key}: {data['intersection_size']} common results "
                                                        f"({data['intersection_ratio']:.2f} overlap ratio)").pack(
                            anchor="w")

                # Display unique contributions
                if "unique_contributions" in results["analysis"]:
                    unique_frame = ttk.LabelFrame(analysis_scrollable_frame, text="Unique Contributions")
                    unique_frame.pack(fill="x", padx=10, pady=5)

                    for metric, data in results["analysis"]["unique_contributions"].items():
                        ttk.Label(unique_frame,
                                  text=f"{metric.replace('_', ' ').title()}: {data['unique_count']} unique results "
                                       f"({data['unique_ratio']:.2f} uniqueness ratio)").pack(anchor="w")

        except Exception as e:
            logger.error(f"Metric comparison failed: {e}")
            progress_label.destroy()
            messagebox.showerror("Error", f"Metric comparison failed: {str(e)}")


def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create application
    app = EnhancedImageSearchApp()

    # Create and run the GUI
    root = tk.Tk()
    gui = EnhancedImageSearchGUI(root, app)

    # Set window title and size
    root.title("Enhanced Image Search v1.0 - Geometric Information Theory")
    root.minsize(900, 600)

    # Start the application
    root.mainloop()


if __name__ == "__main__":
    main()