import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image
import numpy as np
import threading
import os
import psutil
import gc

Image.MAX_IMAGE_PIXELS = None

class ChunkwiseTiffViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("TIFF Chunkwise Downsampling Viewer")
        self.root.geometry("1200x800")
        
        # Memory management settings
        self.max_memory_mb = 500  # Maximum memory usage in MB
        self.chunk_size = 2048    # Chunk size for processing
        self.downsample_factor = 4  # Default downsample factor
        
        # Image data
        self.tiff_path = None
        self.image_size = None
        self.image_mode = None
        self.total_pages = 0
        self.current_page = 0
        
        # Viewport and zoom settings
        self.view_center = None  # (x, y) center of current view
        self.view_width = None   # width of current view in original pixels
        self.view_height = None  # height of current view in original pixels
        self.zoom_level = 1.0    # current zoom level
        self.is_dragging = False
        self.drag_start = None
        
        # Processing state
        self.is_processing = False
        self.current_display_data = None
        
        # Create interface
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # File selection
        ttk.Button(control_frame, text="Open TIFF", 
                  command=self.open_file).pack(side=tk.LEFT, padx=(0, 10))
        
        # Memory settings
        memory_frame = ttk.Frame(control_frame)
        memory_frame.pack(side=tk.LEFT, padx=20)
        
        ttk.Label(memory_frame, text="Max Memory (MB):").pack(side=tk.LEFT)
        self.memory_var = tk.StringVar(value="500")
        memory_entry = ttk.Entry(memory_frame, textvariable=self.memory_var, width=6)
        memory_entry.pack(side=tk.LEFT, padx=(5, 10))
        
        ttk.Label(memory_frame, text="Chunk Size:").pack(side=tk.LEFT)
        self.chunk_var = tk.StringVar(value="2048")
        chunk_combo = ttk.Combobox(memory_frame, textvariable=self.chunk_var,
                                  values=["1024", "2048", "4096", "8192"], width=6)
        chunk_combo.pack(side=tk.LEFT, padx=(5, 10))
        
        # Downsample settings
        ttk.Label(memory_frame, text="Base DS:").pack(side=tk.LEFT)
        self.downsample_var = tk.StringVar(value="4")
        downsample_combo = ttk.Combobox(memory_frame, textvariable=self.downsample_var,
                                       values=["1", "2", "4", "8", "16", "32"], width=4)
        downsample_combo.pack(side=tk.LEFT, padx=(5, 5))
        
        # Zoomed region downsampling
        ttk.Label(memory_frame, text="Zoom DS:").pack(side=tk.LEFT, padx=(5, 2))
        self.zoom_ds_var = tk.StringVar(value="1")
        zoom_ds_combo = ttk.Combobox(memory_frame, textvariable=self.zoom_ds_var,
                                    values=["1", "2", "4", "8"], width=3)
        zoom_ds_combo.pack(side=tk.LEFT, padx=(0, 10))
        
        # Process button
        ttk.Button(control_frame, text="Process Image", 
                  command=self.process_image).pack(side=tk.LEFT, padx=20)
        
        # Zoom controls
        zoom_frame = ttk.Frame(control_frame)
        zoom_frame.pack(side=tk.LEFT, padx=20)
        
        # Coordinate input for zoom region
        coord_frame = ttk.Frame(zoom_frame)
        coord_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        ttk.Label(coord_frame, text="Region:").pack(side=tk.LEFT)
        ttk.Label(coord_frame, text="X1:").pack(side=tk.LEFT, padx=(5, 2))
        self.x1_var = tk.StringVar()
        x1_entry = ttk.Entry(coord_frame, textvariable=self.x1_var, width=6)
        x1_entry.pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Label(coord_frame, text="Y1:").pack(side=tk.LEFT, padx=(5, 2))
        self.y1_var = tk.StringVar()
        y1_entry = ttk.Entry(coord_frame, textvariable=self.y1_var, width=6)
        y1_entry.pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Label(coord_frame, text="X2:").pack(side=tk.LEFT, padx=(5, 2))
        self.x2_var = tk.StringVar()
        x2_entry = ttk.Entry(coord_frame, textvariable=self.x2_var, width=6)
        x2_entry.pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Label(coord_frame, text="Y2:").pack(side=tk.LEFT, padx=(5, 2))
        self.y2_var = tk.StringVar()
        y2_entry = ttk.Entry(coord_frame, textvariable=self.y2_var, width=6)
        y2_entry.pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(coord_frame, text="Zoom to Region", 
                  command=self.zoom_to_region).pack(side=tk.LEFT, padx=(10, 0))
        
        # Zoom level controls
        level_frame = ttk.Frame(zoom_frame)
        level_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        ttk.Button(level_frame, text="Zoom In", 
                  command=self.zoom_in).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(level_frame, text="Zoom Out", 
                  command=self.zoom_out).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(level_frame, text="Reset View", 
                  command=self.reset_view).pack(side=tk.LEFT)
        
        # Zoom level display
        ttk.Label(level_frame, text="Zoom:").pack(side=tk.LEFT, padx=(20, 5))
        self.zoom_label = ttk.Label(level_frame, text="1.0x")
        self.zoom_label.pack(side=tk.LEFT)
        
        # Memory usage display
        self.memory_label = ttk.Label(control_frame, text="Memory: 0 MB")
        self.memory_label.pack(side=tk.RIGHT)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=(0, 5))
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Display area
        display_frame = ttk.Frame(main_frame)
        display_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, display_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Information display
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.info_text = tk.Text(info_frame, height=4, width=80, font=('Arial', 9))
        self.info_text.pack(fill=tk.X)
        
        # Start memory monitoring
        self.update_memory_display()
        
    def update_memory_display(self):
        """Update memory usage display"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.memory_label.config(text=f"Memory: {memory_mb:.1f} MB")
        self.root.after(1000, self.update_memory_display)
        
    def open_file(self):
        filename = filedialog.askopenfilename(
            title="Select TIFF file",
            filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")]
        )
        
        if filename:
            self.load_tiff_metadata(filename)
    
    def load_tiff_metadata(self, filename):
        """Load only TIFF metadata without loading the full image"""
        try:
            self.status_var.set("Loading metadata...")
            
            def load_thread():
                try:
                    self.tiff_path = filename
                    with Image.open(filename) as temp_tiff:
                        # Get image dimensions and mode
                        self.image_size = temp_tiff.size
                        self.image_mode = temp_tiff.mode
                        self.total_pages = 0
                        
                        # Get multi-page information
                        while True:
                            try:
                                temp_tiff.seek(self.total_pages)
                                self.total_pages += 1
                            except EOFError:
                                break
                    
                    # Update interface
                    self.root.after(0, self.update_file_info)
                    
                except Exception as e:
                    self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to load metadata: {e}"))
                finally:
                    self.root.after(0, lambda: self.status_var.set("Ready"))
            
            threading.Thread(target=load_thread, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open file: {e}")
    
    def update_file_info(self):
        """Update file information display"""
        if self.tiff_path:
            info = f"File: {os.path.basename(self.tiff_path)}\n"
            info += f"Original Size: {self.image_size}\n"
            info += f"Mode: {self.image_mode}\n"
            info += f"Pages: {self.total_pages}\n"
            info += f"Estimated Memory Usage: {self.estimate_memory_usage()} MB"
            
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(1.0, info)
    
    def estimate_memory_usage(self):
        """Estimate memory usage for downsampled image"""
        if not self.image_size:
            return 0
            
        width, height = self.image_size
        downsample = int(self.downsample_var.get())
        
        # Calculate downsampled dimensions
        ds_width = width // downsample
        ds_height = height // downsample
        
        # Estimate memory (assuming 8-bit per channel)
        if self.image_mode == "L":  # Grayscale
            memory_bytes = ds_width * ds_height
        else:  # Color (RGB/RGBA)
            channels = 3 if self.image_mode == "RGB" else 4
            memory_bytes = ds_width * ds_height * channels
        
        return memory_bytes / 1024 / 1024  # Convert to MB
    
    def process_image(self):
        """Process the image using chunkwise downsampling"""
        if not self.tiff_path:
            messagebox.showwarning("Warning", "Please open a TIFF file first")
            return
            
        if self.is_processing:
            messagebox.showwarning("Warning", "Processing already in progress")
            return
            
        self.is_processing = True
        self.status_var.set("Processing image...")
        
        def process_thread():
            try:
                # Get processing parameters
                max_memory = int(self.memory_var.get())
                chunk_size = int(self.chunk_var.get())
                downsample = int(self.downsample_var.get())
                
                # Process image in chunks
                result = self.process_chunkwise(downsample, chunk_size)
                
                if result is not None:
                    # Initialize viewport for zooming
                    width, height = self.image_size
                    self.view_center = (width // 2, height // 2)
                    self.view_width = width // 4
                    self.view_height = height // 4
                    self.zoom_level = 1.0
                    self.zoom_label.config(text="1.0x")
                    
                    self.root.after(0, self.display_result, result)
                else:
                    self.root.after(0, lambda: messagebox.showerror("Error", "Failed to process image"))
                    
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Processing failed: {e}"))
            finally:
                self.root.after(0, lambda: self.status_var.set("Ready"))
                self.root.after(0, lambda: setattr(self, 'is_processing', False))
                self.root.after(0, lambda: self.progress_var.set(0))
        
        threading.Thread(target=process_thread, daemon=True).start()
    
    def process_chunkwise(self, downsample, chunk_size):
        """Process image using chunkwise downsampling and concatenation"""
        try:
            width, height = self.image_size
            
            # Calculate downsampled dimensions
            ds_width = width // downsample
            ds_height = height // downsample
            
            # Create result array
            if self.image_mode == "L":  # Grayscale
                result = np.zeros((ds_height, ds_width), dtype=np.uint8)
            else:  # Color
                channels = 3 if self.image_mode == "RGB" else 4
                result = np.zeros((ds_height, ds_width, channels), dtype=np.uint8)
            
            # Calculate number of chunks
            chunks_x = (width + chunk_size - 1) // chunk_size
            chunks_y = (height + chunk_size - 1) // chunk_size
            total_chunks = chunks_x * chunks_y
            
            current_chunk = 0
            
            with Image.open(self.tiff_path) as tiff:
                if self.total_pages > 1:
                    tiff.seek(self.current_page)
                
                # Process each chunk
                for chunk_y in range(chunks_y):
                    for chunk_x in range(chunks_x):
                        # Calculate chunk coordinates
                        x_start = chunk_x * chunk_size
                        y_start = chunk_y * chunk_size
                        x_end = min(x_start + chunk_size, width)
                        y_end = min(y_start + chunk_size, height)
                        
                        # Update progress
                        current_chunk += 1
                        progress = int((current_chunk / total_chunks) * 100)
                        self.root.after(0, lambda p=progress: self.progress_var.set(p))
                        self.root.after(0, lambda: self.status_var.set(f"Processing chunk {current_chunk}/{total_chunks}"))
                        
                        # Extract chunk
                        chunk_region = (x_start, y_start, x_end, y_end)
                        chunk = tiff.crop(chunk_region)
                        
                        # Downsample chunk
                        chunk_ds_width = (x_end - x_start) // downsample
                        chunk_ds_height = (y_end - y_start) // downsample
                        
                        if chunk_ds_width > 0 and chunk_ds_height > 0:
                            chunk_ds = chunk.resize((chunk_ds_width, chunk_ds_height), 
                                                   Image.Resampling.LANCZOS)
                            
                            # Convert to numpy array
                            chunk_array = np.array(chunk_ds)
                            
                            # Calculate position in result array
                            result_x = x_start // downsample
                            result_y = y_start // downsample
                            
                            # Place chunk in result
                            if len(result.shape) == 2:  # Grayscale
                                result[result_y:result_y+chunk_ds_height, 
                                      result_x:result_x+chunk_ds_width] = chunk_array
                            else:  # Color
                                result[result_y:result_y+chunk_ds_height, 
                                      result_x:result_x+chunk_ds_width, :] = chunk_array
                        
                        # Force garbage collection periodically
                        if current_chunk % 10 == 0:
                            gc.collect()
                
                return result
                
        except Exception as e:
            print(f"Chunkwise processing failed: {e}")
            return None
    
    def display_result(self, image_data):
        """Display the processed image"""
        try:
            self.current_display_data = image_data
            
            # Clear previous plot
            self.ax.clear()
            
            # Get original image dimensions
            width, height = self.image_size
            
            # Display image with absolute pixel coordinates
            if len(image_data.shape) == 2:  # Grayscale
                im = self.ax.imshow(image_data, cmap='gray', aspect='equal',
                                   extent=[0, width, height, 0])
                # Add colorbar
                self.fig.colorbar(im, ax=self.ax, shrink=0.8)
            else:  # Color
                im = self.ax.imshow(image_data, aspect='equal',
                                   extent=[0, width, height, 0])
            
            # Update title and information
            downsample = int(self.downsample_var.get())
            title = f"Downsampled Image (1:{downsample}) - Size: {image_data.shape[1]}x{image_data.shape[0]}"
            self.ax.set_title(title)
            
            # Set axis labels
            self.ax.set_xlabel('X (pixels)')
            self.ax.set_ylabel('Y (pixels)')
            
            # Update file information with actual memory usage
            actual_memory = image_data.nbytes / 1024 / 1024
            info = f"File: {os.path.basename(self.tiff_path)}\n"
            info += f"Original Size: {self.image_size}\n"
            info += f"Downsampled Size: {image_data.shape[1]}x{image_data.shape[0]}\n"
            info += f"Downsample Factor: 1:{downsample}\n"
            info += f"Actual Memory Usage: {actual_memory:.2f} MB\n"
            info += f"Compression Ratio: {self.image_size[0]*self.image_size[1]/(image_data.shape[0]*image_data.shape[1]):.1f}x"
            
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(1.0, info)
            
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display image: {e}")
    
    def zoom_to_region(self):
        """Zoom to a specific region defined by coordinates"""
        if not self.tiff_path:
            messagebox.showwarning("Warning", "Please open a TIFF file first")
            return
            
        try:
            # Get coordinates from input fields
            x1 = int(self.x1_var.get())
            y1 = int(self.y1_var.get())
            x2 = int(self.x2_var.get())
            y2 = int(self.y2_var.get())
            
            # Validate coordinates
            width, height = self.image_size
            if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x1 >= x2 or y1 >= y2:
                messagebox.showerror("Error", "Invalid coordinates. Please ensure:\n"
                                            f"- X1 < X2 and Y1 < Y2\n"
                                            f"- Coordinates within image bounds (0-{width}, 0-{height})")
                return
            
            # Calculate region center and size
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            region_width = x2 - x1
            region_height = y2 - y1
            
            # Update viewport
            self.view_center = (center_x, center_y)
            self.view_width = region_width
            self.view_height = region_height
            
            # Calculate zoom level based on region size relative to image size
            image_width, image_height = self.image_size
            self.zoom_level = max(image_width / region_width, image_height / region_height)
            
            # Update zoom level display
            self.zoom_label.config(text=f"{self.zoom_level:.1f}x")
            
            # Process and display the zoomed region with Zoom DS setting
            self.process_zoomed_region()
            
        except ValueError:
            messagebox.showerror("Error", "Please enter valid integer coordinates")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to zoom to region: {e}")
    
    def process_zoomed_region(self):
        """Process and display the current zoomed region"""
        if not self.tiff_path or self.view_center is None:
            return
            
        if self.is_processing:
            return
            
        self.is_processing = True
        self.status_var.set("Processing zoomed region...")
        
        def process_thread():
            try:
                # Calculate region bounds
                center_x, center_y = self.view_center
                width, height = self.image_size
                
                # Calculate region boundaries
                left = max(0, center_x - self.view_width // 2)
                top = max(0, center_y - self.view_height // 2)
                right = min(width, center_x + self.view_width // 2)
                bottom = min(height, center_y + self.view_height // 2)
                
                # Use user-selected zoom downsampling instead of automatic adaptive downsampling
                zoom_downsample = int(self.zoom_ds_var.get())
                
                # Process the zoomed region with user-selected downsampling
                result = self.process_region(left, top, right, bottom, zoom_downsample)
                
                if result is not None:
                    self.root.after(0, self.display_zoomed_result, result, left, top, right, bottom, zoom_downsample)
                else:
                    self.root.after(0, lambda: messagebox.showerror("Error", "Failed to process zoomed region"))
                    
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Zoom processing failed: {e}"))
            finally:
                self.root.after(0, lambda: self.status_var.set("Ready"))
                self.root.after(0, lambda: setattr(self, 'is_processing', False))
        
        threading.Thread(target=process_thread, daemon=True).start()
    
    def process_region(self, left, top, right, bottom, downsample):
        """Process a specific region of the image"""
        try:
            region_width = right - left
            region_height = bottom - top
            
            # Calculate downsampled dimensions
            ds_width = region_width // downsample
            ds_height = region_height // downsample
            
            if ds_width <= 0 or ds_height <= 0:
                return None
            
            # Create result array
            if self.image_mode == "L":  # Grayscale
                result = np.zeros((ds_height, ds_width), dtype=np.uint8)
            else:  # Color
                channels = 3 if self.image_mode == "RGB" else 4
                result = np.zeros((ds_height, ds_width, channels), dtype=np.uint8)
            
            # Process region in chunks
            chunk_size = int(self.chunk_var.get())
            chunks_x = (region_width + chunk_size - 1) // chunk_size
            chunks_y = (region_height + chunk_size - 1) // chunk_size
            total_chunks = chunks_x * chunks_y
            
            current_chunk = 0
            
            with Image.open(self.tiff_path) as tiff:
                if self.total_pages > 1:
                    tiff.seek(self.current_page)
                
                for chunk_y in range(chunks_y):
                    for chunk_x in range(chunks_x):
                        # Calculate chunk coordinates in original image
                        chunk_left = left + chunk_x * chunk_size
                        chunk_top = top + chunk_y * chunk_size
                        chunk_right = min(chunk_left + chunk_size, right)
                        chunk_bottom = min(chunk_top + chunk_size, bottom)
                        
                        # Update progress
                        current_chunk += 1
                        progress = int((current_chunk / total_chunks) * 100)
                        self.root.after(0, lambda p=progress: self.progress_var.set(p))
                        self.root.after(0, lambda: self.status_var.set(f"Processing chunk {current_chunk}/{total_chunks}"))
                        
                        # Extract and process chunk
                        chunk_region = (chunk_left, chunk_top, chunk_right, chunk_bottom)
                        chunk = tiff.crop(chunk_region)
                        
                        # Downsample chunk
                        chunk_ds_width = (chunk_right - chunk_left) // downsample
                        chunk_ds_height = (chunk_bottom - chunk_top) // downsample
                        
                        if chunk_ds_width > 0 and chunk_ds_height > 0:
                            chunk_ds = chunk.resize((chunk_ds_width, chunk_ds_height), 
                                                   Image.Resampling.LANCZOS)
                            
                            # Convert to numpy array
                            chunk_array = np.array(chunk_ds)
                            
                            # Calculate position in result array
                            result_x = (chunk_left - left) // downsample
                            result_y = (chunk_top - top) // downsample
                            
                            # Place chunk in result
                            if len(result.shape) == 2:  # Grayscale
                                result[result_y:result_y+chunk_ds_height, 
                                      result_x:result_x+chunk_ds_width] = chunk_array
                            else:  # Color
                                result[result_y:result_y+chunk_ds_height, 
                                      result_x:result_x+chunk_ds_width, :] = chunk_array
                        
                        # Force garbage collection periodically
                        if current_chunk % 10 == 0:
                            gc.collect()
                
                return result
                
        except Exception as e:
            print(f"Region processing failed: {e}")
            return None
    
    def display_zoomed_result(self, image_data, left, top, right, bottom, zoom_downsample):
        """Display the zoomed region"""
        try:
            self.current_display_data = image_data
            
            # Clear previous plot
            self.ax.clear()
            
            # Calculate absolute pixel coordinates for axis ticks
            region_width = right - left
            region_height = bottom - top
            
            # Display image with absolute pixel coordinates
            if len(image_data.shape) == 2:  # Grayscale
                im = self.ax.imshow(image_data, cmap='gray', aspect='equal',
                                   extent=[left, right, bottom, top])
                # Add colorbar
                self.fig.colorbar(im, ax=self.ax, shrink=0.8)
            else:  # Color
                im = self.ax.imshow(image_data, aspect='equal',
                                   extent=[left, right, bottom, top])
            
            # Update title and information
            title = f"Zoomed Region (1:{zoom_downsample}) - Zoom: {self.zoom_level:.1f}x"
            self.ax.set_title(title)
            
            # Set axis labels
            self.ax.set_xlabel('X (pixels)')
            self.ax.set_ylabel('Y (pixels)')
            
            # Update zoom level display
            self.zoom_label.config(text=f"{self.zoom_level:.1f}x")
            
            # Update file information
            actual_memory = image_data.nbytes / 1024 / 1024
            info = f"File: {os.path.basename(self.tiff_path)}\n"
            info += f"Region: ({left}, {top}) to ({right}, {bottom})\n"
            info += f"Region Size: {right-left}x{bottom-top}\n"
            info += f"Display Size: {image_data.shape[1]}x{image_data.shape[0]}\n"
            info += f"Zoom Level: {self.zoom_level:.1f}x\n"
            info += f"Zoom Downsample: 1:{zoom_downsample}\n"
            info += f"Memory Usage: {actual_memory:.2f} MB"
            
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(1.0, info)
            
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display zoomed region: {e}")
    
    
    def zoom_in(self):
        """Zoom in at current view center"""
        if self.view_center is not None and self.tiff_path:
            try:
                # Update zoom level
                self.zoom_level *= 1.2
                
                # Update viewport size (smaller viewport = more zoom)
                self.view_width = max(100, int(self.view_width / 1.2))
                self.view_height = max(100, int(self.view_height / 1.2))
                
                # Update zoom level display
                self.zoom_label.config(text=f"{self.zoom_level:.1f}x")
                
                # Process and display the zoomed region
                self.process_zoomed_region()
            except Exception as e:
                print(f"Zoom in error: {e}")
    
    def zoom_out(self):
        """Zoom out at current view center"""
        if self.view_center is not None and self.tiff_path:
            try:
                # Update zoom level
                self.zoom_level /= 1.2
                
                # Update viewport size (larger viewport = less zoom)
                width, height = self.image_size
                self.view_width = min(width, int(self.view_width * 1.2))
                self.view_height = min(height, int(self.view_height * 1.2))
                
                # Update zoom level display
                self.zoom_label.config(text=f"{self.zoom_level:.1f}x")
                
                # Process and display the zoomed region
                self.process_zoomed_region()
            except Exception as e:
                print(f"Zoom out error: {e}")
    
    def reset_view(self):
        """Reset to full image view"""
        if self.image_size is not None:
            width, height = self.image_size
            self.view_center = (width // 2, height // 2)
            self.view_width = width // 4
            self.view_height = height // 4
            self.zoom_level = 1.0
            self.zoom_label.config(text="1.0x")
            self.process_image()

def main():
    root = tk.Tk()
    app = ChunkwiseTiffViewer(root)
    root.mainloop()

if __name__ == "__main__":
    main()
