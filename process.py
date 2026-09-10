import cv2
import numpy as np
import networkx as nx
from ultralytics import YOLO
import math
import time
import traceback

class EvacuationProcessor:
    def __init__(self):
        # ========== IMPROVED: YOLOv8m for better accuracy ==========
        self.model = YOLO('yolov8m.pt')
        self.grid_size = 40
        self.alpha = 0.1
        self.prev_cckg = {}
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        
        # For speed optimization
        self.frame_skip = 1
        self.frame_counter = 0
        
        # Exit management
        self.exits = []
        self.exit_capacities = {}
        self.exit_loads = {}
        
        # Cache for skipped frames
        self._cached_result = None
        
        # ========== IMPROVED: DETECTION SETTINGS ==========
        # Higher resolution for small objects
        self.inference_size = 1280
        
        # Multi-scale inference settings
        self.scales = [0.8, 1.0, 1.2]  # Scales to test
        
        # Confidence thresholds
        self.primary_conf = 0.15
        self.secondary_conf = 0.08
        self.tiled_conf = 0.10
        
        # NMS settings - adaptive based on box size
        self.base_nms = 0.40
        self.small_box_nms = 0.30  # Stricter for small boxes
        self.large_box_nms = 0.50  # Looser for large boxes
        
        # Person class ID
        self.person_class_id = 0
        
        # ========== IMPROVED: TILED INFERENCE SETTINGS ==========
        self.tile_size = 640
        self.tile_overlap = 0.2  # 20% overlap between tiles
        
        # Image enhancement
        self.use_clahe = True
        self.use_gamma = True
        self.gamma = 1.2  # Slight gamma correction for better contrast
        
        # Edge padding for better boundary detection
        self.edge_padding = 50
        
    def set_exits(self, exits):
        """Set multiple exits with capacities"""
        self.exits = exits if exits else []
        for i, (x, y) in enumerate(self.exits):
            self.exit_capacities[i] = 20
            self.exit_loads[i] = 0
        print(f"✅ Exits set: {len(self.exits)} exits")

    def enhance_image(self, frame):
        """
        ========== IMPROVED: Advanced image enhancement ==========
        Combines CLAHE and Gamma correction for better detection
        """
        try:
            # CLAHE for contrast enhancement
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            cl = self.clahe.apply(l)
            enhanced = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # Gamma correction for better brightness
            if self.use_gamma:
                inv_gamma = 1.0 / self.gamma
                table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                enhanced = cv2.LUT(enhanced, table)
            
            return enhanced
        except Exception:
            return frame

    def preprocess_for_detection(self, frame):
        """
        ========== IMPROVED: Preprocess frame for optimal detection ==========
        """
        processed = frame.copy()
        
        # Contrast enhancement for dark scenes
        avg_brightness = np.mean(frame)
        if avg_brightness < 80:
            processed = self.enhance_image(processed)
        
        # Sharpening for better edges
        kernel = np.array([
            [-0.1, -0.1, -0.1],
            [-0.1,  1.8, -0.1],
            [-0.1, -0.1, -0.1]
        ])
        sharpened = cv2.filter2D(processed, -1, kernel)
        processed = cv2.addWeighted(processed, 0.7, sharpened, 0.3, 0)
        
        return processed

    def get_adaptive_nms(self, box):
        """
        ========== IMPROVED: Adaptive NMS based on box size ==========
        Smaller boxes get stricter NMS to avoid suppressing nearby detections
        """
        x1, y1, x2, y2 = box[:4]
        width = x2 - x1
        height = y2 - y1
        area = width * height
        
        # Very small boxes (distant people) - stricter NMS
        if area < 1000:
            return self.small_box_nms
        # Very large boxes (close people) - looser NMS
        elif area > 5000:
            return self.large_box_nms
        else:
            return self.base_nms

    def detect_people(self, frame):
        """
        ========== IMPROVED: Advanced Multi-Scale + Tiled Detection ==========
        """
        person_boxes = []
        h, w = frame.shape[:2]
        
        try:
            # 1. Preprocess
            processed_frame = self.preprocess_for_detection(frame)
            
            # 2. Primary detection with original size
            results = self.model(
                processed_frame,
                imgsz=self.inference_size,
                conf=self.primary_conf,
                iou=self.base_nms,
                classes=[self.person_class_id],
                verbose=False,
                max_det=500,
                agnostic_nms=False,
            )
            
            # Extract detections
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        if conf >= self.primary_conf:
                            person_boxes.append([x1, y1, x2, y2, conf])
            
            # ========== IMPROVED: Multi-Scale Inference ==========
            # Process at different scales to detect people of different sizes
            for scale in self.scales:
                if scale == 1.0:
                    continue
                
                scaled_w = int(w * scale)
                scaled_h = int(h * scale)
                
                if scaled_w < 320 or scaled_h < 240:
                    continue
                
                scaled_frame = cv2.resize(processed_frame, (scaled_w, scaled_h))
                
                results_scaled = self.model(
                    scaled_frame,
                    imgsz=min(self.inference_size, max(scaled_w, scaled_h)),
                    conf=self.primary_conf * 0.9,
                    iou=self.base_nms * 0.9,
                    classes=[self.person_class_id],
                    verbose=False,
                    max_det=400,
                )
                
                for result in results_scaled:
                    if result.boxes is not None:
                        for box in result.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            conf = float(box.conf[0])
                            
                            # Scale back to original coordinates
                            x1 = x1 / scale
                            y1 = y1 / scale
                            x2 = x2 / scale
                            y2 = y2 / scale
                            
                            # Check if this detection is significantly different from existing
                            is_duplicate = False
                            for existing in person_boxes:
                                ex1, ey1, ex2, ey2, _ = existing
                                # Stricter IoU check for duplicates
                                ix1 = max(x1, ex1)
                                iy1 = max(y1, ey1)
                                ix2 = min(x2, ex2)
                                iy2 = min(y2, ey2)
                                
                                if ix2 > ix1 and iy2 > iy1:
                                    intersection = (ix2 - ix1) * (iy2 - iy1)
                                    area1 = (x2 - x1) * (y2 - y1)
                                    area2 = (ex2 - ex1) * (ey2 - ey1)
                                    union = area1 + area2 - intersection
                                    if union > 0 and intersection / union > 0.3:
                                        is_duplicate = True
                                        break
                            
                            if not is_duplicate and conf >= self.primary_conf * 0.8:
                                person_boxes.append([x1, y1, x2, y2, conf])
            
            # ========== IMPROVED: Tiled Inference for Small Objects ==========
            # Process image in tiles to detect small people with higher resolution
            tile_stride = int(self.tile_size * (1 - self.tile_overlap))
            
            # Only use tiled inference if image is large enough
            if w > self.tile_size or h > self.tile_size:
                for y in range(0, h, tile_stride):
                    for x in range(0, w, tile_stride):
                        # Get tile with padding at edges
                        x1 = x
                        y1 = y
                        x2 = min(x + self.tile_size, w)
                        y2 = min(y + self.tile_size, h)
                        
                        # Skip if tile is too small
                        if (x2 - x1) < 200 or (y2 - y1) < 200:
                            continue
                        
                        # Extract tile
                        tile = processed_frame[y1:y2, x1:x2]
                        
                        # Run detection on tile
                        results_tile = self.model(
                            tile,
                            imgsz=max(tile.shape[:2]),
                            conf=self.tiled_conf,
                            iou=self.base_nms * 0.8,
                            classes=[self.person_class_id],
                            verbose=False,
                            max_det=200,
                        )
                        
                        for result in results_tile:
                            if result.boxes is not None:
                                for box in result.boxes:
                                    tx1, ty1, tx2, ty2 = box.xyxy[0].tolist()
                                    conf = float(box.conf[0])
                                    
                                    # Convert to global coordinates
                                    gx1 = x1 + tx1
                                    gy1 = y1 + ty1
                                    gx2 = x1 + tx2
                                    gy2 = y1 + ty2
                                    
                                    # Check if this detection is new
                                    is_duplicate = False
                                    for existing in person_boxes:
                                        ex1, ey1, ex2, ey2, _ = existing
                                        ix1 = max(gx1, ex1)
                                        iy1 = max(gy1, ey1)
                                        ix2 = min(gx2, ex2)
                                        iy2 = min(gy2, ey2)
                                        
                                        if ix2 > ix1 and iy2 > iy1:
                                            intersection = (ix2 - ix1) * (iy2 - iy1)
                                            area1 = (gx2 - gx1) * (gy2 - gy1)
                                            area2 = (ex2 - ex1) * (ey2 - ey1)
                                            union = area1 + area2 - intersection
                                            if union > 0 and intersection / union > 0.3:
                                                is_duplicate = True
                                                break
                                    
                                    if not is_duplicate and conf >= self.tiled_conf:
                                        person_boxes.append([gx1, gy1, gx2, gy2, conf])
            
            # ========== Secondary Pass for Missed Detections ==========
            # Only if we have few detections
            if len(person_boxes) < 15:
                results_secondary = self.model(
                    processed_frame,
                    imgsz=self.inference_size,
                    conf=self.secondary_conf,
                    iou=self.base_nms * 0.8,
                    classes=[self.person_class_id],
                    verbose=False,
                    max_det=500,
                )
                
                for result in results_secondary:
                    if result.boxes is not None:
                        for box in result.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            conf = float(box.conf[0])
                            
                            if conf < self.primary_conf and conf >= self.secondary_conf:
                                # Check if this is a new detection
                                is_duplicate = False
                                for existing in person_boxes:
                                    ex1, ey1, ex2, ey2, _ = existing
                                    ix1 = max(x1, ex1)
                                    iy1 = max(y1, ey1)
                                    ix2 = min(x2, ex2)
                                    iy2 = min(y2, ey2)
                                    
                                    if ix2 > ix1 and iy2 > iy1:
                                        intersection = (ix2 - ix1) * (iy2 - iy1)
                                        area1 = (x2 - x1) * (y2 - y1)
                                        area2 = (ex2 - ex1) * (ey2 - ey1)
                                        union = area1 + area2 - intersection
                                        if union > 0 and intersection / union > 0.3:
                                            is_duplicate = True
                                            break
                                
                                width = x2 - x1
                                height = y2 - y1
                                
                                if not is_duplicate and width > 8 and height > 15:
                                    person_boxes.append([x1, y1, x2, y2, conf])
            
            # ========== Final Cleanup: Remove Duplicates ==========
            if len(person_boxes) > 1:
                unique_boxes = []
                for box in person_boxes:
                    x1, y1, x2, y2, conf = box
                    is_duplicate = False
                    
                    for existing in unique_boxes:
                        ex1, ey1, ex2, ey2, econf = existing
                        ix1 = max(x1, ex1)
                        iy1 = max(y1, ey1)
                        ix2 = min(x2, ex2)
                        iy2 = min(y2, ey2)
                        
                        if ix2 > ix1 and iy2 > iy1:
                            intersection = (ix2 - ix1) * (iy2 - iy1)
                            area1 = (x2 - x1) * (y2 - y1)
                            area2 = (ex2 - ex1) * (ey2 - ey1)
                            union = area1 + area2 - intersection
                            if union > 0 and intersection / union > 0.35:
                                # Keep the one with higher confidence
                                if conf > econf:
                                    # Replace existing with this one
                                    existing_index = unique_boxes.index(existing)
                                    unique_boxes[existing_index] = box
                                is_duplicate = True
                                break
                    
                    if not is_duplicate:
                        unique_boxes.append(box)
                
                person_boxes = unique_boxes
            
        except Exception as e:
            print(f"⚠️ YOLO detection error: {e}")
            traceback.print_exc()
            person_boxes = []
        
        # Optional: Debug output
        # if len(person_boxes) > 0:
        #     print(f"🔍 Detected: {len(person_boxes)} people")
        
        return person_boxes

    def get_gravity(self, dist_to_exit, density):
        g_static = 1000 / (dist_to_exit + 1)
        g_dynamic = math.exp(-self.alpha * density)
        return g_static * g_dynamic

    def compute_gravity_field(self, grid, exits):
        """Calculate gravity field for multiple exits"""
        rows, cols = grid.shape
        if len(exits) == 0:
            gravity_field = np.zeros((rows, cols))
            exit_gravity = np.zeros((rows, cols, 1))
            return gravity_field, exit_gravity
            
        gravity_field = np.zeros((rows, cols))
        exit_gravity = np.zeros((rows, cols, len(exits)))
        
        for r in range(rows):
            for c in range(cols):
                center_x = c * self.grid_size + self.grid_size // 2
                center_y = r * self.grid_size + self.grid_size // 2
                
                for i, (ex, ey) in enumerate(exits):
                    dist = math.sqrt((center_x - ex)**2 + (center_y - ey)**2)
                    g_static = 1000 / (dist + 1)
                    density = grid[r][c]
                    g_dynamic = math.exp(-self.alpha * density)
                    exit_gravity[r][c][i] = g_static * g_dynamic
                
                total_gravity = 0
                for i in range(len(exits)):
                    load_factor = 1 - (self.exit_loads.get(i, 0) / self.exit_capacities.get(i, 20))
                    load_factor = max(load_factor, 0.1)
                    total_gravity += exit_gravity[r][c][i] * load_factor
                
                gravity_field[r][c] = total_gravity
        
        return gravity_field, exit_gravity

    def predict_density_next(self, current_grid):
        """Predict next density using simplified model"""
        rows, cols = current_grid.shape
        predicted = np.zeros((rows, cols))
        
        for r in range(rows):
            for c in range(cols):
                neighbors = []
                for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        neighbors.append(current_grid[nr][nc])
                
                if neighbors:
                    avg_neighbor = np.mean(neighbors)
                    predicted[r][c] = current_grid[r][c] * 0.6 + avg_neighbor * 0.4
        
        return np.maximum(predicted, 0)

    def assign_exits(self, density_grid, exit_gravity):
        """Assign each cell to the best exit"""
        rows, cols = density_grid.shape
        exit_assignment = np.zeros((rows, cols), dtype=int)
        exit_distances = np.zeros((rows, cols))
        
        if len(self.exits) == 0:
            return exit_assignment, exit_distances
        
        for r in range(rows):
            for c in range(cols):
                best_exit = 0
                best_score = -1
                
                for i in range(len(self.exits)):
                    load_factor = self.exit_loads.get(i, 0) / max(self.exit_capacities.get(i, 20), 1)
                    load_factor = min(load_factor, 0.9)
                    score = exit_gravity[r][c][i] * (1 - load_factor * 0.5)
                    
                    if score > best_score:
                        best_score = score
                        best_exit = i
                
                exit_assignment[r][c] = best_exit
                ex, ey = self.exits[best_exit]
                center_x = c * self.grid_size + self.grid_size // 2
                center_y = r * self.grid_size + self.grid_size // 2
                exit_distances[r][c] = math.sqrt((center_x - ex)**2 + (center_y - ey)**2)
        
        return exit_assignment, exit_distances

    def process_frame(self, frame, exits):
        """Optimized frame processing - ALWAYS RETURNS 10 VALUES"""
        try:
            self.frame_counter += 1
            
            # Skip frames for speed
            if self.frame_counter % self.frame_skip != 0:
                if self._cached_result is not None:
                    return self._cached_result
            
            # Set exits if changed
            if exits != self.exits:
                self.set_exits(exits)
            
            # If no exits, use default paper-based exits
            if len(self.exits) == 0:
                print("⚠️ No exits configured, using defaults")
                self.exits = [(320, 30), (60, 250), (580, 250)]
                self.set_exits(self.exits)
            
            # Get frame dimensions
            h, w, _ = frame.shape
            rows = max(1, h // self.grid_size)
            cols = max(1, w // self.grid_size)
            
            # ========== ENHANCED PERSON DETECTION ==========
            person_boxes = self.detect_people(frame)
            
            # Build density grid
            cckg = {}
            density_grid = np.zeros((rows, cols))
            
            for r in range(rows):
                for c in range(cols):
                    node_id = (r, c)
                    count = 0
                    for b in person_boxes:
                        cx = (b[0] + b[2]) / 2
                        cy = (b[1] + b[3]) / 2
                        if (c * self.grid_size <= cx < (c + 1) * self.grid_size and
                            r * self.grid_size <= cy < (r + 1) * self.grid_size):
                            count += 1
                    
                    curr_density = count
                    prev_density = self.prev_cckg.get(node_id, curr_density)
                    predicted_density = (curr_density + prev_density) * 0.5
                    cckg[node_id] = predicted_density
                    density_grid[r][c] = predicted_density
            
            self.prev_cckg = cckg
            
            # Compute gravity field
            gravity_field, exit_gravity = self.compute_gravity_field(density_grid, self.exits)
            
            # Predict next density
            next_density = self.predict_density_next(density_grid)
            
            # Assign exits
            exit_assignment, exit_distances = self.assign_exits(density_grid, exit_gravity)
            
            # Update exit loads
            self.exit_loads = {i: 0 for i in range(len(self.exits))}
            for person in person_boxes:
                cx = int((person[0] + person[2]) / 2)
                cy = int((person[1] + person[3]) / 2)
                row = min(rows - 1, max(0, cy // self.grid_size))
                col = min(cols - 1, max(0, cx // self.grid_size))
                exit_id = exit_assignment[row][col]
                if 0 <= exit_id < len(self.exits):
                    self.exit_loads[exit_id] = self.exit_loads.get(exit_id, 0) + 1
            
            # Build graph
            G = nx.DiGraph()
            for r in range(rows):
                for c in range(cols):
                    center_x, center_y = c * self.grid_size, r * self.grid_size
                    if self.exits:
                        min_dist = min([math.sqrt((center_x - ex)**2 + (center_y - ey)**2) 
                                       for ex, ey in self.exits])
                    else:
                        min_dist = float("inf")
                    gravity = self.get_gravity(min_dist, cckg.get((r, c), 0))
                    
                    for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            G.add_edge((r, c), (nr, nc), 
                                      weight=1/(gravity + 0.0001),
                                      gravity_weight=1/(gravity_field[r][c] + 0.0001))
            
            # Find paths to exits
            paths = {}
            for i, (ex, ey) in enumerate(self.exits):
                try:
                    exit_node = (ey // self.grid_size, ex // self.grid_size)
                    start_node = (rows - 1, cols // 2)
                    path = nx.shortest_path(
                        G,
                        source=start_node,
                        target=exit_node,
                        weight='gravity_weight'
                    )
                    paths[i] = path
                except nx.NetworkXNoPath:
                    try:
                        start_node = (0, cols // 2)
                        path = nx.shortest_path(
                            G,
                            source=start_node,
                            target=exit_node,
                            weight='gravity_weight'
                        )
                        paths[i] = path
                    except:
                        paths[i] = None
                except Exception as e:
                    paths[i] = None
            
            # Build result with exactly 10 values
            result = (
                person_boxes,        # 0
                cckg,                # 1
                G,                   # 2
                rows,                # 3
                cols,                # 4
                gravity_field,       # 5
                next_density,        # 6
                paths,               # 7
                exit_assignment,     # 8
                exit_distances       # 9
            )
            
            self._cached_result = result
            return result
            
        except Exception as e:
            print(f"❌ process_frame error: {e}")
            traceback.print_exc()
            # Return empty valid result with 10 values
            empty_result = ([], {}, nx.DiGraph(), 6, 8, np.array([]), np.array([]), {}, np.array([]), np.array([]))
            self._cached_result = empty_result
            return empty_result