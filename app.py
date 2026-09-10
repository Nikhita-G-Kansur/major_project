import os
import cv2
import math
import time
import json
import numpy as np
import networkx as nx
from flask import Flask, render_template, Response, request, jsonify, send_file
from process import EvacuationProcessor
import threading
import traceback
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

processor = EvacuationProcessor()

# ============== EXITS - MANUALLY SELECTED ==============
EXITS = []
processor.set_exits(EXITS)

# ============== STATS USING LOCK FOR THREAD SAFETY ==============
stats_lock = threading.Lock()
current_stats = {
    'total_people': 0,
    'density_grid': [],
    'paths': {},
    'exit_assignment': [],
    'exit_loads': {},
    'fps': 0,
    'timestamp': 0,
    'frame_count': 0,
    'evacuation_time': 0,
    'density_level': 0,
    'status': 'idle',
    'rows': 0,
    'cols': 0,
    'exits': [],
    'evacuation_complete': False,
    'exits_detected': False
}

alerts = []
evacuation_start_time = None
evacuation_complete = False
exits_detected = False

# ============== CACHED STATS FOR 5-SECOND UPDATE ==============
last_stats_update = 0
cached_stats = current_stats.copy()
stats_update_interval = 1

def get_stats():
    """Get current stats (thread-safe)"""
    with stats_lock:
        return convert_to_serializable(current_stats.copy())

def update_stats(new_stats):
    """Update stats (thread-safe)"""
    global current_stats
    with stats_lock:
        current_stats = convert_to_serializable(new_stats.copy())

def convert_to_serializable(obj):
    """Convert numpy types to Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

# ============== ROUTES ==============

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/upload', methods=['POST'])
def upload():
    global evacuation_start_time, evacuation_complete, exits_detected, EXITS
    try:
        file = request.files['video']
        if not file:
            return jsonify({"status": "error", "message": "No file provided"}), 400
        
        path = os.path.join(app.config['UPLOAD_FOLDER'], 'input.mp4')
        file.save(path)
        
        # Validate video
        print("🔍 Validating video...")
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print("❌ Video is corrupted! Attempting to fix...")
            cap.release()
            temp_path = path + '.temp.mp4'
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
                
                frame_count = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    out.write(frame)
                    frame_count += 1
                
                cap.release()
                out.release()
                
                if frame_count > 0 and os.path.exists(temp_path):
                    os.remove(path)
                    os.rename(temp_path, path)
                    print(f"✅ Video fixed! {frame_count} frames")
                else:
                    print("❌ Could not fix video")
                    return jsonify({"status": "error", "message": "Video is corrupted"}), 400
            else:
                return jsonify({"status": "error", "message": "Cannot read video file"}), 400
        else:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            print(f"✅ Video validated: {frame_count} frames")
        
        evacuation_start_time = time.time()
        evacuation_complete = False
        exits_detected = False
        EXITS = []
        
        new_stats = current_stats.copy()
        new_stats['status'] = 'selecting'
        new_stats['exits'] = []
        new_stats['exits_detected'] = False
        update_stats(new_stats)
        
        print(f"✅ Video uploaded: {path}")
        print("🎯 Waiting for user to select exits...")
        return jsonify({"status": "success", "message": "Video uploaded successfully"})
    except Exception as e:
        print(f"❌ Upload error: {e}")
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/stats')
def get_stats_api():
    """Return current statistics with proper JSON serialization"""
    stats = get_stats()
    
    return jsonify({
        'total_people': int(stats.get('total_people', 0)),
        'density_grid': convert_to_serializable(stats.get('density_grid', [])),
        'paths': convert_to_serializable(stats.get('paths', {})),
        'exit_assignment': convert_to_serializable(stats.get('exit_assignment', [])),
        'exit_loads': convert_to_serializable(stats.get('exit_loads', {})),
        'fps': float(stats.get('fps', 0)),
        'timestamp': float(stats.get('timestamp', 0)),
        'frame_count': int(stats.get('frame_count', 0)),
        'evacuation_time': float(stats.get('evacuation_time', 0)),
        'density_level': float(stats.get('density_level', 0)),
        'status': str(stats.get('status', 'idle')),
        'rows': int(stats.get('rows', 0)),
        'cols': int(stats.get('cols', 0)),
        'exits': convert_to_serializable(stats.get('exits', [])),
        'evacuation_complete': bool(stats.get('evacuation_complete', False)),
        'exits_detected': bool(stats.get('exits_detected', False))
    })

@app.route('/api/alerts')
def get_alerts():
    return jsonify({'alerts': alerts[-20:], 'count': len(alerts)})

@app.route('/clear_alerts', methods=['POST'])
def clear_alerts():
    global alerts
    alerts = []
    return jsonify({"status": "success"})

@app.route('/api/export_data')
def export_data():
    stats = get_stats()
    data = {
        'timestamp': time.time(),
        'stats': convert_to_serializable(stats),
        'alerts': convert_to_serializable(alerts[-50:]),
        'exits': convert_to_serializable(stats.get('exits', []))
    }
    export_path = os.path.join(app.config['UPLOAD_FOLDER'], 'export_data.json')
    with open(export_path, 'w') as f:
        json.dump(data, f, indent=2)
    return send_file(export_path, as_attachment=True, download_name='evacuation_data.json')

# ============== ALERT SYSTEM ==============

def check_alerts(density_grid, total_people, paths, exit_loads):
    global alerts
    
    if not density_grid:
        return
    
    grid = np.array(density_grid)
    max_density = np.max(grid) if grid.size > 0 else 0
    
    if max_density >= 8:
        max_pos = np.unravel_index(np.argmax(grid), grid.shape)
        alerts.append({
            'type': 'CRITICAL',
            'severity': 3,
            'message': f'🚨 Extreme congestion at cell ({max_pos[0]}, {max_pos[1]})!',
            'timestamp': time.time()
        })
    elif max_density >= 5:
        max_pos = np.unravel_index(np.argmax(grid), grid.shape)
        alerts.append({
            'type': 'WARNING',
            'severity': 2,
            'message': f'⚠️ High density at cell ({max_pos[0]}, {max_pos[1]})',
            'timestamp': time.time()
        })
    
    if total_people >= 50:
        alerts.append({
            'type': 'CRITICAL',
            'severity': 3,
            'message': f'🚨 Overcrowding: {total_people} people!',
            'timestamp': time.time()
        })
    elif total_people >= 30:
        alerts.append({
            'type': 'WARNING',
            'severity': 2,
            'message': f'⚠️ High occupancy: {total_people} people',
            'timestamp': time.time()
        })
    
    for i, load in exit_loads.items():
        capacity = 20
        if load > capacity:
            alerts.append({
                'type': 'CRITICAL',
                'severity': 3,
                'message': f'🚨 Exit {int(i)+1} is OVERLOADED! ({int(load)} people)',
                'timestamp': time.time()
            })
        elif load > capacity * 0.8:
            alerts.append({
                'type': 'WARNING',
                'severity': 2,
                'message': f'⚠️ Exit {int(i)+1} at {int(load/capacity*100)}% capacity',
                'timestamp': time.time()
            })
    
    if len(alerts) > 50:
        alerts = alerts[-50:]

# ============== EXIT SELECTION (Manual) ==============
exit_selection_frame = None

@app.route('/api/get_first_frame')
def get_first_frame():
    """Return the first frame of the uploaded video for exit selection"""
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input.mp4')
    
    if not os.path.exists(video_path):
        return jsonify({"status": "error", "message": "No video uploaded"}), 404
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return jsonify({"status": "error", "message": "Cannot read video"}), 500
    
    # Store frame for exit selection
    global exit_selection_frame
    exit_selection_frame = frame.copy()
    
    # Encode frame to JPEG
    _, buffer = cv2.imencode('.jpg', frame)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        "status": "success",
        "image": img_base64,
        "width": frame.shape[1],
        "height": frame.shape[0]
    })

@app.route('/api/set_exits', methods=['POST'])
def set_exits():
    """Save the manually selected exits"""
    global EXITS, exits_detected
    
    try:
        data = request.get_json()
        exits = data.get('exits', [])
        
        if not exits or len(exits) < 1:
            return jsonify({"status": "error", "message": "Please select at least 1 exit"}), 400
        
        # Clear any existing exits
        EXITS = []
        
        # Save the selected exits
        for ex, ey in exits:
            EXITS.append((int(ex), int(ey)))
        
        # Set exits in processor
        processor.set_exits(EXITS)
        exits_detected = True
        
        # Update stats
        stats = get_stats()
        stats['exits'] = EXITS
        stats['exits_detected'] = True
        stats['status'] = 'processing'
        update_stats(stats)
        
        print(f"✅ Manually selected {len(EXITS)} exits: {EXITS}")
        
        return jsonify({
            "status": "success",
            "exits": EXITS,
            "count": len(EXITS)
        })
        
    except Exception as e:
        print(f"❌ Error setting exits: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/exit_status')
def exit_status():
    """Check if exits are set and return count"""
    return jsonify({
        "exits_set": exits_detected,
        "exit_count": len(EXITS),
        "exits": EXITS
    })

# ============== DRAWING FUNCTIONS ==============

def draw_exit_arrows(frame, exit_assignment, exits, grid_size):
    """
    Draw beautiful glowing arrows showing which exit each cell should go to.
    NO PATH LINES - only arrows for visual guidance.
    """
    if exit_assignment is None or len(exit_assignment) == 0 or len(exits) == 0:
        return
    
    rows, cols = exit_assignment.shape
    
    # Brighter, more attractive colors with glow
    colors = [
        (0, 180, 255),      # Bright Orange
        (255, 50, 150),     # Hot Pink
        (50, 255, 100),     # Neon Green
        (180, 50, 255),     # Purple
        (255, 200, 0),      # Gold
        (0, 255, 200),      # Cyan
        (255, 100, 50),     # Coral
        (200, 200, 255),    # Light Purple
    ]
    
    for r in range(rows):
        for c in range(cols):
            exit_id = exit_assignment[r][c]
            if exit_id >= 0 and exit_id < len(exits):
                center_x = c * grid_size + grid_size // 2
                center_y = r * grid_size + grid_size // 2
                ex, ey = exits[exit_id]
                
                dx = ex - center_x
                dy = ey - center_y
                dist = math.sqrt(dx**2 + dy**2)
                
                if dist > 15:
                    # Normalize direction
                    dx = dx / dist
                    dy = dy / dist
                    
                    # Arrow length based on distance
                    arrow_length = min(20, dist * 0.3)
                    end_x = int(center_x + dx * arrow_length)
                    end_y = int(center_y + dy * arrow_length)
                    
                    color = colors[exit_id % len(colors)]
                    
                    # ===== GLOW EFFECT =====
                    # Create a glow layer
                    glow_overlay = frame.copy()
                    
                    # Outer glow (larger, more transparent)
                    cv2.arrowedLine(
                        glow_overlay,
                        (center_x, center_y),
                        (end_x, end_y),
                        color,
                        6,  # Thicker for glow
                        tipLength=0.35
                    )
                    cv2.addWeighted(glow_overlay, 0.2, frame, 0.8, 0, frame)
                    
                    # Inner glow
                    glow_overlay2 = frame.copy()
                    cv2.arrowedLine(
                        glow_overlay2,
                        (center_x, center_y),
                        (end_x, end_y),
                        color,
                        4,
                        tipLength=0.35
                    )
                    cv2.addWeighted(glow_overlay2, 0.4, frame, 0.6, 0, frame)
                    
                    # Main arrow (bright, solid)
                    cv2.arrowedLine(
                        frame,
                        (center_x, center_y),
                        (end_x, end_y),
                        (255, 255, 255),  # White center for visibility
                        1,
                        tipLength=0.3
                    )
                    
                    # Colored arrow on top
                    cv2.arrowedLine(
                        frame,
                        (center_x, center_y),
                        (end_x, end_y),
                        color,
                        2,
                        tipLength=0.35
                    )

# ============== VIDEO PROCESSING ==============

def generate_frames():
    """Generate video frames with overlays and stats updates"""
    global EXITS, exits_detected, last_stats_update, cached_stats
    
    # Initialize cache variables
    last_stats_update = time.time()
    cached_stats = current_stats.copy()
    stats_update_interval = 1
    
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input.mp4')
    
    if not os.path.exists(video_path):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "No video uploaded", (180, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Please upload a video", (190, 280), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        return
    
    # Wait for exits to be selected
    while not exits_detected:
        time.sleep(0.1)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "🎯 Select exits on the main page", (160, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "Click on the first frame to mark exit locations", (150, 280), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        continue
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "Error opening video", (200, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        return
    
    frame_count = 0
    start_time = time.time()
    fps_counter = 0
    fps_time = start_time
    
    global evacuation_start_time, evacuation_complete
    
    display_skip = 2
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            stats = get_stats()
            stats['status'] = 'complete'
            stats['evacuation_complete'] = True
            update_stats(stats)
            continue
        
        frame_count += 1
        fps_counter += 1
        current_time = time.time()
        
        if current_time - fps_time >= 1.0:
            stats = get_stats()
            stats['fps'] = fps_counter
            update_stats(stats)
            fps_counter = 0
            fps_time = current_time
        
        if frame_count % display_skip == 0:
            try:
                # Process frame with detected exits
                result = processor.process_frame(frame, EXITS)
                
                # Safety check - handle None or invalid result
                if result is None:
                    print("⚠️ process_frame returned None")
                    continue
                
                if not isinstance(result, (list, tuple)):
                    print(f"⚠️ Invalid process_frame output: {type(result)}")
                    continue
                
                # Unpack result - handles 10 values
                if len(result) >= 10:
                    (boxes, cckg, G, rows, cols, gravity_field, 
                     next_density, paths, exit_assignment, exit_distances) = result[:10]
                elif len(result) >= 5:
                    boxes, cckg, G, rows, cols = result[:5]
                    paths = {}
                    exit_assignment = np.zeros((rows, cols), dtype=int)
                    gravity_field = None
                    next_density = None
                    exit_distances = None
                else:
                    print(f"⚠️ Incomplete process_frame output: {len(result)} values")
                    continue
                
                # Build density grid
                density_grid = []
                for r in range(rows):
                    row = []
                    for c in range(cols):
                        density = cckg.get((r, c), 0)
                        row.append(density)
                    density_grid.append(row)
                
                total_people = len(boxes)
                density_level = 0
                if density_grid:
                    flat = [d for row in density_grid for d in row]
                    if flat:
                        max_density = max(flat)
                        density_level = min(max_density * 10, 100)
                
                exit_loads = processor.exit_loads
                
                serializable_paths = {}
                for i, path in paths.items():
                    if path:
                        serializable_paths[str(i)] = [(int(p[0]), int(p[1])) for p in path]
                
                # ========== UPDATE STATS ==========
                if current_time - last_stats_update >= stats_update_interval:
                    new_stats = {
                        'total_people': int(total_people),
                        'density_grid': density_grid,
                        'density_level': float(density_level),
                        'frame_count': int(frame_count),
                        'timestamp': float(current_time),
                        'status': 'processing',
                        'rows': int(rows),
                        'cols': int(cols),
                        'exit_loads': {int(k): int(v) for k, v in exit_loads.items()},
                        'paths': serializable_paths,
                        'exit_assignment': exit_assignment.tolist() if exit_assignment is not None else [],
                        'exits': convert_to_serializable(EXITS),
                        'evacuation_complete': False,
                        'exits_detected': True,
                        'fps': get_stats().get('fps', 0)
                    }
                    
                    if evacuation_start_time and not evacuation_complete:
                        new_stats['evacuation_time'] = float(round(current_time - evacuation_start_time, 1))
                    
                    cached_stats = new_stats.copy()
                    update_stats(new_stats)
                    last_stats_update = current_time
                    
                    print(f"📊 Updated stats: {total_people} people, {density_level}% density, {len(EXITS)} exits")
                
                check_alerts(density_grid, total_people, paths, exit_loads)
                
                # --- DRAW OVERLAYS ---
                h, w = frame.shape[:2]
                grid_size = processor.grid_size
                
                # ===== TRANSPARENT HEATMAP =====
                overlay = frame.copy()
                for (r, c), density in cckg.items():
                    if density <= 0:
                        continue
                    if density <= 2:
                        color = (0, 255, 100)      # Bright Green
                    elif density <= 5:
                        color = (0, 255, 255)      # Yellow/Cyan
                    else:
                        color = (0, 50, 255)       # Bright Red
                    
                    cv2.rectangle(
                        overlay,
                        (c * grid_size, r * grid_size),
                        ((c + 1) * grid_size, (r + 1) * grid_size),
                        color,
                        -1
                    )
                
                # Blend overlay with original frame
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                
                # Draw density values on top of the heatmap
                for (r, c), density in cckg.items():
                    if density <= 0:
                        continue
                    cv2.putText(
                        frame,
                        f"{density:.1f}",
                        (c * grid_size + 5, r * grid_size + 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1
                    )
                
                # ===== DRAW EXIT ARROWS (NO PATH LINES) =====
                if exit_assignment is not None and len(exit_assignment) > 0 and len(EXITS) > 0:
                    draw_exit_arrows(frame, exit_assignment, EXITS, grid_size)
                
                # Draw person boxes with IDs and confidence
                for person_id, b in enumerate(boxes, start=1):
                    x1 = int(b[0])
                    y1 = int(b[1])
                    x2 = int(b[2])
                    y2 = int(b[3])
                    
                    # Person box with gradient effect
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 200), 2)
                    cv2.rectangle(frame, (x1, y1-22), (x1+110, y1), (0, 255, 200), -1)
                    
                    confidence = b[4] if len(b) > 4 else 0.0
                    label = f"P{person_id} {confidence * 100:.0f}%"
                    cv2.putText(frame, label, (x1+5, y1-6), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
                
                # ===== DRAW EXITS WITH GLOW =====
                exit_colors = [
                    (0, 200, 255),   # Bright Orange
                    (255, 50, 150),  # Hot Pink
                    (50, 255, 100),  # Neon Green
                    (180, 50, 255),  # Purple
                    (255, 200, 0),   # Gold
                    (0, 255, 200),   # Cyan
                ]
                
                for i, (ex, ey) in enumerate(EXITS):
                    color = exit_colors[i % len(exit_colors)]
                    
                    # Glow effect for exit
                    glow_overlay = frame.copy()
                    cv2.circle(glow_overlay, (ex, ey), 25, color, -1)
                    cv2.addWeighted(glow_overlay, 0.15, frame, 0.85, 0, frame)
                    
                    # Outer circle
                    cv2.circle(frame, (ex, ey), 18, color, 2)
                    
                    # Inner circle (filled)
                    cv2.circle(frame, (ex, ey), 10, color, -1)
                    
                    # Exit number
                    cv2.putText(frame, f"EXIT {i+1}", (ex-28, ey-22), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                    
                    # Load indicator
                    load = exit_loads.get(i, 0)
                    load_color = (0, 255, 100) if load < 14 else (0, 255, 255) if load < 20 else (0, 50, 255)
                    cv2.putText(frame, f"👥 {int(load)}/20", (ex-25, ey+35), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, load_color, 1)
                
                # ===== INFO OVERLAY =====
                cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)
                cv2.rectangle(frame, (0, 0), (w, 70), (100, 100, 100), 1)
                
                # Use cached stats for display (no flicker)
                display_people = cached_stats.get('total_people', total_people)
                display_density = cached_stats.get('density_level', density_level)
                display_evac_time = cached_stats.get('evacuation_time', 0)
                display_exit_loads = cached_stats.get('exit_loads', exit_loads)
                
                # Row 1: Main stats with icons
                people_color = (0, 255, 100) if display_people < 30 else (0, 255, 255) if display_people < 50 else (0, 50, 255)
                cv2.putText(frame, f" {display_people}", (15, 28), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, people_color, 2)
                cv2.putText(frame, f" {get_stats().get('fps', 0):.1f} FPS", (130, 28), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 255), 2)
                cv2.putText(frame, f" {display_density:.0f}% Density", (280, 28), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 2)
                cv2.putText(frame, f" {display_evac_time:.1f}s", (440, 28), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 2)
                
                # Row 2: Exit info
                cv2.putText(frame, f"🚪 Exits: {len(EXITS)}", (15, 52), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                x_offset = 130
                for i in range(len(EXITS)):
                    load = display_exit_loads.get(i, 0)
                    color = (0, 255, 100) if load < 14 else (0, 255, 255) if load < 20 else (0, 50, 255)
                    cv2.putText(frame, f"E{i+1}: {int(load)}", (x_offset, 52), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                    x_offset += 75
                
                # Update timer
                time_left = max(0, stats_update_interval - (current_time - last_stats_update))
                cv2.putText(frame, f"⏳ {time_left:.0f}s", (w-70, 28), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
                
            except Exception as e:
                print(f"Processing error:")
                traceback.print_exc()
                continue
        
        # Encode and yield
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ============== MAIN ==============

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 COMPLETE CROWD MANAGEMENT SYSTEM")
    print("=" * 60)
    print("📁 Upload folder: uploads/")
    print("🎯 Click exits on the first frame to select them")
    print("🗺️ Arrows guide people to exits (no path lines)")
    print("Website link: http://localhost:5000")
    print("📊 Dashboard: http://localhost:5000/dashboard")
    print("=" * 60)
    print("🔥 Press Ctrl+C to stop the server")
    print("=" * 60)
    
    app.run(debug=True, port=5000, threaded=True)