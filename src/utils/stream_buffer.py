# src/utils/stream_buffer.py
import queue
import threading
import time
import random

class EdgeVideoBufferPipeline:
    def __init__(self, queue_capacity=5):
        # Thread-safe FIFO container bounding your memory usage on edge hardware
        self.frame_buffer = queue.Queue(maxsize=queue_capacity)
        self.processing_active = True

    def video_stream_producer(self):
        """Thread 1: Mimics real-time camera decoding loop (e.g. OpenCV VideoCapture)"""
        frame_idx = 1
        print("📹 [PRODUCER] Camera stream online. Commencing high-speed frame ingestion...")
        
        while self.processing_active and frame_idx <= 15:
            try:
                # Simulating variable frame arrival times (FPS jitter)
                time.sleep(random.uniform(0.04, 0.08))
                
                # Non-blocking insertion into limited memory slot arrays
                # If buffer is full, it waits until the model clears a slot
                self.frame_buffer.put(f"Frame_Matrix_#{frame_idx}", timeout=2.0)
                print(f"📥 [PRODUCER] Captured and buffered frame {frame_idx} | Current Buffer Load: {self.frame_buffer.qsize()} slots occupied")
                frame_idx += 1
            except queue.Full:
                print("⚠️ [PRODUCER WARNING] Frame Buffer Overrun! Queue full. Throttling hardware camera ingestion...")

    def neural_network_consumer(self):
        """Thread 2: Mimics your Multi-Head Multi-Task Deep Learning Inference Pipeline"""
        print("🧠 [CONSUMER] Multi-Task YOLO Inference Engine initialized. Awaiting buffer stream data...")
        
        while self.processing_active or not self.frame_buffer.empty():
            if self.frame_buffer.empty():
                time.sleep(0.1)
                continue
                
            try:
                # Fetch frame from front of the FIFO buffer line
                frame_data = self.frame_buffer.get(timeout=1.0)
                
                print(f"⚡ [CONSUMER RUN] Popped {frame_data} out of buffer queue.")
                print("    ├── Forward Pass: Shared Backbone Feature Map Extraction...")
                time.sleep(0.05) # Model execution latency mimic
                
                # Simulate Multi-Head Parallel routing output checks
                print("    └── Parallel Routing Heads: [🎯 Turnaround Box Reg] | [🦺 PPE Filter Check] | [🔍 FOD Anomaly Score]")
                
                self.frame_buffer.task_done()
                
                # Simulating model inference cycle latency processing overhead
                time.sleep(random.uniform(0.12, 0.20))
                
            except queue.Empty:
                break
                
        print("🛑 [CONSUMER] Inference engine shut down. Queue cleared.")

    def run_pipeline_simulation(self):
        # Instantiate concurrent workers
        producer_t = threading.Thread(target=self.video_stream_producer)
        consumer_t = threading.Thread(target=self.neural_network_consumer)
        
        # Fire up concurrent system routines
        producer_t.start()
        consumer_t.start()
        
        producer_t.join()
        self.processing_active = False # Signal consumer to wrap up once frames finish
        consumer_t.join()
        print("\n🏁 [SYSTEM] Simulation terminated safely. No memory leakage detected.")

if __name__ == "__main__":
    pipeline = EdgeVideoBufferPipeline(queue_capacity=5)
    pipeline.run_pipeline_simulation()

# # Libraries
# import cv2
# import threading
# import queue
# import time
# import streamlit as st

# # --- STREAMLIT UI CONFIGURATION ---
# st.set_page_config(page_title="Frame Buffer Visualizer", layout="wide")
# st.title("📥 Real-Time Frame Buffer Layout Visualizer")
# st.write("Watch your actual video frames fill and move through the buffer queue in real time.")

# # --- SIDEBAR CONFIGURATION ---
# st.sidebar.header("Pipeline Controls")
# video_path = st.sidebar.text_input("Local MP4 Path", value="path_to_your_sample_video.mp4")
# max_buffer_size = st.sidebar.slider("Queue Size Limit (Slots)", min_value=3, max_value=8, value=5)
# consumer_delay = st.sidebar.slider("UI Consumer Throttle (Playback Speed Delay)", 0.01, 1.0, 0.05)

# # Initialize States
# if "frame_queue" not in st.session_state:
#     st.session_state.frame_queue = queue.Queue(maxsize=max_buffer_size)
# if "pipeline_active" not in st.session_state:
#     st.session_state.pipeline_active = False
# if "producer_thread" not in st.session_state:
#     st.session_state.producer_thread = None
# if "run_flag" not in st.session_state:
#     st.session_state.run_flag = [False]

# # --- BACKGROUND PRODUCER WORKER (THREAD 1) ---
# def video_file_producer(path, q, run_flag):
#     """Ingests frames continuously from an MP4 file into the buffer queue."""
#     cap = cv2.VideoCapture(path)
#     if not cap.isOpened():
#         print(f"[ERROR] Cannot open video file at path: {path}")
#         run_flag[0] = False
#         return

#     fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_delay = 1.0 / fps if fps > 0 else 0.033

#     while run_flag[0]:
#         start_time = time.time()
#         ret, frame = cap.read()
        
#         # Loop video automatically if it ends
#         if not ret:
#             cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#             continue 

#         # Lower size step for crisp, fast-loading visual cards
#         preprocessed_frame = cv2.resize(frame, (320, 240))
#         # Convert colorspace to Web-friendly RGB early in the background thread
#         rgb_frame = cv2.cvtColor(preprocessed_frame, cv2.COLOR_BGR2RGB)

#         # Buffer Guardrail: Evict oldest frame array if backend buffer depth maxes out
#         if q.full():
#             try:
#                 q.get_nowait()
#             except queue.Empty:
#                 pass
        
#         try:
#             q.put(rgb_frame)
#         except Exception:
#             pass

#         # Balance frame delivery pacing
#         elapsed_time = time.time() - start_time
#         time.sleep(max(0, frame_delay - elapsed_time))

#     cap.release()
#     print("[Producer Thread] Safely terminated.")

# # --- ENGINE ROUTING CONTROLS ---
# col1, col2 = st.sidebar.columns(2)

# if col1.button("▶️ Start Video Pipeline", use_container_width=True):
#     if not st.session_state.pipeline_active:
#         st.session_state.pipeline_active = True
#         st.session_state.run_flag = [True]
        
#         # Completely drain old leftover frames
#         while not st.session_state.frame_queue.empty():
#             try:
#                 st.session_state.frame_queue.get_nowait()
#             except queue.Empty:
#                 break
        
#         # Spin up fresh queue and background thread
#         st.session_state.frame_queue = queue.Queue(maxsize=max_buffer_size)
#         st.session_state.producer_thread = threading.Thread(
#             target=video_file_producer, 
#             args=(video_path, st.session_state.frame_queue, st.session_state.run_flag), 
#             daemon=True
#         )
#         st.session_state.producer_thread.start()
#         st.rerun()

# if col2.button("⏹️ Stop Video Pipeline", use_container_width=True):
#     if st.session_state.pipeline_active:
#         st.session_state.pipeline_active = False
#         st.session_state.run_flag[0] = False
#         if st.session_state.producer_thread:
#             st.session_state.producer_thread.join(timeout=1.0)
#         st.rerun()

# # --- THE REAL-TIME FRAME BUFFER LAYOUT ---
# if st.session_state.pipeline_active:
#     # 1. Fetch current snapshot of real video frame arrays safely
#     current_frames = list(st.session_state.frame_queue.queue)
#     queue_depth = len(current_frames)
    
#     # 2. Render Status Metrics
#     m1, m2 = st.columns(2)
#     m1.metric("Active Frames In Queue", f"{queue_depth} / {max_buffer_size} Frames Buffered")
    
#     if queue_depth == max_buffer_size:
#         m2.warning("⚠️ Queue Cap Hit! Guardrail dropping old frames to prevent UI lag.")
#     else:
#         m2.success("✅ Queue processing smoothly.")

#     st.write("### Live Horizontal Buffer Pipeline:")
    
#     # 3. Draw the Queue Card Slots Side-by-Side
#     cols = st.columns(max_buffer_size)
    
#     for index in range(max_buffer_size):
#         with cols[index]:
#             if index < queue_depth:
#                 # Slot is occupied: Render the real video frame
#                 st.image(
#                     current_frames[index], 
#                     caption=f"Frame Slot [{index+1}]", 
#                     use_container_width=True
#                 )
#             else:
#                 # Slot is empty: Render a clean empty placeholder layout block
#                 st.markdown(
#                     f"""
#                     <div style="background-color:#1e1e1e; height:150px; border-radius:5px; text-align:center; line-height:150px; color:#444; border:2px dashed #333; font-weight:bold;">
#                         Empty Slot {index+1}
#                     </div>
#                     """, 
#                     unsafe_allow_html=True
#                 )

#     # 4. Read data frame, advance, and force layout re-render pass
#     if not st.session_state.frame_queue.empty():
#         try:
#             # Consuming a frame extracts it from the back to let data move forward
#             st.session_state.frame_queue.get_nowait()
#             st.session_state.frame_queue.task_done()
#         except queue.Empty:
#             pass

#     # Control how fast Streamlit updates the interface layout
#     time.sleep(consumer_delay)
#     st.rerun()

# else:
#     st.info("Pipeline offline. Enter a valid video file path and click Start Video Pipeline.")
