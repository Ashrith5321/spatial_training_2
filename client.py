import requests
import time
import io
import numpy as np
from PIL import Image

# Configuration
SERVER_URL = "http://localhost:8000/act"
def get_action(image_bytes, instruction=None):
    """
    Stateless client function.
    Args:
        image_bytes (bytes): The JPEG/PNG encoded bytes of the image.
        instruction (str, optional): Text instruction. If present, resets server state.
    Returns:
        dict: The JSON response containing 'action_id' and 'action_probs'.
    """
    # Prepare multipart/form-data payload
    files = {'file': ('obs.jpg', image_bytes, 'image/jpeg')}
    data = {}
    
    if instruction:
        data['instr_or_goal'] = instruction

    try:
        # Send POST request
        response = requests.post(SERVER_URL, files=files, data=data)
        response.raise_for_status() # Raise error for 4xx/5xx status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

def generate_dummy_image():
    """Helper to create a random noise image in memory."""
    # Create random 256x256 RGB image
    arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    
    # Encode to JPEG bytes (simulating a camera capture)
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    return buf.getvalue()

if __name__ == "__main__":
    print(f"--- Starting VLA Client Demo (Target: {SERVER_URL}) ---")

    # --- Step 1: Initialize Episode (Image + Instruction) ---
    print("\n[Step 0] Sending RESET + Instruction...")
    
    img_bytes = generate_dummy_image()
    instruction = "Find the green apple on the table"
    
    start_time = time.time()
    result = get_action(img_bytes, instruction=instruction)
    latency = (time.time() - start_time) * 1000 # ms
    
    print(f"   -> Latency: {latency:.2f} ms")
    print(f"   -> Response: {result}")

    # --- Step 2: Continuous Control (Images only, State is preserved on server) ---
    print("\n[Step 1-5] Entering control loop (Images only)...")
    
    for i in range(1, 6):
        # Simulate robot moving and getting new camera frame
        img_bytes = generate_dummy_image()
        
        start_time = time.time()
        # Note: No instruction passed here, so server maintains context
        result = get_action(img_bytes) 
        latency = (time.time() - start_time) * 1000
        
        print(f"   [Step {i}] Latency: {latency:.2f} ms | Action ID: {result.get('action_id')}")
        
        # Optional: Sleep to simulate control frequency (e.g., 5Hz)
        time.sleep(0.2)

    print("\n--- Demo Complete ---")
# ssh -p 34929 -L 8000:localhost:8000 root@136.59.129.136 -N[]