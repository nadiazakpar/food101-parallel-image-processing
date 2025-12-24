from multiprocessing import pool
import cv2
import numpy as np
import os
import time
import multiprocessing
import argparse

# ==========================================
# CONFIGURATION
# ==========================================

# Path to the downloaded Food-101 'images' folder
INPUT_ROOT_DIR = "food-101/" 

# Where to save processed results
OUTPUT_ROOT_DIR = "food-101_processed_mp"

# Manageable subsets: specific folders to process
SELECTED_CATEGORIES = [
    "pizza",
    "sushi",
    "hamburger"
]

# Default max images per category for testing
DEFAULT_LIMIT = 50

# ==========================================
# IMAGE PROCESSING PIPELINE (WORKERS)
# ==========================================
def process_single_image(file_info):
    """
    Worker function to process a single image through the full image processing pipeline.
    Args:
        file_info (tuple): (input_path, output_path)
    """
    input_path, output_path = file_info
    
    try:
        # Read image
        img = cv2.imread(input_path)
        if img is None:
            return f"Failed to read: {os.path.basename(input_path)}"

        # 1. Grayscale Conversion 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. Gaussian Blur (3x3) on grayscale
        blur = cv2.GaussianBlur(gray, (3, 3), 0)

        # 3. Edge Detection (Sobel)
        sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
        edges = cv2.magnitude(sobelx, sobely)
        edges = np.uint8(edges)

        # 4. Image Sharpening (unsharp masking style)
        sharpened = cv2.addWeighted(blur, 1.5, edges, -0.5, 0)

        # 5. Brightness Adjustment
        # Increase brightness by adding a constant (beta=50)
        brightness = cv2.convertScaleAbs(sharpened, alpha=1.0, beta=50)

        # --- SAVING RESULTS ---
        final_output = brightness

        # Ensure output directory exists 
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, final_output)

        return None # Success returns None

    except Exception as e:
        return f"Error in {os.path.basename(input_path)}: {str(e)}"

# ==========================================
# MAIN DRIVER 
# ==========================================
def main():
    # --- 1. ARGUMENT PARSING ---
    parser = argparse.ArgumentParser(description="Food-101 Parallel Image Processor")
    parser.add_argument(
        '--workers', 
        type=int, 
        default=0, 
        help="Number of workers to use. (0 = Use all available cores.)"
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=DEFAULT_LIMIT, 
        help="Limit number of images per category for testing."
    )
    args = parser.parse_args()

    # --- 2. SETUP ---
    print(f"--- Starting Multiprocessing Pipeline ---")
    print(f"Selected Categories: {SELECTED_CATEGORIES}")

    # Determine CPU Count
    max_cores = multiprocessing.cpu_count()
    if args.workers > 0:
        num_cores = args.workers
        print(f"Custom worker count selected: {num_cores}")
    else:
        num_cores = max_cores
        print(f"Auto-detecting cores... Using all {num_cores} cores.")

    # Override the global limit if the user specified one
    image_limit = args.limit
    
    # --- 3. GATHER TASKS ---
    tasks = []
    for category in SELECTED_CATEGORIES:
        source_folder = os.path.join(INPUT_ROOT_DIR, category)
        target_folder = os.path.join(OUTPUT_ROOT_DIR, category)
        
        # Verify source exists
        if not os.path.exists(source_folder):
            print(f"Warning: Source folder not found: {source_folder}")
            continue
            
        # Get list of images
        files = [f for f in os.listdir(source_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Apply limit
        if image_limit:
            files = files[:image_limit]
            
        for f in files:
            input_path = os.path.join(source_folder, f)
            output_path = os.path.join(target_folder, f)
            tasks.append((input_path, output_path))

    print(f"Total images to process: {len(tasks)}")
    
    if not tasks:
        print("No images found. Check your directory structure.")
        return

    print(f"Running processing with {num_cores} workers...")
    start_time = time.time()

    # --- 4. EXECUTE PARALLEL PROCESSING ---
    with multiprocessing.Pool(processes=num_cores) as pool:
        # imap_unordered hands out tasks one-by-one to whoever is free for good load balancing.
        results = list(pool.imap_unordered(process_single_image, tasks, chunksize=1))

    end_time = time.time()
    
    # --- 5. REPORTING ---
    errors = [res for res in results if res is not None]
    
    print("\n" + "="*30)
    print("PROCESSING COMPLETE")
    print("="*30)
    print(f"Execution Time : {end_time - start_time:.4f} seconds")
    print(f"Images Processed: {len(tasks)}")
    print(f"Errors Encountered: {len(errors)}")
    
    if errors:
        print("Sample Error:", errors[0])

if __name__ == "__main__":
    main()