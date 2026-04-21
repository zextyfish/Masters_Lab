import cv2
import csv
import multiprocessing as mp

video_path = "G_01.mp4"
output_csv = "laser_tracking.csv"

# Function to process a chunk of frames
def process_chunk(start_frame, end_frame):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    results = []

    for frame_idx in range(start_frame, end_frame):
        if frame_idx % 50 != 0:
            continue
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Background subtraction to isolate laser
        blur = cv2.GaussianBlur(gray, (31, 31), 0)
        diff = cv2.subtract(gray, blur)

        # Find brightest point
        _, maxVal, _, maxLoc = cv2.minMaxLoc(diff)

        # Optional: reject weak detections
        if maxVal < 10:
            cx, cy = None, None
        else:
            cx, cy = maxLoc

        results.append([frame_idx, cx, cy])

    cap.release()
    return results

# Main multiprocessing function
if __name__ == "__main__":
    # Get total frames
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    num_cores = mp.cpu_count()
    chunk_size = total_frames // num_cores

    pool = mp.Pool(num_cores)
    jobs = []

    for i in range(num_cores):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i != num_cores - 1 else total_frames
        jobs.append(pool.apply_async(process_chunk, (start, end)))

    # Collect results from all processes
    all_results = []
    for job in jobs:
        all_results.extend(job.get())

    pool.close()
    pool.join()

    # Sort results by frame index (important!)
    all_results.sort(key=lambda x: x[0])

    # Write to CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "x", "y"])
        writer.writerows(all_results)

    print(f"Done! Results saved to {output_csv}")