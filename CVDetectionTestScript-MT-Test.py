import os
import cv2
import pytesseract
import subprocess
import re
import threading
from queue import Queue

# Set the path to the Tesseract executable (update this according to your installation)
pytesseract.pytesseract.tesseract_cmd = r'.\\Tesseract-OCR\\tesseract.exe'

def create_output_folder(video_path):
    video_dir = os.path.dirname(video_path)
    output_folder = os.path.join(video_dir, 'Highlights')

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    return output_folder

def read_frames(cap, frame_queue, total_frames, frame_interval):
    for frame_number in range(0, total_frames, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if ret:
            frame_queue.put((frame_number, frame))

    # Signal the end of frames
    frame_queue.put(None)

def process_frames(frame_queue, fps, target_texts, ignore_range, timestamps_dict, output_folder, display_duration, display_scale, ffmpeg_lock):
    while True:
        item = frame_queue.get()
        if item is None:
            break

        frame_number, frame = item

        # Apply a negative filter to invert colors
        inverted_frame = cv2.bitwise_not(frame)

        # Convert the inverted frame to grayscale
        gray_frame = cv2.cvtColor(inverted_frame, cv2.COLOR_BGR2GRAY)

        # Perform OCR on the frame
        text = pytesseract.image_to_string(gray_frame)

        # Check if any of the target texts are present
        for target_text in target_texts:
            # Check if the exact target text is present as a whole word
            if re.search(rf'\b{re.escape(target_text.lower())}\b', text.lower()):
                # Calculate the timestamp
                timestamp = frame_number / fps

                # Check if the timestamp is within the ignore range of any previous detection
                if not any(abs(prev_timestamp - timestamp) < ignore_range for prev_timestamp in timestamps_dict[target_text]):
                    # Acquire lock before executing FFmpeg operation
                    with ffmpeg_lock:
                        print(f".\\ffmpeg\\bin\\ffmpeg -y -ss {int(timestamp)-4} -i \"{video_path}\" -t 10 -acodec copy -vcodec copy -avoid_negative_ts 1 \"{output_folder}\\{int(timestamp)}-{target_text}.mp4\"")
                        command = f".\\ffmpeg\\bin\\ffmpeg -y -ss {int(timestamp)-4} -i \"{video_path}\" -t 10 -acodec copy -vcodec copy -avoid_negative_ts 1 \"{output_folder}\\{int(timestamp)}-{target_text}.mp4\""
                        subprocess.run(command, shell=True)

                    # Resize the frame for display
                    resized_frame = cv2.resize(frame, None, fx=display_scale, fy=display_scale)

                    # Display the frame where the text was found
                    cv2.imshow(f'Frame with {target_text}', resized_frame)
                    cv2.waitKey(display_duration)  # Wait for display_duration milliseconds
                    cv2.destroyAllWindows()  # Close all OpenCV windows
                    timestamps_dict[target_text].append(timestamp)  # Store the timestamp in the array

# Example usage with multiple target texts
video_path = r'G:\CVDETECTIONTEST\Montage.mp4'
target_texts = ['KNOCKED DOWN', 'ELIMINATED', 'ASSIST, ELIMINATION', 'CHAMPION']

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Create output folder
output_folder = create_output_folder(video_path)

# Dictionary to store timestamps for each target text
timestamps_dict = {target_text: [] for target_text in target_texts}

# Create a lock to synchronize FFmpeg operations
ffmpeg_lock = threading.Lock()

# Create a queue to communicate between threads
frame_queue = Queue()

# Create and start threads for reading and processing frames
read_thread = threading.Thread(target=read_frames, args=(cap, frame_queue, total_frames, 60))
process_thread = threading.Thread(target=process_frames, args=(frame_queue, fps, target_texts, 5, timestamps_dict, output_folder, 1000, 0.2, ffmpeg_lock))

read_thread.start()
process_thread.start()

# Wait for both threads to complete
read_thread.join()
process_thread.join()

# Release the video capture object
cap.release()
