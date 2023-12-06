import os
import cv2
import pytesseract
import subprocess
import re
import time


# Check if tqdm is installed and install it if necessary
try:
    from tqdm import tqdm
except ImportError:
    subprocess.run("pip install tqdm", shell=True)
    from tqdm import tqdm

# Set the path to the Tesseract executable (update this according to your installation)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
#pytesseract.pytesseract.tesseract_cmd = r'.\\Tesseract-OCR\\tesseract.exe'

#Get the location of the video from user input
video_path = input(r'Enter the location of the video: ')
print(f" Starting to process : {video_path}")

# Create a folder for saving highlights
def create_output_folder(video_path):
    video_dir = os.path.dirname(video_path)
    output_folder = os.path.join(video_dir, 'Highlights')

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    return output_folder

def detect_multiple_texts_in_video(video_path, target_texts, frame_interval=10, display_duration=2000, ignore_range=5, display_scale=0.5, crop_bottom_height=450):
    # Record the start time
    #start_time = time.time()
	
    # Create output folder
    output_folder = create_output_folder(video_path)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Dictionary to store timestamps for each target text
    timestamps_dict = {target_text: [] for target_text in target_texts}

    # Iterate through frames with the specified interval
    for frame_number in range(0, total_frames, frame_interval):
        # Set the frame position to the current frame number
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # Read the frame
        ret, frame = cap.read()
        if not ret:
            break

        # Crop the frame to the specified height (e.g., take the bottom 360 pixels)
        height, width, _ = frame.shape
        crop_start = height - crop_bottom_height
        cropped_frame = frame[crop_start:, :]

        # Apply a negative filter to invert colors
        inverted_frame = cv2.bitwise_not(cropped_frame)

        # Convert the inverted frame to grayscale
        gray_frame = cv2.cvtColor(inverted_frame, cv2.COLOR_BGR2GRAY)

        # Perform OCR on the frame
        text = pytesseract.image_to_string(gray_frame)
		
		# Iterate through frames with the specified interval
    #for frame_number in tqdm(range(0, total_frames), desc="Processing frames", unit="frames"):

        # Check if any of the target texts are present
        for target_text in target_texts:
            # Check if the exact target text is present as a whole word
            if re.search(rf'\b{re.escape(target_text.lower())}\b', text.lower()):
            #if target_text.lower() in text.lower():  # Case-insensitive comparison
                # Calculate the timestamp
                timestamp = frame_number / fps

                # Check if the timestamp is within the ignore range of any previous detection
                if not any(abs(prev_timestamp - timestamp) < ignore_range for prev_timestamp in timestamps_dict[target_text]):
                    #print(f"{target_text} at timestamp: {int(timestamp)} seconds")
                    print(f".\\ffmpeg\\bin\\ffmpeg -y -ss {int(timestamp)-4} -i \"{video_path}\" -t 10 -acodec copy -vcodec copy -avoid_negative_ts 1 \"{output_folder}\\{int(timestamp)}-{target_text}.mp4\"")			
                    command = f".\\ffmpeg\\bin\\ffmpeg -y -ss {int(timestamp)-4} -i \"{video_path}\" -t 10 -acodec copy -vcodec copy -avoid_negative_ts 1 \"{output_folder}\\{int(timestamp)}-{target_text}.mp4\""
                    subprocess.run(command, shell=True)
                    # Resize the frame for display
                    resized_frame = cv2.resize(frame, None, fx=display_scale, fy=display_scale)
                    # Resize the cropped frame for display
                    resized_frame_cropped = cv2.resize(cropped_frame, None, fx=display_scale, fy=display_scale)
                    # Display the frame where the text was found
                    cv2.imshow(f'Frame with {target_text}', resized_frame)
                    # Display the cropped frame where the text was found
                    cv2.imshow(f'Frame with {target_text}', resized_frame_cropped)
                    cv2.waitKey(display_duration)  # Wait for display_duration milliseconds
                    cv2.destroyAllWindows()  # Close all OpenCV windows					
                    timestamps_dict[target_text].append(timestamp)  # Store the timestamp in the array

    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()

# Example usage with multiple target texts
#video_path = r'G:\CVDETECTIONTEST\Montage.mp4'
#for testing champion detection
#video_path = r'G:\CVDETECTIONTEST\cham.mp4'
#video_path = r'G:\CVDETECTIONTEST\[11-3-23] KarMukil - 18+ [Tamil_Eng] _ diamond this season_ _ season 19 #apexproplay #Apex #apexlegends #tamilgaming #tfnots.mp4'
target_texts = ['KNOCKED DOWN', 'ELIMINATED', 'ASSIST, ELIMINATION', 'CHAMPION']
timestamps_dict = detect_multiple_texts_in_video(video_path, target_texts, frame_interval=60, display_duration=10000, ignore_range=5, display_scale=0.2)

# Print the dictionary of timestamps for each target text
#for target_text, timestamps in timestamps_dict.items():
    #print(f"{target_text} Timestamps:", timestamps)
