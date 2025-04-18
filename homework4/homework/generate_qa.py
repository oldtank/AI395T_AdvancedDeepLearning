import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# Define object type mapping
OBJECT_TYPES = {
    1: "Kart",
    2: "Track Boundary",
    3: "Track Element",
    4: "Special Element 1",
    5: "Special Element 2",
    6: "Special Element 3",
}

# Define colors for different object types (RGB format)
COLORS = {
    1: (0, 255, 0),  # Green for karts
    2: (255, 0, 0),  # Blue for track boundaries
    3: (0, 0, 255),  # Red for track elements
    4: (255, 255, 0),  # Cyan for special elements
    5: (255, 0, 255),  # Magenta for special elements
    6: (0, 255, 255),  # Yellow for special elements
}

# Original image dimensions for the bounding box coordinates
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def extract_frame_info(image_path: str) -> tuple[int, int]:
    """
    Extract frame ID and view index from image filename.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (frame_id, view_index)
    """
    filename = Path(image_path).name
    # Format is typically: XXXXX_YY_im.png where XXXXX is frame_id and YY is view_index
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)  # Convert hex to decimal
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0  # Default values if parsing fails


def draw_detections(
    image_path: str, info_path: str, font_scale: float = 0.5, thickness: int = 1, min_box_size: int = 5
) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on the image.

    Args:
        image_path: Path to the image file
        info_path: Path to the corresponding info.json file
        font_scale: Scale of the font for labels
        thickness: Thickness of the bounding box lines
        min_box_size: Minimum size for bounding boxes to be drawn

    Returns:
        The annotated image as a numpy array
    """
    # Read the image using PIL
    pil_image = Image.open(image_path)
    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Get image dimensions
    img_width, img_height = pil_image.size

    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)

    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    # Extract frame ID and view index from image filename
    _, view_index = extract_frame_info(image_path)

    # Get the correct detection frame based on view index
    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections")
        return np.array(pil_image)

    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    # Draw each detection
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        # Get color for this object type
        if track_id == 0:
            color = (255, 0, 0)
        else:
            color = COLORS.get(class_id, (255, 255, 255))

        # Draw bounding box using PIL
        draw.rectangle([(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)], outline=color, width=thickness)
        draw.point(((x1_scaled+x2_scaled)/2, (y1_scaled+y2_scaled)/2), fill="yellow")

    draw.point((img_width/2, img_height/2), fill="yellow")
    # Convert PIL image to numpy array for matplotlib
    return np.array(pil_image)


def extract_kart_objects(
    info_path: str, view_index: int, img_width: int = 150, img_height: int = 100, min_box_size: int = 3
) -> list:
    """
    Extract kart objects from the info.json file, including their center points and identify the center kart.
    Filters out karts that are out of sight (outside the image boundaries).

    Args:
        info_path: Path to the corresponding info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 100)
        img_height: Height of the image (default: 150)

    Returns:
        List of kart objects, each containing:
        - instance_id: The track ID of the kart
        - kart_name: The name of the kart
        - center: (x, y) coordinates of the kart's center
        - is_center_kart: Boolean indicating if this is the kart closest to image center
    """

    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    kart_names = info["karts"]

    karts = []
    center_index = -1
    center_dist = 9999999
    image_center_x = int(img_width/2)
    image_center_y = int(img_height/2)

    detections = info["detections"][view_index]
    curr_index = 0
    for detection in detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)
        if class_id != 1:
            continue

        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        kart_name = kart_names[track_id]

        center_x = int((x1_scaled+x2_scaled)/2)
        center_y = int((y1_scaled+y2_scaled)/2)
        curr_center_dist = ((center_x - image_center_x)**2 + (center_y - image_center_y)**2)**0.5
        # print(f"distance {curr_center_dist}")
        if curr_center_dist < center_dist:
            center_dist = curr_center_dist
            center_index = curr_index

        karts.append({"track_id": track_id, "kart_name": kart_name, "center_x": center_x, "center_y": center_y, "is_center": False})
        curr_index += 1

    if len(karts) > 0:
        # print(f"center index: {center_index}")
        karts[center_index]["is_center"] = True
    return karts

def extract_track_info(info_path: str) -> str:
    with open(info_path) as f:
        info = json.load(f)

    return info["track"]

def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate question-answer pairs for a given view.

    Args:
        info_path: Path to the info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 100)
        img_height: Height of the image (default: 150)

    Returns:
        List of dictionaries, each containing a question and answer
    """
    # 1. Ego car question
    # What kart is the ego car?

    # 2. Total karts question
    # How many karts are there in the scenario?

    # 3. Track information questions
    # What track is this?

    # 4. Relative position questions for each kart
    # Is {kart_name} to the left or right of the ego car?
    # Is {kart_name} in front of or behind the ego car?

    # 5. Counting questions
    # How many karts are to the left of the ego car?
    # How many karts are to the right of the ego car?
    # How many karts are in front of the ego car?
    # How many karts are behind the ego car?

    qa_pairs = []

    infofile_path = Path(info_path)
    base_name = infofile_path.stem.replace("_info", "")
    image_file = list(infofile_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]
    image_file = str(image_file)
    index = image_file.index("data/")
    image_file = image_file[index + len("data/"):]

    karts = extract_kart_objects(info_path, view_index, img_width, img_height, min_box_size=3)

    if len(karts) ==0:
        return []
    ego_kart = [kart for kart in karts if kart["is_center"]][0]

    ego_left = 0
    ego_right = 0
    ego_front = 0
    ego_back = 0
    non_ego_position_x = {}
    non_ego_position_y = {}

    for kart in karts:
        if kart["kart_name"] == ego_kart["kart_name"]:
            continue
        if kart["center_x"] < ego_kart["center_x"]:
            ego_left += 1
            non_ego_position_x[kart["kart_name"]] = "left"
        else:
            ego_right += 1
            non_ego_position_x[kart["kart_name"]] = "right"
        if kart["center_y"] < ego_kart["center_y"]:
            ego_front += 1
            non_ego_position_y[kart["kart_name"]] = "front"
        else:
            ego_back += 1
            non_ego_position_y[kart["kart_name"]] = "back"

    for kart in karts:
        kart_name = kart["kart_name"]
        if kart_name == ego_kart["kart_name"]:
            continue
        relative = ""
        if non_ego_position_y[kart_name] == "front":
            qa_pairs.append({
                "question": f"Is {kart_name} in front of or behind the ego car?",
                "answer": "front",
                "image_file": image_file
            })
            relative += "front and "
        else:
            qa_pairs.append({
                "question": f"Is {kart_name} in front of or behind the ego car?",
                "answer": "back",
                "image_file": image_file

            })
            relative += "back and "
        if non_ego_position_x[kart_name] == "left":
            qa_pairs.append({
                "question": f"Is {kart_name} to the left or right of the ego car?",
                "answer": "left",
                "image_file": image_file
            })
            relative += "left"
        else:
            qa_pairs.append({
                "question": f"Is {kart_name} to the left or right of the ego car?",
                "answer": "right",
                "image_file": image_file
            })
            relative += "right"
        qa_pairs.append({
            "question": f"Where is {kart_name} relative to the ego car?",
            "answer": relative,
            "image_file": image_file
        })

    qa_pairs.append({
        "question": "What kart is the ego car?",
        "answer": ego_kart["kart_name"],
        "image_file": image_file
    })

    qa_pairs.append({
        "question": "How many karts are there in the scenario?",
        "answer": str(len(karts)),
        "image_file": image_file
    })

    qa_pairs.append({
        "question": "How many karts are to the left of the ego car?",
        "answer": str(ego_left),
        "image_file": image_file
    })

    qa_pairs.append({
        "question": "How many karts are to the right of the ego car?",
        "answer": str(ego_right),
        "image_file": image_file
    })

    qa_pairs.append({
        "question": "How many karts are in front of the ego car?",
        "answer": str(ego_front),
        "image_file": image_file
    })

    qa_pairs.append({
        "question": "How many karts are behind the ego car?",
        "answer": str(ego_back),
        "image_file": image_file
    })

    track_name = extract_track_info(info_path)
    qa_pairs.append(
        {"question": "What track is this?",
         "answer": track_name,
         "image_file": image_file})

    return qa_pairs


def check_qa_pairs(info_file: str, view_index: int):
    """
    Check QA pairs for a specific info file and view index.

    Args:
        info_file: Path to the info.json file
        view_index: Index of the view to analyze
    """
    # Find corresponding image file
    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    # Visualize detections
    annotated_image = draw_detections(str(image_file), info_file, min_box_size=3)

    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    # plt.show()
    plt_path = Path(f"plots/{base_name}_{view_index:02d}_im.jpg")
    plt.savefig(plt_path)

    # Generate QA pairs
    qa_pairs = generate_qa_pairs(info_file, view_index)

    # Print QA pairs
    print("\nQuestion-Answer Pairs:")
    print("-" * 50)
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print(f"F: {qa['image_file']}")
        print("-" * 50)

def generate_all(split="train", max_batch=15):
    data_dir = Path(__file__).parent.parent / "data"
    info_files = list(data_dir.glob(f"{split}/*_info.json"))

    to_write = []
    batch_num = 0
    output_file = f"data/train/{batch_num:03d}_qa_pairs.json"
    curr_info_file_index = 0
    total_count = 0
    for info_file in info_files:
        if curr_info_file_index == 100:
            try:
                with open(output_file, 'w', encoding='utf-8') as json_file:
                    json.dump(to_write, json_file)
                total_count += len(to_write)
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
            curr_info_file_index = 0
            batch_num += 1
            output_file = f"data/train/{batch_num:03d}_qa_pairs.json"
            to_write = []

            if batch_num == max_batch:
                break

        # print(f"current info file: {info_file}")
        base_name = info_file.stem.replace("_info", "")
        for view_index in range(10):
            image_file = list(info_file.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]
            if not image_file.is_file():
                break
            # print(f"generating questions for {info_file} view_index {view_index}")
            to_write.extend(generate_qa_pairs(str(info_file), view_index))

        curr_info_file_index += 1

    print(f"total number of data: {total_count}")

    # try:
    #     with open(output_file, 'w', encoding='utf-8') as json_file:
    #         json.dump(to_write, json_file)
    # except Exception as e:
    #     print(f"An unexpected error occurred: {e}")


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_qa.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_qa_pairs, "generate_all": generate_all})


if __name__ == "__main__":
    main()
