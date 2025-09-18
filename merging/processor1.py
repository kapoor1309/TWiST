import json
import numpy as np
from pathlib import Path
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_frame_id(frame_id):
    """Clean frame ID by removing tensor() wrapper and converting to string"""
    if isinstance(frame_id, str):
        return frame_id.replace('tensor(', '').replace(')', '')
    return str(frame_id)

def load_triplet_tool_mapping(mapping_data):
    """Parse the triplet-tool mapping data and create lookup dictionaries"""
    tool_to_triplets = {}
    
    for entry in mapping_data:
        triplet_id = entry[0]
        tool_id = entry[1]
        
        if tool_id not in tool_to_triplets:
            tool_to_triplets[tool_id] = []
        tool_to_triplets[tool_id].append(triplet_id)
    
    logger.info(f"Loaded mapping for {len(tool_to_triplets)} unique tool IDs")
    for tool_id, triplets in tool_to_triplets.items():
        logger.info(f"Tool {tool_id} can be part of triplets: {triplets}")
    
    return tool_to_triplets

def get_best_triplets_for_tool(tool_id, recognition_probs, tool_to_triplets, num_instances=1):
    """Get the triplet IDs with highest probabilities for a given tool ID"""
    possible_triplets = tool_to_triplets[tool_id]
    triplet_probs = [(tid, recognition_probs[tid]) for tid in possible_triplets]
    triplet_probs.sort(key=lambda x: x[1], reverse=True)
    return [tp[0] for tp in triplet_probs[:num_instances]]

def process_video(video_id, predictions_path, bboxes_path, tool_to_triplets):
    """Process a single video and generate its prediction JSON"""
    logger.info(f"\nProcessing video {video_id}")
    
    # Load predictions and bbox files
    pred_file = f"{predictions_path}/VID{video_id}.json"
    bbox_file = f"{bboxes_path}/VID{video_id}.json"
    
    try:
        with open(pred_file, 'r') as f:
            predictions = json.load(f)
        logger.info(f"Loaded predictions for {len(predictions)} frames from {pred_file}")
    except Exception as e:
        logger.error(f"Error loading predictions file {pred_file}: {str(e)}")
        return {}
    
    try:
        with open(bbox_file, 'r') as f:
            bboxes_data = json.load(f)
        logger.info(f"Loaded bbox data for {len(bboxes_data)} frames from {bbox_file}")
    except Exception as e:
        logger.error(f"Error loading bbox file {bbox_file}: {str(e)}")
        return {}
    
    video_result = {}
    frame_tools = {clean_frame_id(frame_data['frame_id']): frame_data['tools'] 
                  for frame_data in bboxes_data}
    
    matched_frames = 0
    for frame_id, pred_data in predictions.items():
        if frame_id not in frame_tools:
            logger.warning(f"Frame {frame_id} found in predictions but not in bbox data")
            continue
            
        tools = frame_tools[frame_id]
        recognition_probs = pred_data['recognition']
        
        if not tools or not recognition_probs:
            logger.warning(f"Frame {frame_id} has missing data")
            continue
        
        tool_counts = {}
        for tool in tools:
            tool_id = tool['tool_id']
            tool_counts[tool_id] = tool_counts.get(tool_id, 0) + 1
        
        frame_result = {
            'recognition': recognition_probs,
            'detection': []
        }
        
        processed_tools = {tool_id: 0 for tool_id in tool_counts.keys()}
        valid_detections = 0
        
        for tool in tools:
            tool_id = tool['tool_id']
            if tool_id not in tool_to_triplets:
                logger.warning(f"Frame {frame_id}: Tool ID {tool_id} not found in mapping")
                continue
                
            bbox = tool['bbox']
            
            try:
                instance_num = processed_tools[tool_id]
                best_triplets = get_best_triplets_for_tool(
                    tool_id, recognition_probs, tool_to_triplets, tool_counts[tool_id]
                )
                
                if not best_triplets:
                    logger.warning(f"Frame {frame_id}: No valid triplets found for tool {tool_id}")
                    continue
                    
                triplet_id = best_triplets[instance_num]
                
                detection_entry = {
                    'triplet': triplet_id,
                    'instrument': [
                        tool_id,
                        0,
                        bbox[0],
                        bbox[1],
                        bbox[2],
                        bbox[3]
                    ]
                }
                frame_result['detection'].append(detection_entry)
                valid_detections += 1
                processed_tools[tool_id] += 1
                
            except Exception as e:
                logger.error(f"Error processing tool in frame {frame_id}: {str(e)}")
                continue
        
        if valid_detections > 0:
            video_result[frame_id] = frame_result
            matched_frames += 1
    
    logger.info(f"Successfully processed {matched_frames} frames with valid detections")
    return video_result

def update_video_probabilities(model_data, video_id, studio_path):
    """Update video data with tool probabilities"""
    tool_prob_file = os.path.join(studio_path, 'predictions', f'video_{video_id}.json') 
    
    try:
        with open(tool_prob_file, 'r') as f:
            tool_probs = json.load(f)
        logger.info(f"Loaded tool probabilities from {tool_prob_file}")
    except FileNotFoundError:
        logger.warning(f"Tool probability file not found for video {video_id}")
        return model_data
    
    video_key = f"VID{video_id}"
    if video_key not in model_data:
        logger.warning(f"{video_key} not found in model data")
        return model_data
    
    for frame_id, frame_data in model_data[video_key].items():
        if "detection" not in frame_data:
            continue
        
        frame_data_prob = tool_probs.get(frame_id)
        if not frame_data_prob or "tool_probabilities" not in frame_data_prob:
            continue
        
        frame_probs = frame_data_prob["tool_probabilities"]
        
        for detection in frame_data["detection"]:
            if "instrument" not in detection:
                continue
            
            try:
                instrument = detection["instrument"]
                if not isinstance(instrument, list) or len(instrument) < 2:
                    continue
                
                tool_id = instrument[0]
                if tool_id is not None and isinstance(tool_id, int) and instrument[1] is None:
                    if tool_id < len(frame_probs):
                        instrument[1] = frame_probs[tool_id]
                    else:
                        logger.warning(f"Tool ID {tool_id} is out of range for frame {frame_id}")
            except Exception as e:
                logger.error(f"Error processing detection in frame {frame_id}: {str(e)}")
                continue
    
    return model_data

def process_all_videos(video_ids, studio_path, predictions_path, bboxes_path, mapping_data):
    """Process all videos with both functionalities"""
    tool_to_triplets = load_triplet_tool_mapping(mapping_data)
    
    for video_id in video_ids:
        logger.info(f"\nProcessing video {video_id}")
        
        # First phase: Generate predictions with bounding boxes
        result = process_video(video_id, predictions_path, bboxes_path, tool_to_triplets)
        initial_output = {f"VID{video_id}": result}
        
        # Second phase: Update with tool probabilities
        updated_output = update_video_probabilities(initial_output, video_id, studio_path)
        
        # Save final results
        output_filename = os.path.join(studio_path, f"updated_model_{video_id}.json")
        with open(output_filename, 'w') as f:
            json.dump(updated_output, f, indent=4)
        
        logger.info(f"Saved final results for video {video_id} with {len(result)} frames")

# Example usage and configuration
if __name__ == "__main__":
    video_ids = [92, 96, 103, 110, 111]
    studio_path = '/teamspace/studios/this_studio'
    predictions_path = os.path.join(studio_path, "predictions")
    bboxes_path = os.path.join(studio_path, "output/test_cams_with_bboxes")
    
    # Define mapping data
    mapping_data = [
    [0, 0, 2, 1, 2, 1],
    [1, 0, 2, 0, 2, 0],
    [2, 0, 2, 10, 2, 10],
    
    [3, 0, 0, 3, 0, 3],
    [4, 0, 0, 2, 0, 2],
    [5, 0, 0, 4, 0, 4],
    [6, 0, 0, 1, 0, 1],
    [7, 0, 0, 0, 0, 0],
    [8, 0, 0, 12, 0, 12],
    [9, 0, 0, 8, 0, 8],
    [10, 0, 0, 10, 0, 10],
    [11, 0, 0, 11, 0, 11],
    [12, 0, 0, 13, 0, 13],
    [13, 0, 8, 0, 8, 0],
    [14, 0, 1, 2, 1, 2],
    [15, 0, 1, 4, 1, 4],
    [16, 0, 1, 1, 1, 1],
    [17, 0, 1, 0, 1, 0],
    [18, 0, 1, 12, 1, 12],
    [19, 0, 1, 8, 1, 8],
    [20, 0, 1, 10, 1, 10],
    [21, 0, 1, 11, 1, 11],
    [22, 1, 3, 7, 13, 22],
    [23, 1, 3, 5, 13, 20],
    [24, 1, 3, 3, 13, 18],
    [25, 1, 3, 2, 13, 17],
    [26, 1, 3, 4, 13, 19],
    [27, 1, 3, 1, 13, 16],
    [28, 1, 3, 0, 13, 15],
    [29, 1, 3, 8, 13, 23],
    [30, 1, 3, 10, 13, 25],
    [31, 1, 3, 11, 13, 26],
    [32, 1, 2, 9, 12, 24],
    [33, 1, 2, 3, 12, 18],
    [34, 1, 2, 2, 12, 17],
    [35, 1, 2, 1, 12, 16],
    [36, 1, 2, 0, 12, 15],
    [37, 1, 2, 10, 12, 25],
    [38, 1, 0, 1, 10, 16],
    [39, 1, 0, 8, 10, 23],
    [40, 1, 0, 13, 10, 28],
    [41, 1, 1, 2, 11, 17],
    [42, 1, 1, 4, 11, 19],
    [43, 1, 1, 0, 11, 15],
    [44, 1, 1, 8, 11, 23],
    [45, 1, 1, 10, 11, 25],
    [46, 2, 3, 5, 23, 35],
    [47, 2, 3, 3, 23, 33],
    [48, 2, 3, 2, 23, 32],
    [49, 2, 3, 4, 23, 34],
    [50, 2, 3, 1, 23, 31],
    [51, 2, 3, 0, 23, 30],
    [52, 2, 3, 8, 23, 38],
    [53, 2, 3, 10, 23, 40],
    [54, 2, 5, 5, 25, 35],
    [55, 2, 5, 11, 25, 41],
    [56, 2, 2, 5, 22, 35],
    [57, 2, 2, 3, 22, 33],
    [58, 2, 2, 2, 22, 32],
    [59, 2, 2, 1, 22, 31],
    [60, 2, 2, 0, 22, 30],
    [61, 2, 2, 10, 22, 40],
    [62, 2, 2, 11, 22, 41],
    [63, 2, 1, 0, 21, 30],
    [64, 2, 1, 8, 21, 38],
    [65, 3, 3, 10, 33, 55],
    [66, 3, 5, 9, 35, 54],
    [67, 3, 5, 5, 35, 50],
    [68, 3, 5, 3, 35, 48],
    [69, 3, 5, 2, 35, 47],
    [70, 3, 5, 1, 35, 46],
    [71, 3, 5, 8, 35, 53],
    [72, 3, 5, 10, 35, 55],
    [73, 3, 5, 11, 35, 56],
    [74, 3, 2, 1, 32, 46],
    [75, 3, 2, 0, 32, 45],
    [76, 3, 2, 10, 32, 55],
    [77, 4, 4, 5, 44, 65],
    [78, 4, 4, 3, 44, 63],
    [79, 4, 4, 2, 44, 62],
    [80, 4, 4, 4, 44, 64],
    [81, 4, 4, 1, 44, 61],
    [82, 5, 6, 6, 56, 81],
    [83, 5, 2, 2, 52, 77],
    [84, 5, 2, 4, 52, 79],
    [85, 5, 2, 1, 52, 76],
    [86, 5, 2, 0, 52, 75],
    [87, 5, 2, 10, 52, 85],
    [88, 5, 7, 7, 57, 82],
    [89, 5, 7, 4, 57, 79],
    [90, 5, 7, 8, 57, 83],
    [91, 5, 1, 0, 51, 75],
    [92, 5, 1, 8, 51, 83],
    [93, 5, 1, 10, 51, 85],
    [94, 0, 9, 14, 9, 14],
    [95, 1, 9, 14, 19, 29],
    [96, 2, 9, 14, 29, 44],
    [97, 3, 9, 14, 39, 59],
    [98, 4, 9, 14, 49, 74],
    [99, 5, 9, 14, 59, 89],


]
    
    # Run the combined processing
    process_all_videos(video_ids, studio_path, predictions_path, bboxes_path, mapping_data)