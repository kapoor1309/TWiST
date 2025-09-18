import json as json_lib
import os
import pprint

# ---------- Config ----------
# Set to True to process only one video (easier to debug)
DEBUG_SINGLE_VIDEO = False
DEBUG_VIDEO_ID = '92'   # used when DEBUG_SINGLE_VIDEO=True

# ---------- Helpers ----------
def create_empty_frame_data():
    """Create empty frame data with zero probabilities and default detection"""
    return {
        "recognition": [0.0] * 100,  # 100 zeros for triplet probabilities
        "detection": [
            {
                "triplet": -1,
                "instrument": [-1, 0, -1, -1, -1, -1]
            }
        ]
    }

def safe_load_json(path):
    """Load JSON and return (data, error_str). error_str is None on success."""
    if not os.path.exists(path):
        return None, f"File does not exist: {path}"
    try:
        with open(path, 'r') as f:
            return json_lib.load(f), None
    except Exception as e:
        return None, f"Error loading JSON {path}: {e}"

def normalize_frame_keys(frames_dict):
    """Normalize dictionary keys to strings (so 0 and '0' both become '0')."""
    if not isinstance(frames_dict, dict):
        return {}
    return { str(k): v for k, v in frames_dict.items() }

def is_frames_dict_like(d):
    """Heuristic: check if dict keys look like frame ids (all numeric strings or ints)."""
    if not isinstance(d, dict) or len(d) == 0:
        return False
    sample_keys = list(d.keys())[:20]
    for k in sample_keys:
        if isinstance(k, int):
            continue
        if isinstance(k, str) and k.isdigit():
            continue
        return False
    return True

# ---------- Main merging function (debug-enabled) ----------
def process_and_merge_videos(studio_path, video_configs):
    merged_data = {}

    for video_id, max_frame in video_configs.items():
        if DEBUG_SINGLE_VIDEO and video_id != DEBUG_VIDEO_ID:
            continue

        print("\n" + "="*60)
        print(f"Processing video {video_id}")
        video_key = f"VID{video_id}"

        # Load the updated model file
        model_file = os.path.join(studio_path, f'updated_model_{video_id}.json')
        print("Model file:", model_file)
        file_exists = os.path.exists(model_file)
        print("Exists?", file_exists, "Size (bytes):", os.path.getsize(model_file) if file_exists else "NA")

        video_data, err = safe_load_json(model_file)
        if err:
            print("WARN:", err)
            current_frames = {}
        else:
            # Show top-level structure
            top_level_keys = list(video_data.keys()) if isinstance(video_data, dict) else []
            print("Top-level keys in loaded JSON (sample):", top_level_keys[:10])

            # Try different possible structures:
            # 1) { "VID92": { "0": {...}, "1": {...} } }
            # 2) { "0": {...}, "1": {...} }  (frames dict only)
            if isinstance(video_data, dict) and video_key in video_data:
                print(f"Found top-level video key '{video_key}'.")
                current_frames_raw = video_data[video_key]
            elif is_frames_dict_like(video_data):
                print("File appears to be a frames dict directly (no top-level VID key).")
                current_frames_raw = video_data
            else:
                print(f"WARNING: Couldn't find '{video_key}' in file, and file doesn't look like a frames dict.")
                print("Showing sample of the file (truncated):")
                pprint.pprint({k: video_data[k] for k in list(video_data.keys())[:5]} if isinstance(video_data, dict) else str(video_data)[:500])
                current_frames_raw = {}

            # Normalize frame keys (to strings) and inspect samples
            current_frames = normalize_frame_keys(current_frames_raw)
            print(f"Frames loaded: {len(current_frames)} (after normalizing keys to strings)")
            sample_keys = list(current_frames.keys())[:10]
            print("Sample frame keys (as strings):", sample_keys)
            print("Sample frame key types:", [type(k) for k in sample_keys])
            if len(sample_keys) > 0:
                # preview a single sample value (truncated)
                sample_val = current_frames[sample_keys[0]]
                print("Sample frame value preview (keys):", list(sample_val.keys()) if isinstance(sample_val, dict) else type(sample_val))
        # end load block

        # Build complete_frames with normalized string frame ids (0..max_frame)
        complete_frames = {}
        missing_count = 0
        for frame_num in range(max_frame + 1):
            frame_id = str(frame_num)
            if 'current_frames' in locals() and frame_id in current_frames:
                complete_frames[frame_id] = current_frames[frame_id]
            else:
                complete_frames[frame_id] = create_empty_frame_data()
                missing_count += 1

        total = len(complete_frames)
        print(f"Total frames expected: {total}; Missing frames filled with placeholders: {missing_count}")

        # Count placeholders vs non-placeholders and show examples
        def is_placeholder(frame_val):
            try:
                if not isinstance(frame_val, dict):
                    return True
                det = frame_val.get('detection')
                if not det or not isinstance(det, list):
                    return True
                first = det[0] if len(det) > 0 else {}
                return first.get('triplet') == -1
            except Exception:
                return True

        placeholder_count = sum(1 for v in complete_frames.values() if is_placeholder(v))
        non_placeholder_count = total - placeholder_count
        print(f"After building: {placeholder_count} placeholders, {non_placeholder_count} real frames")

        if non_placeholder_count > 0:
            # print up to 3 non-placeholder examples
            examples_shown = 0
            print("Examples of non-placeholder frames (up to 3):")
            for k, v in complete_frames.items():
                if not is_placeholder(v):
                    print(f" - frame {k}: detection keys -> {list(v.keys())}; detection[0] -> {v.get('detection')[0] if isinstance(v.get('detection'), list) and len(v.get('detection'))>0 else 'N/A'}")
                    examples_shown += 1
                    if examples_shown >= 3:
                        break
        else:
            print("No non-placeholder frames found; this means the loaded file provided no usable frames for this video.")

        # Add to merged data
        merged_data[video_key] = complete_frames

        # If debugging single video, optionally write a temp dump for inspection
        if DEBUG_SINGLE_VIDEO:
            tmp_out = os.path.join(studio_path, f'final_debug_{video_id}.json')
            with open(tmp_out, 'w') as f:
                json_lib.dump({video_key: complete_frames}, f, indent=2)
            print("Wrote debug output to", tmp_out)

    # Save merged data
    output_file = os.path.join(studio_path, 'final_vgg.json')
    try:
        with open(output_file, 'w') as f:
            json_lib.dump(merged_data, f, indent=2)
        print(f"\nSuccessfully created merged file: {output_file}")
        print("Output size (bytes):", os.path.getsize(output_file))
    except Exception as e:
        print(f"Error saving merged file: {str(e)}")


# ---------- Main ----------
def main():
    # Configuration for each video
    video_configs = {
        '92': 2123,
        '96': 1706,
        '103': 2219,
        '110': 2176,
        '111': 2145
    }

    # studio path (change if needed)
    studio_path = '/teamspace/studios/this_studio'

    # Run (single-video debug mode if selected)
    global DEBUG_SINGLE_VIDEO, DEBUG_VIDEO_ID
    # Optionally set DEBUG_SINGLE_VIDEO here if you want from code (overrides top config)
    # DEBUG_SINGLE_VIDEO = True
    # DEBUG_VIDEO_ID = '92'

    if DEBUG_SINGLE_VIDEO:
        print("RUNNING IN DEBUG SINGLE-VIDEO MODE for video:", DEBUG_VIDEO_ID)

    process_and_merge_videos(studio_path, video_configs)

if __name__ == "__main__":
    main()
