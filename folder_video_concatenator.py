import os
import glob
import subprocess
import imageio_ffmpeg

class FolderVideoConcatenator:
    """
    Concatenates all MP4 files from a folder into a single video file using FFmpeg.
    Supports Audio.
    """
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": ""}),
                "output_filename": ("STRING", {"default": "concatenated_output"}),
                "sort_by": (["name_asc", "name_desc", "date_asc", "date_desc"],),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "concatenate_videos"
    OUTPUT_NODE = True
    CATEGORY = "video/processing"

    def concatenate_videos(self, folder_path, output_filename, sort_by):
        if not os.path.exists(folder_path):
            raise ValueError(f"Folder not found: {folder_path}")

        # Find mp4 files
        search_pattern = os.path.join(folder_path, "*.mp4")
        video_files = glob.glob(search_pattern)
        
        # Filter out the output file if it exists
        full_output_path = os.path.join(folder_path, f"{output_filename}.mp4")
        video_files = [f for f in video_files if os.path.normpath(f) != os.path.normpath(full_output_path)]

        if not video_files:
            raise ValueError(f"No .mp4 files found in {folder_path} (excluding output file)")

        # Sorting
        if sort_by == "name_asc":
            video_files.sort()
        elif sort_by == "name_desc":
            video_files.sort(reverse=True)
        elif sort_by == "date_asc":
            video_files.sort(key=os.path.getmtime)
        elif sort_by == "date_desc":
            video_files.sort(key=os.path.getmtime, reverse=True)

        print(f"Found {len(video_files)} videos to concatenate.")

        # Create FFmpeg concat list file
        # Format: file 'path/to/file.mp4'
        list_txt_path = os.path.join(folder_path, "concat_list.txt")
        
        try:
            with open(list_txt_path, "w", encoding="utf-8") as f:
                for video_path in video_files:
                    # Escape single quotes in path
                    safe_path = video_path.replace("'", "'\\''")
                    f.write(f"file '{safe_path}'\n")

            # Get FFmpeg executable
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
            
            # Construct command
            # -f concat: usage of concat demuxer
            # -safe 0: allow absolute paths
            # -i list.txt: input list
            # -c copy: stream copy (no re-encoding, fast, preserves audio)
            # -y: overwrite output
            cmd = [
                ffmpeg_path,
                "-f", "concat",
                "-safe", "0",
                "-i", list_txt_path,
                "-c", "copy",
                "-y",
                full_output_path
            ]
            
            print(f"Executing FFmpeg command: {' '.join(cmd)}")
            
            # Run FFmpeg
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Check for errors (sometimes subprocess raises, sometimes it just prints stderr)
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg failed: {result.stderr}")
                
            print(f"Concatenation complete: {full_output_path}")
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg execution failed: {e.stderr}")
        finally:
            # Cleanup temporary list
            if os.path.exists(list_txt_path):
                os.remove(list_txt_path)

        return (full_output_path,)
