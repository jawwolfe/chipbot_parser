import os
import re

# --- CONFIGURATION ---
DIRECTORY_PATH = "/mnt/c/temp/CHIPBOT_DATA_ROOT/audio_batches/Philippines_Cebu_Pacijan-Campo-Wenceslao_2026-10-10-151308_2026-10-10-202813"  # Change to your target directory if not the current one
NEW_LATITUDE = "10.658367"  # Your new latitude value (as string)
NEW_LONGITUDE = "124.357490"  # Your new longitude value (as string)
# ---------------------

# Regular expression to match the filename pattern:
# Group 1: Everything before the lat/long (e.g., aw-chipbot-02_2026-10-10-151308_)
# Group 2: The old latitude
# Group 3: The old longitude
pattern = re.compile(r"^(.+?)_([+-]?\d+\.\d+)_([+-]?\d+\.\d+)$")


def update_filenames():
  if not os.path.exists(DIRECTORY_PATH):
    print(f"Error: Directory '{DIRECTORY_PATH}' does not exist.")
    return

  files = os.listdir(DIRECTORY_PATH)
  renamed_count = 0

  for filename in files:
    # If there's a file extension (like .txt or .png), split it off to preserve it
    name_without_ext, ext = os.path.splitext(filename)

    match = pattern.match(name_without_ext)
    if match:
      prefix = match.group(1)

      # Construct the new filename
      new_name_core = f"{prefix}_{NEW_LATITUDE}_{NEW_LONGITUDE}"
      new_filename = new_name_core + ext

      old_path = os.path.join(DIRECTORY_PATH, filename)
      new_path = os.path.join(DIRECTORY_PATH, new_filename)

      # Perform the rename
      os.rename(old_path, new_path)
      print(f"Renamed: {filename} -> {new_filename}")
      renamed_count += 1

  print(f"\nDone! Successfully updated {renamed_count} files.")


if __name__ == "__main__":
  update_filenames()