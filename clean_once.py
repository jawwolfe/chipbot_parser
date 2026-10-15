import re
from pathlib import Path
from mu_utilities.utilities import SQLServerUtilities

def analyze_segment_discrepancies(directory_path, query_ids):
  """Compares files in a directory against a list of valid IDs from a query.

  Returns:
      orphan_files: Files on disk with no matching query ID. missing_ids: Query
      IDs that have no file on disk.
  """
  dir_path = Path(directory_path)

  # Convert query IDs to strings for reliable comparison
  valid_ids = {str(i) for i in query_ids}

  disk_files = {}  # maps seg_id -> Path object
  orphan_files = []

  # Regex to match segment_<id>.wav (case-insensitive)
  pattern = re.compile(r"^segment_(\d+)\.wav$", re.IGNORECASE)

  # 1. Scan directory and extract IDs
  for file_path in dir_path.glob("segment_*.wav"):
    match = pattern.match(file_path.name)
    if match:
      seg_id = match.group(1)
      disk_files[seg_id] = file_path

  disk_id_set = set(disk_files.keys())

  # 2. Find orphans (on disk, but NOT in the query list)
  orphan_id_set = disk_id_set - valid_ids
  orphan_files = [disk_files[seg_id] for seg_id in orphan_id_set]

  # 3. Optional: Find missing files (in the query list, but NOT on disk)
  missing_id_set = valid_ids - disk_id_set

  return orphan_files, missing_id_set


# --- Example Usage ---
if __name__ == "__main__":
  ORPHAN_QUERY = """SELECT SegmentID FROM Segments FROM Segments;"""
  utilities = SQLServerUtilities(sql=ORPHAN_QUERY, sql_server_connection=self.sqlserver_connection,
                                 logger=self.logger)
  query_results = utilities.run_plain_sql_return()

  directory = "/mnt/c/temp/CHIPBOT_DATA_ROOT/segment_clips"

  orphans, missing = analyze_segment_discrepancies(directory, query_results)

  print(f"Found {len(orphans)} orphan files on disk:")
  for f in orphans:
    print(f"  - {f.name}")

  print(f"\nFound {len(missing)} query IDs missing files on disk:")
  for mid in missing:
    print(f"  - segment_{mid}.wav")