from mu_utilities.utilities import SQLServerUtilities
from pathlib import Path
import re, sys, collections

ORPHAN_QUERY = """
SELECT SegmentID FROM Segments
WHERE SegmentID NOT IN (
    SELECT SegmentID FROM SegmentDetection
    UNION
    SELECT SegmentID FROM SegmentCluster
);
"""
SEGMENTS_QUERY = """SELECT SegmentID FROM Segments;"""
SEG_DETECT_QUERY = """SELECT distinct SegmentID FROM [ChipBot].[dbo].[SegmentDetection]"""
SEG_START_TIMES = """SELECT distinct s.[SegmentID]
            ,FORMAT(StartTime,'HHmm')
      --,s.[Species]
  FROM [ChipBot].[dbo].[Segments] s 
  inner join SegmentDetection sd on s.SegmentID = sd.SegmentID
  inner join SegmentChunks sc on sc.SegmentID = s.SegmentID
  inner join Chunks c on c.ChunkID = sc.ChunkID
  inner join Files f on f.FileFullName = c.[FileName]
  inner join [Batches] b on b.BatchID = f.BatchID
  left outer join Detections d on d.ChunkID = c.ChunkID"""
DRY_RUN = False

class CleanSegmentsBase:
    def __init__(self, logger):
        self.logger = logger

class CleanSegments(CleanSegmentsBase):
    def __init__(self, logger, clips_path, sqlserver_connection, links_detections, links_clusters):
        self.clips_path = clips_path
        self.detections_path = links_detections
        self.clusters_path = links_clusters
        self.sqlserver_connection = sqlserver_connection
        CleanSegmentsBase.__init__(self, logger=logger)

    def get_orphaned_segment_ids(self) -> list[int]:
        utilities = SQLServerUtilities(sql=ORPHAN_QUERY, sql_server_connection=self.sqlserver_connection,
                                       logger=self.logger)
        ids = utilities.run_plain_sql_return()
        return ids

    def find_file_for_segment(self, segment_id: int) -> Path | None:
        candidate = Path(self.clips_path) / f"segment_{segment_id}.wav"
        return candidate if candidate.exists() else None

    def phase1_delete_files(self) -> list[int]:
        utilities = SQLServerUtilities(sql=ORPHAN_QUERY, sql_server_connection=self.sqlserver_connection,
                                       logger=self.logger)
        orphaned_ids = utilities.run_plain_sql_return()
        print(f"Found {len(orphaned_ids)} orphaned segment rows.")

        matched_files = []
        missing = []
        for seg_id in orphaned_ids:
            f = self.find_file_for_segment(seg_id[0])
            if f is None:
                missing.append(seg_id[0])
            else:
                matched_files.append(f)

        print(f"{len(matched_files)} files matched, {len(missing)} orphaned segments had no file on disk.")
        if missing:
            print(f"  (no file for SegmentIDs: {missing})")

        if DRY_RUN:
            print("DRY_RUN is True — nothing deleted. Files that WOULD be deleted:")
            for f in matched_files:
                print(f"  {f}")
            return orphaned_ids

        for f in matched_files:
            f.unlink()
            print(f"Deleted: {f}")

        return orphaned_ids

    def phase2_delete_db_rows(self, orphaned_ids: list[int]):
        if not orphaned_ids:
            print("No orphaned IDs to delete.")
            return
        placeholders = ",".join("?" for _ in orphaned_ids)
        sql = f"DELETE FROM Segments WHERE SegmentID IN ({placeholders})"

        utilities = SQLServerUtilities(sql=sql, sql_server_connection=self.sqlserver_connection,
                                       logger=self.logger)
        utilities.run_plain_sql()
        print(f"Deleted {len(orphaned_ids)} rows from Segments.")

    def analyze_segment_discrepancies(self):
        """Compares files in a directory against a list of valid IDs from a query.

        Returns:
            orphan_files: Files on disk with no matching query ID. missing_ids: Query
            IDs that have no file on disk.
        """
        dir_path = Path(self.clips_path)


        utilities = SQLServerUtilities(sql=SEGMENTS_QUERY, sql_server_connection=self.sqlserver_connection,
                                       logger=self.logger)
        query_results = utilities.run_plain_sql_return()

        # Convert query IDs to strings for reliable comparison
        valid_ids = {str(row[0]) for row in query_results}

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

        print(orphan_files)
        print(missing_id_set)

        # Assuming 'orphans' is a collection (like a list, set, or generator) of PosixPath objects
        for file_path in orphan_files:
            try:
                if file_path.is_file():
                    file_path.unlink()
                    self.logger.info(f"Deleted orphaned file: {file_path}")
            except Exception as e:
                self.logger.error(f"Failed to delete {file_path}: {e}")

    def analyze_detections_discrepancies(self):
        """Compares files in a directory against a list of valid IDs from a query.

        Returns:
            orphan_files: Files on disk with no matching query ID. missing_ids: Query
            IDs that have no file on disk.
        """
        dir_path = Path(self.detections_path)

        utilities = SQLServerUtilities(sql=SEG_START_TIMES, sql_server_connection=self.sqlserver_connection,
                                       logger=self.logger)
        seg_starts = utilities.run_plain_sql_return()


        utilities = SQLServerUtilities(sql=SEG_DETECT_QUERY, sql_server_connection=self.sqlserver_connection,
                                       logger=self.logger)
        query_results = utilities.run_plain_sql_return()

        # Convert query IDs to strings for reliable comparison
        valid_ids = {str(row[0]) for row in query_results}

        disk_files = {}  # maps seg_id -> Path object
        orphan_files = []
        deleted_orphans = []
        unmatched_files = []
        total_wavs = 0
        # Regex to match segment_<id>.wav (case-insensitive)
        pattern = re.compile(r'_(\d+)(?:_[^_]+){3}\.[^.]+$')
        total_ids = []
        for file_path in dir_path.rglob('*.wav'):
            if file_path.is_file():
                total_wavs += 1
                filename = file_path.name
                match = pattern.search(filename)

                if match:
                    segment_id = match.group(1)
                    total_ids.append(segment_id)




                    # Check if it's an orphan
                    if segment_id not in valid_ids:
                        try:
                            #file_path.unlink()
                            deleted_orphans += 1
                            self.logger.info(f"Deleted orphaned file: {file_path}")
                        except Exception as e:
                            self.logger.error(f"Failed to delete {file_path}: {e}")
                else:
                    # Track files that failed regex matching
                    unmatched_files.append(file_path)
                    self.logger.warning(f"Regex mismatch on file: {filename}")

        c=0

        # 1. Ensure total_ids and valid_id are properly defined sets/lists
        # (Example: total_ids = [match.group(1) for ...])
        set_total_ids = set(total_ids)

        # 2. Re-create missing_from_db as a set to guarantee it's iterable
        missing_from_db = set_total_ids - valid_ids

        deleted_count = 0
        total_wavs = 0

        # 3. Run the loop safely
        for file_path in dir_path.rglob('*.wav'):
            if file_path.is_file():
                total_wavs += 1
                filename = file_path.name
                match = pattern.search(filename)

                if match:
                    segment_id = match.group(1)
                    # Safe membership check because missing_from_db is a set
                    if str(segment_id) in missing_from_db:
                        try:
                            file_path.unlink()
                            deleted_count += 1
                            self.logger.info(f"Deleted orphaned file: {file_path}")
                        except Exception as e:
                            self.logger.error(f"Failed to delete {file_path}: {e}")

        # 4. Log final summary once at the end
        self.logger.info(
            f"Cleanup complete. Total WAVs scanned: {total_wavs} | Deleted: {deleted_count}"
        )

        sys.exit()
        for file_path in unmatched_files:
            filename = file_path.name
            stem = Path(filename).stem
            parts = stem.split('_')
            segment_id = parts[-3]
            for seg in seg_starts:
                if str(seg[0]) == str(segment_id):
                    parts = filename.rsplit('_', 2)
                    new_name = parts[0] + '_' + seg[1] + '_' + parts[1] + '_' + parts[2]
                    new_file_path = file_path.with_name(new_name)
                    file_path.rename(new_file_path)
                    c += 1





        sys.exit()
        # 1. Scan directory and extract IDs
        for file_path in dir_path.rglob('*.wav'):
            if file_path.is_file():
                filename = file_path.name
                match = pattern.search(filename)
                if match:
                    seg_id = match.group(1)
                    disk_files[seg_id] = file_path
                disk_id_set = set(disk_files.keys())

        # 2. Find orphans (on disk, but NOT in the query list)
        orphan_id_set = disk_id_set - valid_ids
        orphan_files = [disk_files[seg_id] for seg_id in orphan_id_set]

        # 3. Optional: Find missing files (in the query list, but NOT on disk)
        missing_id_set = valid_ids - disk_id_set

        print(orphan_files)
        print(missing_id_set)

        # Assuming 'orphans' is a collection (like a list, set, or generator) of PosixPath objects
        for file_path in orphan_files:
            try:
                if file_path.is_file():
                    file_path.unlink()
                    self.logger.info(f"Deleted orphaned file: {file_path}")
            except Exception as e:
                self.logger.error(f"Failed to delete {file_path}: {e}")
