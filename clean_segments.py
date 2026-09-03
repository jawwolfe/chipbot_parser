from mu_utilities.utilities import SQLServerUtilities
from pathlib import Path

ORPHAN_QUERY = """
SELECT SegmentID FROM Segments
WHERE SegmentID NOT IN (
    SELECT SegmentID FROM SegmentDetection
    UNION
    SELECT SegmentID FROM SegmentCluster
);
"""
DRY_RUN = False

class CleanSegmentsBase:
    def __init__(self, logger):
        self.logger = logger

class CleanSegments(CleanSegmentsBase):
    def __init__(self, logger, clips_path, sqlserver_connection):
        self.clips_path = clips_path
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