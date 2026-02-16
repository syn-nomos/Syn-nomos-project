import sqlite3
import os
import json

import numpy as np

from datetime import datetime



class DBManager:

    def __init__(self, db_path="data/annotations.db"):
        # Check for Environment Variable Override (For switching DBs easily)
        if os.getenv("LEGAL_HYDRA_DB"):
            db_path = os.getenv("LEGAL_HYDRA_DB")
            
        self.db_path = db_path

        # Timeout 10s is usually enough. 60s might be too long if it's stuck.

        self.conn = sqlite3.connect(db_path, check_same_thread=False, timeout=10)

        self.conn.row_factory = sqlite3.Row

        

        # Enable WAL mode for better concurrency

        try:

            # Check first to avoid unnecessary lock requests

            cursor = self.conn.cursor()

            cursor.execute("PRAGMA journal_mode;")

            current_mode = cursor.fetchone()[0]

            

            if current_mode.lower() != 'wal':

                # Only try to switch if not already in WAL

                self.conn.execute("PRAGMA journal_mode=WAL;")

                self.conn.commit()

        except Exception as e:

            print(f"⚠️ Could not set WAL mode: {e}")

            

        self.create_tables()



    def create_tables(self):

        cursor = self.conn.cursor()

        

        # 1. Πίνακας Προτάσεων (Sentences)

        cursor.execute('''

        CREATE TABLE IF NOT EXISTS sentences (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            text TEXT NOT NULL,

            source_doc TEXT,

            dataset_split TEXT DEFAULT 'train', -- 'train', 'test', 'user_input'

            status TEXT DEFAULT 'pending',

            is_flagged BOOLEAN DEFAULT 0,

            comments TEXT,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

        )

        ''')



        # 2. Πίνακας Annotations (Η Μνήμη μας)

        # Προσθέσαμε: frequency, vector (BLOB), is_golden

        cursor.execute('''

        CREATE TABLE IF NOT EXISTS annotations (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            sentence_id INTEGER,

            text_span TEXT NOT NULL,

            label TEXT NOT NULL,

            start_char INTEGER,

            end_char INTEGER,

            vector BLOB,                -- Augmented Embedding (2313 floats)
            
            -- TRIGGER INFO ADDED
            trigger_text TEXT,
            trigger_vector BLOB,
            trigger_offset INTEGER,
            trigger_confidence REAL,

            confidence REAL DEFAULT 1.0,

            frequency INTEGER DEFAULT 1, -- Πόσες φορές το έχουμε δει

            source_agent TEXT,          -- 'gold_dataset', 'user', 'model'

            resolution_type TEXT,       -- 'LLM_Conflict', 'LLM_Ambiguity', 'Score_Based', 'Direct'

            manual_edit_type TEXT,      -- 'accepted_modified_boundaries', 'accepted_modified_label', 'accepted_modified_both', 'accepted_verified', 'manual_add'

            is_golden BOOLEAN DEFAULT 0, -- Αν προήλθε από το Training Set

            is_accepted BOOLEAN DEFAULT 1,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY(sentence_id) REFERENCES sentences(id)

        )

        ''')



        # 3. Πίνακας Prototypes (Κέντρα Βάρους) - ΓΙΑ ΤΑΧΥΤΗΤΑ

        cursor.execute('''

        CREATE TABLE IF NOT EXISTS prototypes (

            label TEXT PRIMARY KEY,

            vector BLOB,

            count INTEGER, -- Πόσα δείγματα χρησιμοποιήθηκαν

            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP

        )

        ''')



        # 4. Πίνακας Entity Centroids (Κέντρα Βάρους ανά Κείμενο/Ετικέτα)

        cursor.execute('''

        CREATE TABLE IF NOT EXISTS entity_centroids (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            text_span TEXT NOT NULL,

            label TEXT NOT NULL,

            vector BLOB,

            count INTEGER DEFAULT 1,

            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            UNIQUE(text_span, label)

        )

        ''')



        # 5. Πίνακας Memory Centroids (Κέντρα Βάρους ανά Source Annotation ID)

        cursor.execute('''

        CREATE TABLE IF NOT EXISTS memory_centroids (

            source_annotation_id INTEGER PRIMARY KEY,

            text_span TEXT,

            label TEXT,

            vector BLOB,

            count INTEGER DEFAULT 1,

            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP

        )

        ''')



        self.conn.commit()

        self._migrate_db()



    def _migrate_db(self):

        """Checks and applies necessary migrations."""

        cursor = self.conn.cursor()

        

        try:

            # Check if manual_edit_type exists in annotations

            cursor.execute("PRAGMA table_info(annotations)")

            columns_ann = [info[1] for info in cursor.fetchall()]

            if "manual_edit_type" not in columns_ann:

                print("Migrating DB: Adding manual_edit_type column...")

                cursor.execute("ALTER TABLE annotations ADD COLUMN manual_edit_type TEXT")

                self.conn.commit()



            if "is_rejected" not in columns_ann:

                print("Migrating DB: Adding is_rejected column...")

                cursor.execute("ALTER TABLE annotations ADD COLUMN is_rejected BOOLEAN DEFAULT 0")

                self.conn.commit()

                

            # Check if is_flagged and comments exist in sentences

            cursor.execute("PRAGMA table_info(sentences)")

            columns_sent = [info[1] for info in cursor.fetchall()]

            

            if "is_flagged" not in columns_sent:

                print("Migrating DB: Adding is_flagged column to sentences...")

                cursor.execute("ALTER TABLE sentences ADD COLUMN is_flagged BOOLEAN DEFAULT 0")

                self.conn.commit()

                

            if "comments" not in columns_sent:

                print("Migrating DB: Adding comments column to sentences...")

                cursor.execute("ALTER TABLE sentences ADD COLUMN comments TEXT")

                self.conn.commit()

            if "result_json" not in columns_sent:
                print("Migrating DB: Adding result_json column to sentences...")
                cursor.execute("ALTER TABLE sentences ADD COLUMN result_json TEXT")
                self.conn.commit()

            # Create Indices for Performance

            # Use try-except for indices specifically as they might fail if locked even with WAL

            try:

                cursor.execute("CREATE INDEX IF NOT EXISTS idx_ann_sentence_id ON annotations(sentence_id)")

                cursor.execute("CREATE INDEX IF NOT EXISTS idx_ann_is_accepted ON annotations(is_accepted)")

                cursor.execute("CREATE INDEX IF NOT EXISTS idx_ann_is_rejected ON annotations(is_rejected)")

                cursor.execute("CREATE INDEX IF NOT EXISTS idx_ann_label ON annotations(label)")

                cursor.execute("CREATE INDEX IF NOT EXISTS idx_ann_confidence ON annotations(confidence)")

                cursor.execute("CREATE INDEX IF NOT EXISTS idx_ann_sid_accepted ON annotations(sentence_id, is_accepted)")

                self.conn.commit()

            except sqlite3.OperationalError:

                pass # Indices are optional for functionality, skip if locked

                

        except sqlite3.OperationalError as e:

            print(f"⚠️ Migration Warning: {e}. Skipping migration steps for now.")

            # Don't raise, allow app to start even if migration was blocked. 
            # It will retry next time.

    def add_sentence(self, text, source_doc, dataset_split="pending", tokens=None, meta=None, status="pending", split=None):
        """Inserts a new sentence and returns its ID."""
        cursor = self.conn.cursor()
        
        # Handle split parameter alias
        if split and dataset_split == "pending":
            dataset_split = split
        
        cursor.execute('''
            INSERT INTO sentences (text, source_doc, dataset_split, status)
            VALUES (?, ?, ?, ?)
        ''', (text, source_doc, dataset_split, status))
        
        self.conn.commit()
        return cursor.lastrowid

    def update_sentence_status(self, sentence_id, status):
        cursor = self.conn.cursor()
        cursor.execute("UPDATE sentences SET status = ? WHERE id = ?", (status, sentence_id))
        self.conn.commit()

    def update_sentence_result(self, sentence_id, result_json, status=None):
        """Updates the result_json and optional status for a sentence."""
        cursor = self.conn.cursor()
        if status:
            cursor.execute("UPDATE sentences SET result_json = ?, status = ? WHERE id = ?", (result_json, status, sentence_id))
        else:
            cursor.execute("UPDATE sentences SET result_json = ? WHERE id = ?", (result_json, sentence_id))
        self.conn.commit()

    def fetch_pending_sentences(self, split, limit=10):
        """Fetches sentences that are pending review."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM sentences WHERE dataset_split = ? AND status = 'pending' ORDER BY id LIMIT ?",
            (split, limit)
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_sentence(self, sentence_id):
        """Fetches a single sentence by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM sentences WHERE id = ?", (sentence_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def insert_candidate(self, sentence_id, text, label, start, end, vector, confidence, source):
        """Inserts a candidate annotation (prediction)."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO annotations 
            (sentence_id, text_content, label, start_offset, end_offset, vector, confidence, source, is_accepted)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
        """, (sentence_id, text, label, start, end, vector, confidence, source))
        self.conn.commit()

    def add_annotation(self, sentence_id, label, start_offset, end_offset, text_content, source='manual', confidence=1.0, is_accepted=0):
        """Standard method for adding annotations used by Annotator.py"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO annotations (
                sentence_id, label, start_offset, end_offset, text_content, 
                source, confidence, is_accepted
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (sentence_id, label, start_offset, end_offset, text_content, source, confidence, is_accepted))
        self.conn.commit()

    def insert_gold_sentence(self, text, source="train_v2"):

        cursor = self.conn.cursor()

        cursor.execute("INSERT INTO sentences (text, source_doc, dataset_split) VALUES (?, ?, ?)", 

                       (text, source, 'train'))

        self.conn.commit()

        return cursor.lastrowid



    def insert_or_update_annotation(self, sentence_id, text, label, start, end, vector, is_gold=True):

        """

        Έξυπνη εισαγωγή: Αν υπάρχει ήδη το ζεύγος (Text + Label), αυξάνουμε το Frequency.

        Αν είναι διαφορετικό Label (π.χ. GPE vs LOC), φτιάχνει νέα εγγραφή.

        """

        cursor = self.conn.cursor()

        vec_blob = vector.tobytes() if vector is not None else None

        

        # Ψάχνουμε αν υπάρχει ήδη ΑΥΤΟ το κείμενο με ΑΥΤΟ το label

        # (Δεν ψάχνουμε με βάση το start/end γιατί μας ενδιαφέρει η μνήμη γενικά)

        cursor.execute("""

            SELECT id, frequency, vector FROM annotations 

            WHERE text_span = ? AND label = ? AND is_golden = 1

            LIMIT 1

        """, (text, label))

        

        row = cursor.fetchone()

        

        if row:

            # UPDATE: Αυξάνουμε συχνότητα και ενημερώνουμε το vector (Moving Average)

            new_freq = row['frequency'] + 1

            

            # Αν θέλουμε να ενημερώσουμε το vector (να γίνει ο μέσος όρος)

            # old_vec = np.frombuffer(row['vector'], dtype=np.float32)

            # new_vec_avg = (old_vec * (new_freq-1) + vector) / new_freq

            

            cursor.execute("""

                UPDATE annotations 

                SET frequency = ?, sentence_id = ? 

                WHERE id = ?

            """, (new_freq, sentence_id, row['id']))

            # Σημείωση: Κρατάμε το sentence_id της τελευταίας εμφάνισης για reference

        

        else:

            # INSERT: Νέα εγγραφή

            cursor.execute("""

                INSERT INTO annotations (sentence_id, text_span, label, start_char, end_char, vector, frequency, source_agent, is_golden)

                VALUES (?, ?, ?, ?, ?, ?, 1, 'gold_dataset', ?)

            """, (sentence_id, text, label, start, end, vec_blob, 1 if is_gold else 0))

            

        self.conn.commit()



    def get_filtered_sentences(self, filters, limit=50, offset=0):

        """

        Returns sentences that have at least one annotation matching all annotation-level filters.

        Filters: dict with keys 'source_agent', 'label', 'confidence_min', 'status', etc.

        """

        cursor = self.conn.cursor()

        params = []



        # 1. Build Annotation Filter Clause

        ann_clauses = []

        ann_params = []

        

        if filters.get('source_agent'):

            agents = filters['source_agent']

            if agents:

                sub = []

                for agent in agents:

                    sub.append("source_agent LIKE ?")

                    ann_params.append(f"%{agent}%")

                ann_clauses.append("(" + " OR ".join(sub) + ")")



        if filters.get('label'):

            labels = filters['label']

            if labels:

                placeholders = ','.join(['?'] * len(labels))

                ann_clauses.append(f"label IN ({placeholders})")

                ann_params.extend(labels)

            

        if filters.get('confidence_min') is not None:

            min_c = filters['confidence_min']

            if min_c > 0.01:

                ann_clauses.append("confidence >= ?")

                ann_params.append(min_c)

            

        if filters.get('confidence_max') is not None:

            max_c = filters['confidence_max']

            if max_c < 0.99:

                ann_clauses.append("confidence <= ?")

                ann_params.append(max_c)



        # 2. Build Main Query

        query = "SELECT id, text, status, is_flagged, comments FROM sentences s WHERE 1=1"

        

        if filters.get('dataset_split'):

            query += " AND dataset_split = ?"

            params.append(filters['dataset_split'])



        if filters.get('status') == 'pending':

            query += " AND EXISTS (SELECT 1 FROM annotations a_check WHERE a_check.sentence_id = s.id AND a_check.is_accepted = 0 AND (a_check.is_rejected = 0 OR a_check.is_rejected IS NULL))"

        elif filters.get('status') == 'completed':

            query += " AND NOT EXISTS (SELECT 1 FROM annotations a_check WHERE a_check.sentence_id = s.id AND a_check.is_accepted = 0 AND (a_check.is_rejected = 0 OR a_check.is_rejected IS NULL))"

            

        if filters.get('flagged_only'):

            query += " AND is_flagged = 1"



        if ann_clauses:

            query += f" AND EXISTS (SELECT 1 FROM annotations a WHERE a.sentence_id = s.id AND {' AND '.join(ann_clauses)})"

            params.extend(ann_params)



        # Annotation Count Filter

        if filters.get('annotation_count_type'):

            count_type = filters['annotation_count_type']

            min_a = filters.get('min_annotations', 0)

            max_a = filters.get('max_annotations', 9999)

            

            count_cond = ""

            if count_type == "Pending":

                count_cond = "AND is_accepted = 0 AND (is_rejected = 0 OR is_rejected IS NULL)"

            elif count_type == "Accepted":

                count_cond = "AND is_accepted = 1"

            elif count_type == "Rejected":

                count_cond = "AND is_rejected = 1"

            else: # Total (Active)

                count_cond = "AND (is_rejected = 0 OR is_rejected IS NULL)"

                

            query += f" AND (SELECT COUNT(*) FROM annotations a_cnt WHERE a_cnt.sentence_id = s.id {count_cond}) BETWEEN ? AND ?"

            params.extend([min_a, max_a])



        query += " ORDER BY id LIMIT ? OFFSET ?"

        params.extend([limit, offset])

        

        cursor.execute(query, params)

        return [dict(row) for row in cursor.fetchall()]



    def get_total_filtered_count(self, filters):

        cursor = self.conn.cursor()

        params = []

        

        # 1. Build Annotation Filter Clause

        ann_clauses = []

        ann_params = []

        

        if filters.get('source_agent'):

            agents = filters['source_agent']

            if agents:

                sub = []

                for agent in agents:

                    sub.append("source_agent LIKE ?")

                    ann_params.append(f"%{agent}%")

                ann_clauses.append("(" + " OR ".join(sub) + ")")



        if filters.get('label'):

            labels = filters['label']

            if labels:

                placeholders = ','.join(['?'] * len(labels))

                ann_clauses.append(f"label IN ({placeholders})")

                ann_params.extend(labels)

            

        if filters.get('confidence_min') is not None:

            min_c = filters['confidence_min']

            if min_c > 0.01:

                ann_clauses.append("confidence >= ?")

                ann_params.append(min_c)

            

        if filters.get('confidence_max') is not None:

            max_c = filters['confidence_max']

            if max_c < 0.99:

                ann_clauses.append("confidence <= ?")

                ann_params.append(max_c)



        # 2. Build Main Query

        query = "SELECT COUNT(*) as count FROM sentences s WHERE 1=1"

        

        if filters.get('dataset_split'):

            query += " AND dataset_split = ?"

            params.append(filters['dataset_split'])



        if filters.get('status') == 'pending':

            query += " AND EXISTS (SELECT 1 FROM annotations a_check WHERE a_check.sentence_id = s.id AND a_check.is_accepted = 0 AND (a_check.is_rejected = 0 OR a_check.is_rejected IS NULL))"

        elif filters.get('status') == 'completed':

            query += " AND NOT EXISTS (SELECT 1 FROM annotations a_check WHERE a_check.sentence_id = s.id AND a_check.is_accepted = 0 AND (a_check.is_rejected = 0 OR a_check.is_rejected IS NULL))"

            

        if filters.get('flagged_only'):

            query += " AND is_flagged = 1"



        if ann_clauses:

            query += f" AND EXISTS (SELECT 1 FROM annotations a WHERE a.sentence_id = s.id AND {' AND '.join(ann_clauses)})"

            params.extend(ann_params)

            

        # Annotation Count Filter

        if filters.get('annotation_count_type'):

            count_type = filters['annotation_count_type']

            min_a = filters.get('min_annotations', 0)

            max_a = filters.get('max_annotations', 9999)

            

            count_cond = ""

            if count_type == "Pending":

                count_cond = "AND is_accepted = 0 AND (is_rejected = 0 OR is_rejected IS NULL)"

            elif count_type == "Accepted":

                count_cond = "AND is_accepted = 1"

            elif count_type == "Rejected":

                count_cond = "AND is_rejected = 1"

            else: # Total (Active)

                count_cond = "AND (is_rejected = 0 OR is_rejected IS NULL)"

                

            query += f" AND (SELECT COUNT(*) FROM annotations a_cnt WHERE a_cnt.sentence_id = s.id {count_cond}) BETWEEN ? AND ?"

            params.extend([min_a, max_a])

            

        cursor.execute(query, params)

        result = cursor.fetchone()

        return result['count'] if result else 0



        query = """

            SELECT COUNT(*)

            FROM sentences s

            WHERE 1=1

        """

        if filters.get('status'):

            query += " AND s.status = ?"

            params.append(filters['status'])

        if filters.get('flagged_only'):

            query += " AND s.is_flagged = 1"



        query += f" AND EXISTS (SELECT 1 FROM annotations a WHERE a.sentence_id = s.id AND {ann_filter})"



        cursor.execute(query, params)

        return cursor.fetchone()[0]



    def get_annotations_for_sentence(self, sentence_id):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM annotations 
            WHERE sentence_id = ? 
            ORDER BY start_char
        """, (sentence_id,))
        
        results = []
        for row in cursor.fetchall():
            res = dict(row)
            # Map legacy/schema differences
            if 'start_char' in res and 'start_offset' not in res:
                res['start_offset'] = res['start_char']
            if 'end_char' in res and 'end_offset' not in res:
                res['end_offset'] = res['end_char']
            if 'text_span' in res:
                if 'text_content' not in res:
                    res['text_content'] = res['text_span']
                if 'text' not in res:
                    res['text'] = res['text_span']
            elif 'text_content' in res and 'text' not in res:
                res['text'] = res['text_content']
                
            results.append(res)
            
        return results



    def get_unique_values(self, column):

        cursor = self.conn.cursor()

        cursor.execute(f"SELECT DISTINCT {column} FROM annotations WHERE {column} IS NOT NULL")

        return [row[0] for row in cursor.fetchall()]



    def get_confidence_range(self):

        cursor = self.conn.cursor()

        cursor.execute("SELECT MIN(confidence), MAX(confidence) FROM annotations")

        row = cursor.fetchone()

        if row and row[0] is not None and row[1] is not None:

            return row[0], row[1]

        return 0.0, 1.0



    def save_prototype(self, label, vector, count):

        cursor = self.conn.cursor()

        vec_blob = vector.tobytes()

        cursor.execute("""

            INSERT OR REPLACE INTO prototypes (label, vector, count, last_updated)

            VALUES (?, ?, ?, CURRENT_TIMESTAMP)

        """, (label, vec_blob, count))

        self.conn.commit()



    def find_similar_sentences(self, text_span, limit=20):

        """

        Finds pending sentences containing the text_span (Smart Fuzzy Match).

        1. Tries exact substring match.

        2. If few results, tries 'Word Intersection' (all significant words must appear).

        """

        cursor = self.conn.cursor()

        

        # 1. Exact Substring (Normalized)

        search_term = f"%{text_span.lower()}%"

        query = """

            SELECT id, text 

            FROM sentences 

            WHERE status = 'pending' 

            AND lower(text) LIKE ?

            LIMIT ?

        """

        cursor.execute(query, (search_term, limit))

        results = [dict(row) for row in cursor.fetchall()]

        

        # 2. Word Intersection (if results < limit)

        if len(results) < limit:

            # Split into words, keep only significant ones (>2 chars)

            words = [w for w in text_span.lower().split() if len(w) > 2]

            

            if len(words) > 1: # Only if it's a multi-word phrase

                # Construct query: text LIKE %w1% AND text LIKE %w2% ...

                conditions = []

                params = []

                for w in words:

                    conditions.append("lower(text) LIKE ?")

                    params.append(f"%{w}%")

                

                # Exclude already found IDs

                found_ids = [r['id'] for r in results]

                if found_ids:

                    placeholders = ','.join(['?'] * len(found_ids))

                    exclude_clause = f"AND id NOT IN ({placeholders})"

                    params.extend(found_ids)

                else:

                    exclude_clause = ""

                

                params.append(limit - len(results))

                

                query_fuzzy = f"""

                    SELECT id, text 

                    FROM sentences 

                    WHERE status = 'pending' 

                    AND {' AND '.join(conditions)}

                    {exclude_clause}

                    LIMIT ?

                """

                cursor.execute(query_fuzzy, params)

                results.extend([dict(row) for row in cursor.fetchall()])

                

        return results



    def get_similar_pending_annotations(self, target_vector_blob, threshold=0.9, limit=None, label_filter=None, filter_mode="All"):

        """

        Finds pending annotations with high cosine similarity to the target vector.

        Supports label filtering (Same/Diff/All).

        """

        if target_vector_blob is None:

            return []

            

        target_vec = np.frombuffer(target_vector_blob, dtype=np.float32)

        norm_target = np.linalg.norm(target_vec)

        if norm_target == 0: return []



        cursor = self.conn.cursor()

        

        # Base Query

        query = """

            SELECT a.id, a.text_span, a.label, a.confidence, a.vector, a.sentence_id, a.start_char, a.end_char, s.text as sentence_text

            FROM annotations a

            JOIN sentences s ON a.sentence_id = s.id

            WHERE a.is_accepted = 0 

            AND (a.is_rejected = 0 OR a.is_rejected IS NULL)

            AND a.vector IS NOT NULL

        """

        params = []

        

        # Apply Filters

        if filter_mode == "Same" and label_filter:

            query += " AND a.label = ?"

            params.append(label_filter)

        elif filter_mode == "Diff" and label_filter:

            query += " AND a.label != ?"

            params.append(label_filter)

            

        cursor.execute(query, tuple(params))

        

        results = []

        rows = cursor.fetchall()

        

        for row in rows:

            vec_blob = row['vector']

            if not vec_blob: continue

            

            cand_vec = np.frombuffer(vec_blob, dtype=np.float32)

            norm_cand = np.linalg.norm(cand_vec)

            

            if norm_cand == 0: continue

            

            # Cosine Similarity

            sim = np.dot(target_vec, cand_vec) / (norm_target * norm_cand)

            

            if sim >= threshold:

                res = dict(row)

                res['similarity'] = float(sim)

                del res['vector'] # Don't need to send blob back

                results.append(res)

        

        # Sort by similarity desc

        results.sort(key=lambda x: x['similarity'], reverse=True)

        return results[:limit]



    def reject_overlapping_annotations(self, sentence_id, start, end, exclude_id=None):

        """

        Marks overlapping annotations in the same sentence as rejected.

        Overlap logic: (StartA < EndB) and (EndA > StartB)

        """

        cursor = self.conn.cursor()

        

        # NOTE: Updated to strictly reject overlaps, including potential duplicates or accepted items.

        # This will be used when we want to overwrite an area with a new annotation.

        query = """

            UPDATE annotations 

            SET is_rejected = 1, is_accepted = 0

            WHERE sentence_id = ? 

            -- We remove the check for (is_rejected=0) so that we can force-reject everything in this area

            -- except if we specifically exclude an ID (though usually exclude_id handles self).

            AND start_char < ? 

            AND end_char > ?

        """

        params = [sentence_id, end, start]

        

        if exclude_id is not None:

            query += " AND id != ?"

            params.append(exclude_id)

            

        cursor.execute(query, tuple(params))

        self.conn.commit()



    def execute_query(self, query, params=()):

        cursor = self.conn.cursor()

        cursor.execute(query, params)

        self.conn.commit()

        return cursor



    def update_memory_centroid(self, source_annotation_id, text_span, label, new_vector_blob):

        """

        Updates the centroid for a specific Source Annotation ID.

        Formula: NewCentroid = (OldCentroid * Count + NewVector) / (Count + 1)

        """

        if new_vector_blob is None: return



        new_vec = np.frombuffer(new_vector_blob, dtype=np.float32)

        cursor = self.conn.cursor()

        

        # Check if exists

        cursor.execute("SELECT vector, count FROM memory_centroids WHERE source_annotation_id = ?", (source_annotation_id,))

        row = cursor.fetchone()

        

        if row:

            # Update existing

            old_vec_blob = row['vector']

            count = row['count']

            

            if old_vec_blob:

                old_vec = np.frombuffer(old_vec_blob, dtype=np.float32)

                # Calculate weighted average

                updated_vec = (old_vec * count + new_vec) / (count + 1)

            else:

                updated_vec = new_vec

                

            updated_blob = updated_vec.tobytes()

            cursor.execute("""

                UPDATE memory_centroids 

                SET vector = ?, count = count + 1, last_updated = CURRENT_TIMESTAMP 

                WHERE source_annotation_id = ?

            """, (updated_blob, source_annotation_id))

        else:

            # Insert new

            cursor.execute("""

                INSERT INTO memory_centroids (source_annotation_id, text_span, label, vector, count)

                VALUES (?, ?, ?, ?, 1)

            """, (source_annotation_id, text_span, label, new_vector_blob))

            

        self.conn.commit()



    def update_entity_centroid(self, text_span, label, new_vector_blob):

        """

        Updates the centroid for a specific (text_span, label) pair.

        Formula: NewCentroid = (OldCentroid * Count + NewVector) / (Count + 1)

        """

        if new_vector_blob is None: return



        new_vec = np.frombuffer(new_vector_blob, dtype=np.float32)

        cursor = self.conn.cursor()

        

        # Check if exists

        cursor.execute("SELECT vector, count FROM entity_centroids WHERE text_span = ? AND label = ?", (text_span, label))

        row = cursor.fetchone()

        

        if row:

            # Update existing

            old_vec_blob = row['vector']

            count = row['count']

            

            if old_vec_blob:

                old_vec = np.frombuffer(old_vec_blob, dtype=np.float32)

                # Calculate weighted average

                updated_vec = (old_vec * count + new_vec) / (count + 1)

            else:

                updated_vec = new_vec

                

            updated_blob = updated_vec.tobytes()

            cursor.execute("""

                UPDATE entity_centroids 

                SET vector = ?, count = count + 1, last_updated = CURRENT_TIMESTAMP 

                WHERE text_span = ? AND label = ?

            """, (updated_blob, text_span, label))

        else:

            # Insert new

            cursor.execute("""

                INSERT INTO entity_centroids (text_span, label, vector, count)

                VALUES (?, ?, ?, 1)

            """, (text_span, label, new_vector_blob))

            

        self.conn.commit()



    def check_overlapping_annotations(self, sentence_id, start, end, exclude_id=None):

        """

        Checks for overlapping annotations that are NOT rejected.

        Returns list of dicts with keys: id, text_span, label, start_char, end_char

        """

        cursor = self.conn.cursor()

        query = """

            SELECT id, text_span, label, start_char, end_char FROM annotations 

            WHERE sentence_id = ? 

            AND (is_rejected = 0 OR is_rejected IS NULL)

            AND start_char < ? 

            AND end_char > ?

        """

        params = [sentence_id, end, start]

        

        if exclude_id is not None:

            query += " AND id != ?"

            params.append(exclude_id)

            

        cursor.execute(query, tuple(params))

        return [dict(row) for row in cursor.fetchall()]

    def get_top_entities(self, limit=50):
        """Returns the most frequent accepted annotations."""
        cursor = self.conn.cursor()
        
        # Check if text_span or text_content exists
        try:
             cursor.execute("SELECT lower(text_content) as text, label, COUNT(*) as count FROM annotations WHERE is_accepted = 1 GROUP BY lower(text_content), label HAVING count > 1 ORDER BY count DESC LIMIT ?", (limit,))
        except:
             # Fallback to text_span
             cursor.execute("SELECT lower(text_span) as text, label, COUNT(*) as count FROM annotations WHERE is_accepted = 1 GROUP BY lower(text_span), label HAVING count > 1 ORDER BY count DESC LIMIT ?", (limit,))

        return [dict(r) for r in cursor.fetchall()]

    def count_annotations(self, text):
        """Returns the total number of annotations for a specific text."""
        cursor = self.conn.cursor()
        # Try both column names
        try:
            cursor.execute("SELECT COUNT(*) FROM annotations WHERE text_content = ?", (text,))
        except:
            cursor.execute("SELECT COUNT(*) FROM annotations WHERE text_span = ?", (text,))
            
        res = cursor.fetchone()
        return res[0] if res else 0

    def get_high_confidence_candidates(self, min_conf=0.90, limit=50):
        """
        Fetches pending candidates (suggestions) that have high confidence.
        Useful for Batch Accept (False Negatives).
        """
        cursor = self.conn.cursor()
        
        # We look for annotations that are NOT accepted yet (is_accepted=0) but have high confidence
        query = """
            SELECT a.id, a.text_span as text, a.label, a.confidence, s.text as context, a.start_char, a.end_char
            FROM annotations a
            JOIN sentences s ON a.sentence_id = s.id
            WHERE a.is_accepted = 0 
            AND (a.is_rejected = 0 OR a.is_rejected IS NULL)
            AND a.confidence >= ?
            ORDER BY a.confidence DESC
            LIMIT ?
        """
        cursor.execute(query, (min_conf, limit))
        return [dict(r) for r in cursor.fetchall()]

    def get_uncertain_candidates(self, min_conf=0.4, max_conf=0.6, limit=50):
        """
        Fetches pending candidates with uncertain confidence.
        """
        cursor = self.conn.cursor()
        query = """
            SELECT a.id, a.text_span as text, a.label, a.confidence, s.text as context, a.start_char, a.end_char
            FROM annotations a
            JOIN sentences s ON a.sentence_id = s.id
            WHERE a.is_accepted = 0 
            AND (a.is_rejected = 0 OR a.is_rejected IS NULL)
            AND a.confidence >= ? AND a.confidence <= ?
            ORDER BY ABS(a.confidence - 0.5) ASC
            LIMIT ?
        """
        cursor.execute(query, (min_conf, max_conf, limit))
        return [dict(r) for r in cursor.fetchall()]

    def execute_raw(self, query, params=None):
        cursor = self.conn.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        if query.strip().upper().startswith("SELECT"):
            return [dict(r) for r in cursor.fetchall()]
        self.conn.commit()
    
    def search_unannotated_matches(self, text_query):
        """Finds sentences containing exact matches of text_query that are NOT annotated."""
        cursor = self.conn.cursor()
        param_like = f"%{text_query}%"
        
        # Check column names first to be safe
        cursor.execute("PRAGMA table_info(sentences)")
        columns = [r[1] for r in cursor.fetchall()]
        
        target_col = 'text' if 'text' in columns else 'text_content'
        
        if target_col not in columns:
             # Should not happen given init, but handle gracefully
             print("Error: Could not find text column in sentences table.")
             return []

        query = f"SELECT id, {target_col} as text FROM sentences WHERE {target_col} LIKE ?"
        cursor.execute(query, (param_like,))
            
        rows = [dict(r) for r in cursor.fetchall()]
        
        results = []
        import re
        try:
            pattern = re.compile(re.escape(text_query), re.IGNORECASE)
        except:
            return []

        for row in rows:
            anns = self.get_annotations_for_sentence(row['id'])
            # List of EXISTING ranges (start, end)
            occupied_ranges = [(a['start_offset'], a['end_offset']) for a in anns]
            
            for m in pattern.finditer(row['text']):
                s, e = m.start(), m.end()
                
                # Check overlap
                overlap = False
                for (os, oe) in occupied_ranges:
                    if (s < oe) and (e > os):
                        overlap = True
                        break
                
                if not overlap:
                    results.append({
                        'sentence_id': row['id'],
                        'full_text': row['text'],
                        'match_text': row['text'][s:e],
                        'start': s,
                        'end': e
                    })
        return results

    def delete_annotations_by_text(self, text_query, label=None):
        """
        [Consistency] Batch Deletion of all annotations matching a specific text.
        Useful for cleaning up systematic errors.
        """
        cursor = self.conn.cursor()
        
        query = "DELETE FROM annotations WHERE text_span = ?"
        params = [text_query]
        
        if label:
            query += " AND label = ?"
            params.append(label)
            
        cursor.execute(query, tuple(params))
        rows_deleted = cursor.rowcount
        self.conn.commit()
        return rows_deleted

    def rename_annotations(self, text_query, old_label, new_label):
        """
        [Consistency] Batch Rename of annotations.
        """
        cursor = self.conn.cursor()
        query = "UPDATE annotations SET label = ? WHERE text_span = ? AND label = ?"
        cursor.execute(query, (new_label, text_query, old_label))
        rows_updated = cursor.rowcount
        self.conn.commit()
        return rows_updated

    def accept_annotation(self, annotation_id):
        """
        Marks an annotation as accepted (Golden).
        """
        cursor = self.conn.cursor()
        cursor.execute("UPDATE annotations SET is_accepted = 1, is_rejected = 0, is_golden = 1 WHERE id = ?", (annotation_id,))
        self.conn.commit()

    def reject_annotation(self, annotation_id):
        """
        Marks an annotation as rejected.
        """
        cursor = self.conn.cursor()
        cursor.execute("UPDATE annotations SET is_accepted = 0, is_rejected = 1 WHERE id = ?", (annotation_id,))
        self.conn.commit()

    def get_stats(self):
        """Returns general statistics about the database."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM sentences")
        total_sentences = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM annotations")
        total_annotations = cursor.fetchone()[0]
        
        return {
            'total_sentences': total_sentences,
            'total_annotations': total_annotations
        }

    def get_status_counts(self, dataset_split=None):
        """Returns a dictionary of status counts based on dynamic annotation state."""
        cursor = self.conn.cursor()
        
        params = []
        split_clause = ""
        if dataset_split:
            split_clause = " AND s.dataset_split = ?"
            params.append(dataset_split)
        
        # Count Pending
        query_pending = f"""
            SELECT COUNT(DISTINCT s.id) as count
            FROM sentences s
            JOIN annotations a ON s.id = a.sentence_id
            WHERE a.is_accepted = 0 
            AND (a.is_rejected = 0 OR a.is_rejected IS NULL)
            {split_clause}
        """
        cursor.execute(query_pending, tuple(params))
        r_p = cursor.fetchone()
        pending_count = r_p['count'] if r_p else 0
        
        # Count Total in split
        query_total = f"SELECT COUNT(*) as count FROM sentences s WHERE 1=1 {split_clause}"
        cursor.execute(query_total, tuple(params))
        r_t = cursor.fetchone()
        total_count = r_t['count'] if r_t else 0
        
        completed_count = total_count - pending_count
        
        return {
            'pending': pending_count,
            'completed': completed_count
        }

    def update_annotation_vector(self, annotation_id, vector_blob):
        """Updates the vector for a specific annotation."""
        cursor = self.conn.cursor()
        cursor.execute("UPDATE annotations SET vector = ? WHERE id = ?", (vector_blob, annotation_id))
        self.conn.commit()

    def clear_all_vectors(self):
        """Clears all vectors from annotations to force a rebuild."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("UPDATE annotations SET vector = NULL")
            self.conn.commit()
            print("✅ All vectors cleared.")
            return True
        except Exception as e:
            print(f"Error clearing vectors: {e}")
            return False

    def get_annotations_without_vectors(self):
        """Fetches annotations that do not have a vector (for warm-up)."""
        cursor = self.conn.cursor()
        # Join with sentences to get the full text context which is needed for embedding
        cursor.execute("""
            SELECT a.id, a.sentence_id, a.text_span, a.label, a.start_char, a.end_char, s.text as full_text
            FROM annotations a
            JOIN sentences s ON a.sentence_id = s.id
            WHERE a.vector IS NULL
        """)
        return [dict(r) for r in cursor.fetchall()]
