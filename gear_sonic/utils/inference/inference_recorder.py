"""Optional JSONL recording, with disk writes outside the inference/control loops."""

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import queue
import threading
import uuid

import numpy as np


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


class InferenceRecorder:
    """Append one record per request; warn explicitly if recording cannot keep up."""

    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")
        self.session_id = uuid.uuid4().hex
        self._queue = queue.Queue(maxsize=128)
        self._closed = False
        self._failed = False
        self.dropped_records = 0
        self._thread = threading.Thread(target=self._write_loop, daemon=True)
        self._thread.start()
        print(f"Inference JSONL log: {self.path} (session {self.session_id})")

    def record(self, **record):
        if self._closed or self._failed:
            return
        try:
            # The caller may filter/mutate the response after this call.
            snapshot = deepcopy(record)
            snapshot.update(schema_version=1, session_id=self.session_id)
            self._queue.put_nowait(snapshot)
        except queue.Full:
            self.dropped_records += 1
            if self.dropped_records == 1:
                print("[Warning] Inference log queue full; records are being dropped.")
        except Exception as exc:
            print(f"[Warning] Cannot record inference: {exc}")

    def _write_loop(self):
        try:
            while True:
                record = self._queue.get()
                if record is None:
                    break
                self._file.write(json.dumps(record, default=_json_default, ensure_ascii=False) + "\n")
                self._file.flush()
        except Exception as exc:
            self._failed = True
            print(f"[Warning] Inference recording stopped: {exc}")
        finally:
            self._file.close()

    def close(self):
        if self._closed:
            return
        self._closed = True
        # The queue may be full, or its consumer may have failed.
        while self._thread.is_alive():
            try:
                self._queue.put(None, timeout=0.1)
                break
            except queue.Full:
                continue
        self._thread.join()
        if self.dropped_records:
            print(f"[Warning] Inference log dropped {self.dropped_records} records.")


def utc_timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
