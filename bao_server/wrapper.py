#!/usr/bin/env python3
"""Bao Query Optimizer Skill — single-file long-lived HTTP server wrapper.

Wraps the Bao tree-convolution learned optimizer. Evaluates the 5 Bao "arms"
(GUC combinations), predicts per-arm latency, and emits the optimized query
with the selected arm's GUCs as `pg_hint_plan` `Set(...)` hints.

Design: lock-free optimize via an
atomic model snapshot; batched arm scoring (parallel EXPLAIN + one predict);
background EXPLAIN ANALYZE for training data (separate pool); training in a
separate process (`wrapper.py --train --model-dir ...`) that saves the model
atomically; server picks up new weights via an mtime check.

Endpoints: GET /health, POST /optimize, GET /state, POST /shutdown.
"""

import argparse
import contextlib
import io
import json
import os
import queue as queue_mod
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import psycopg2

# Patch sqlite3 busy_timeout before importing storage (runtime sqlite connect).
_orig_sqlite_connect = sqlite3.connect


def _sqlite_connect_with_timeout(*args, **kwargs):
    kwargs.setdefault("timeout", 30.0)
    return _orig_sqlite_connect(*args, **kwargs)


sqlite3.connect = _sqlite_connect_with_timeout

from model import BaoRegression
from reg_blocker import _ALL_OPTIONS, _arm_idx_to_hints
import storage

DEFAULT_MODEL_DIR = "bao_default_model"
RETRAIN_THRESHOLD_DEFAULT = 200
POOL_SIZE_DEFAULT = 4
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Safe-gate for background training: if the selected arm's plan cost is more
# than this many times the default arm (arm 0) cost, skip EXPLAIN ANALYZE /
# experience recording for that arm. A catastrophically worse arm (e.g. one
# that disables a critical join method) would hang the background thread.
_MAX_COST_RATIO = 50.0


def _plan_total_cost(plan_json):
    """Total Cost from a PostgreSQL EXPLAIN (FORMAT JSON) plan, or None."""
    try:
        return float(plan_json[0]["Plan"]["Total Cost"])
    except Exception:
        return None


def _cost_ratio_too_high(chosen_cost, default_cost, max_ratio=_MAX_COST_RATIO):
    """True if the chosen plan's cost is catastrophically higher than the
    default's. Unknown / non-positive costs → False (let background try)."""
    if not chosen_cost or not default_cost:
        return False
    try:
        d = float(default_cost)
        if d <= 0:
            return False
        return float(chosen_cost) / d > max_ratio
    except (TypeError, ValueError, ZeroDivisionError):
        return False


# --------------------------------------------------------------------------- #
# DB connection pool.
# --------------------------------------------------------------------------- #
class ConnectionPool:
    def __init__(self, size=POOL_SIZE_DEFAULT):
        self.size = size
        self._pools = {}
        self._created = {}
        self._lock = threading.Lock()

    def _queue(self, dsn):
        with self._lock:
            q = self._pools.get(dsn)
            if q is None:
                q = queue_mod.Queue(maxsize=self.size)
                self._pools[dsn] = q
                self._created[dsn] = 0
            return q

    def get(self, dsn):
        q = self._queue(dsn)
        try:
            return q.get_nowait()
        except queue_mod.Empty:
            pass
        with self._lock:
            if self._created[dsn] < self.size:
                self._created[dsn] += 1
                conn = psycopg2.connect(dsn)
                conn.set_client_encoding("UTF8")
                return conn
        return q.get(timeout=60)

    def put(self, dsn, conn):
        self._queue(dsn).put(conn)

    def discard(self, dsn, conn):
        self._queue(dsn)
        with self._lock:
            try:
                conn.close()
            except Exception:
                pass
            if self._created.get(dsn, 0) > 0:
                self._created[dsn] -= 1

    def close_all(self):
        with self._lock:
            for q in self._pools.values():
                while True:
                    try:
                        conn = q.get_nowait()
                    except queue_mod.Empty:
                        break
                    try:
                        conn.close()
                    except Exception:
                        pass
            self._pools.clear()
            self._created.clear()


# --------------------------------------------------------------------------- #
# Arm -> pg_hint_plan Set(...) hint conversion + EXPLAIN helpers.
# --------------------------------------------------------------------------- #
def _arm_to_set_hints(arm_idx):
    final_state = {}
    for stmt in _arm_idx_to_hints(arm_idx):
        parts = stmt.split()
        if len(parts) >= 4 and parts[0].upper() == "SET" and parts[2].upper() == "TO":
            final_state[parts[1]] = parts[3]
    return [f"Set({opt} {val})" for opt, val in final_state.items()]


def _render_hint_comment(set_hints):
    if not set_hints:
        return ""
    return "/*+ " + " ".join(set_hints) + " */ "


def _get_buffer_state(conn):
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT c.relname, count(*) "
                "FROM pg_buffercache b "
                "JOIN pg_class c ON c.relfilenode = b.relfilenode "
                "WHERE c.relname NOT LIKE 'pg\\_%' "
                "  AND c.relname NOT LIKE 'sql\\_%' "
                "GROUP BY c.relname"
            )
            return {row[0]: row[1] for row in cur.fetchall()}
    except psycopg2.Error as e:
        print(f"[bao-wrapper] pg_buffercache query failed ({e}); using empty state.",
              file=sys.stderr)
        return {}


def _strip_per_node_buffers(plan_node):
    BUF_KEYS = ["Shared Hit Blocks", "Shared Read Blocks",
                "Shared Dirtied Blocks", "Shared Written Blocks",
                "Local Hit Blocks", "Local Read Blocks",
                "Local Dirtied Blocks", "Local Written Blocks",
                "Temp Read Blocks", "Temp Written Blocks"]

    def recurse(n):
        for k in BUF_KEYS:
            n.pop(k, None)
        for child in n.get("Plans", []) or []:
            recurse(child)

    recurse(plan_node)


def _get_explain_plan(conn, sql, set_hints, with_analyze=False):
    hint_comment = _render_hint_comment(set_hints)
    stmt = ("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) "
            if with_analyze else "EXPLAIN (FORMAT JSON) ")
    full_stmt = stmt + hint_comment + sql
    with conn.cursor() as cur:
        cur.execute("SET statement_timeout TO 300000")
        cur.execute(full_stmt)
        row = cur.fetchall()
        plan = row[0][0]
        if isinstance(plan, str):
            plan = json.loads(plan)
        return plan


def _read_n_trained(model_dir):
    n_path = os.path.join(model_dir, "n")
    if not os.path.exists(n_path):
        return 0
    try:
        with open(n_path, "rb") as f:
            return int(joblib.load(f))
    except Exception:
        return 0


def _model_path(model_dir):
    return os.path.join(model_dir, "nn_weights")


# --------------------------------------------------------------------------- #
# Training subprocess: load experiences + model from disk, fit, save atomically.
# --------------------------------------------------------------------------- #
def run_training(model_dir):
    """Training subprocess: fit a fresh BaoRegression on all collected
    experiences and save atomically. Works whether or not a prior model
    exists (Bao retrains from all experiences each time)."""
    os.makedirs(model_dir, exist_ok=True)
    experiences = storage.experience()
    if not experiences:
        return 0
    x = [row[0] for row in experiences]
    y = np.array([row[1] for row in experiences], dtype=np.float64)
    reg = BaoRegression(have_cache_data=True, verbose=False)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            reg.fit(x, y)
    except Exception as e:
        print(f"training fit failed: {e}", file=sys.stderr)
        return 1
    tmp_dir = f"{model_dir}.tmp.{os.getpid()}"
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        reg.save(tmp_dir)
        for fn in os.listdir(tmp_dir):
            os.replace(os.path.join(tmp_dir, fn), os.path.join(model_dir, fn))
    except Exception as e:
        print(f"training save failed: {e}", file=sys.stderr)
        return 1
    finally:
        try:
            for fn in os.listdir(tmp_dir):
                p = os.path.join(tmp_dir, fn)
                if os.path.exists(p):
                    os.remove(p)
            os.rmdir(tmp_dir)
        except OSError:
            pass
    return 0


# --------------------------------------------------------------------------- #
# Skill: holds all in-memory state; one instance per server process.
# --------------------------------------------------------------------------- #
class BaoSkill:
    def __init__(self):
        self.pool = ConnectionPool(POOL_SIZE_DEFAULT)
        self._bg_pool = ConnectionPool(max(1, POOL_SIZE_DEFAULT // 2))
        self._snapshot = (None, 0)  # (BaoRegression, nn_weights mtime)
        self.model_dir = None
        self.n_trained = 0
        self.retrain_threshold = RETRAIN_THRESHOLD_DEFAULT
        self._state_lock = threading.Lock()
        self._training_in_progress = False
        self._train_proc = None  # Popen of the in-flight training subprocess
        self._shutting_down = False  # set by persist(); /optimize then 503s
        self._persist_lock = threading.Lock()  # single-flight: concurrent /shutdown runs persist once
        self._bg_queue = queue_mod.Queue()
        self._bg_thread = threading.Thread(target=self._bg_loop, daemon=True)
        self._bg_thread.start()

    def _apply_config(self, config):
        self.retrain_threshold = int(config.get("retrain_threshold", RETRAIN_THRESHOLD_DEFAULT))
        pool_size = int(config.get("connection_pool_size", POOL_SIZE_DEFAULT))
        if self.pool.size != pool_size:
            self.pool.close_all()
            self.pool = ConnectionPool(pool_size)

    def ensure_loaded(self, config):
        model_dir = config.get("model_dir") or DEFAULT_MODEL_DIR
        self._apply_config(config)
        if self._snapshot[0] is None or model_dir != self.model_dir:
            self._load_initial(model_dir)

    def _load_initial(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        nn_path = _model_path(model_dir)
        self.model_dir = model_dir
        if not os.path.exists(nn_path):
            # Bootstrap: no pre-trained model. Fall back to arm 0 (default
            # PostgreSQL) and train a fresh model once enough experiences
            # accumulate.
            self.n_trained = 0
            self._snapshot = (None, 0)
            return
        reg = BaoRegression(have_cache_data=True)
        reg.load(model_dir)
        mtime = os.path.getmtime(nn_path)
        self.n_trained = _read_n_trained(model_dir)
        self._snapshot = (reg, mtime)

    def _maybe_reload_model(self):
        if self.model_dir is None:
            return
        nn_path = _model_path(self.model_dir)
        if not os.path.exists(nn_path):
            return  # still bootstrapping
        try:
            mtime = os.path.getmtime(nn_path)
        except OSError:
            return
        if mtime == self._snapshot[1] and self._snapshot[0] is not None:
            return
        try:
            reg = BaoRegression(have_cache_data=True)
            reg.load(self.model_dir)
            self._snapshot = (reg, mtime)
            self.n_trained = _read_n_trained(self.model_dir)
        except Exception as e:
            print(f"model reload failed: {e}", file=sys.stderr)

    # -- DB helper --
    def _explain(self, dsn, query, set_hints, with_analyze, pool):
        conn = pool.get(dsn)
        ok = False
        try:
            conn.autocommit = True
            plan = _get_explain_plan(conn, query, set_hints, with_analyze=with_analyze)
            ok = True
            return plan
        finally:
            if ok:
                try:
                    conn.rollback()
                except Exception:
                    pass
                pool.put(dsn, conn)
            else:
                pool.discard(dsn, conn)

    # -- Critical path (lock-free) --
    def optimize(self, dsn, query, optimize_only):
        self._maybe_reload_model()
        t0 = time.time()
        model = self._snapshot[0]

        arm_set_hints = {a: _arm_to_set_hints(a) for a in range(5)}

        # Snapshot buffer state once (query-independent) using a foreground conn.
        conn = self.pool.get(dsn)
        try:
            conn.autocommit = True
            buffer_state = _get_buffer_state(conn)
        finally:
            try:
                conn.rollback()
            except Exception:
                pass
            self.pool.put(dsn, conn)

        # Collect all 5 arm plans in parallel (cost-only EXPLAIN).
        plans = [None] * 5

        def _one(a):
            try:
                plans[a] = self._explain(dsn, query, arm_set_hints[a], False, self.pool)
            except Exception as e:
                print(f"[bao-wrapper] Arm {a} EXPLAIN failed: {e}", file=sys.stderr)
                plans[a] = None

        with ThreadPoolExecutor(max_workers=max(1, self.pool.size)) as ex:
            list(ex.map(_one, range(5)))

        valid_arms = [a for a in range(5) if plans[a] is not None]
        if not valid_arms:
            return {"optimized_query": query,
                    "metadata": {"strategy_type": "learned-arm-selection",
                                 "optimization_time": round(time.time() - t0, 4),
                                 "estimated_impact": 0.0, "num_arms_evaluated": 0,
                                 "selected_arm": None,
                                 "mode": "inference-only" if optimize_only else "online-training",
                                 "error": "no candidate arms could be planned"}}

        # Batched featurize + predict.
        for a in valid_arms:
            _strip_per_node_buffers(plans[a][0]["Plan"])
        plan_entries = [{"Plan": plans[a][0]["Plan"], "Buffers": buffer_state} for a in valid_arms]
        arm_scores = []
        if model is not None:
            try:
                preds = model.predict(plan_entries)
                for i, a in enumerate(valid_arms):
                    try:
                        arm_scores.append((a, float(preds[i][0])))
                    except Exception:
                        continue
            except Exception as e:
                print(f"[bao-wrapper] batched predict failed: {e}", file=sys.stderr)

        if arm_scores:
            selected_arm = min(arm_scores, key=lambda x: x[1])[0]
            default_pred = next((s for a, s in arm_scores if a == 0), None)
            selected_pred = next((s for a, s in arm_scores if a == selected_arm), None)
            if default_pred is not None and selected_pred is not None and default_pred > 0:
                estimated_impact = max(0.0, (default_pred - selected_pred) / default_pred * 100.0)
            else:
                estimated_impact = 0.0
        else:
            # No model yet (bootstrapping) or predict failed → default PostgreSQL.
            selected_arm = 0
            estimated_impact = 0.0

        optimized_query = _render_hint_comment(arm_set_hints[selected_arm]) + query
        mode = "inference-only" if optimize_only else "online-training"
        result = {"optimized_query": optimized_query,
                  "metadata": {"strategy_type": "learned-arm-selection",
                               "optimization_time": round(time.time() - t0, 4),
                               "estimated_impact": round(estimated_impact, 4),
                               "num_arms_evaluated": len(valid_arms),
                               "selected_arm": selected_arm, "mode": mode,
                               "model_loaded": model is not None,
                               "buffer_state_relations": len(buffer_state)}}

        if not optimize_only:
            # Safe-gate: skip background EXPLAIN ANALYZE / recording if the
            # selected arm's plan is catastrophically costlier than the
            # default arm's (would hang the background thread on execution).
            default_cost = _plan_total_cost(plans[0]) if plans[0] is not None else None
            selected_cost = _plan_total_cost(plans[selected_arm]) if plans[selected_arm] is not None else None
            if not _cost_ratio_too_high(selected_cost, default_cost):
                self._bg_queue.put(("collect", dsn, query, arm_set_hints, selected_arm, buffer_state))
        return result

    # -- Background training-data collection + training spawn --
    def _bg_loop(self):
        while True:
            task = self._bg_queue.get()
            try:
                self._collect_training_data(task)
            except Exception as e:
                print(f"bg training-data collection failed: {e}", file=sys.stderr)

    def _collect_training_data(self, task):
        _, dsn, query, arm_set_hints, selected_arm, buffer_state = task
        spawn = False
        try:
            for arm_idx in {0, selected_arm}:
                plan = self._explain(dsn, query, arm_set_hints[arm_idx], True, self._bg_pool)
                plan_entry = plan[0]
                exec_time = plan_entry.get("Execution Time", 0.0)
                _strip_per_node_buffers(plan_entry["Plan"])
                plan_entry["Buffers"] = buffer_state
                storage.record_reward(plan_entry, float(exec_time), os.getpid())
            with self._state_lock:
                experiences = storage.experience()
                if len(experiences) - self.n_trained >= self.retrain_threshold \
                        and not self._training_in_progress:
                    self._training_in_progress = True
                    self.n_trained = len(experiences)
                    spawn = True
        except Exception as e:
            print(f"[bao-wrapper] training-data collection failed: {e}", file=sys.stderr)
        if spawn:
            self._spawn_training_worker()

    def _spawn_training_worker(self):
        model_dir = self.model_dir

        def _run():
            try:
                p = subprocess.Popen(
                    [sys.executable, os.path.abspath(__file__), "--train", "--model-dir", model_dir]
                )
                with self._state_lock:
                    self._train_proc = p
                p.wait()
            except Exception as e:
                print(f"training worker failed: {e}", file=sys.stderr)
            finally:
                with self._state_lock:
                    self._training_in_progress = False
                    self._train_proc = None

        threading.Thread(target=_run, daemon=True).start()

    def _wait_for_training(self, timeout=300):
        """Block until the in-flight training subprocess finishes (its model
        save completes) or the timeout expires. Polled via the flag set in
        _spawn_training_worker's finally block (no waitpid race with the
        worker thread's p.wait())."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._state_lock:
                in_progress = self._training_in_progress
            if not in_progress:
                return
            time.sleep(0.5)

    def persist(self):
        if not self._persist_lock.acquire(blocking=False):
            # Another persist() is in flight (concurrent /shutdown or SIGTERM
            # racing a skill_runner stop_wrapper); it will complete the save.
            # Returning here avoids two in-process run_training calls
            # clobbering the same pid-based temp dir.
            return
        try:
                # Don't bypass the model save on shutdown. Experiences are already in
                # sqlite (record_reward commits per call), but the model is only saved
                # by the training subprocess when retrain_threshold is reached. Wait
                # for any in-flight subprocess so its save isn't abandoned on exit,
                # then if experiences were collected since the last training, run a
                # final in-process training so the persisted model reflects all data.
                self._shutting_down = True
                self._wait_for_training(timeout=300)
                if self.model_dir is None:
                    return
                try:
                    experiences = storage.experience()
                except Exception:
                    experiences = []
                with self._state_lock:
                    untrained = len(experiences) - self.n_trained
                if untrained > 0:
                    try:
                        run_training(self.model_dir)
                    except Exception as e:
                        print(f"final training on shutdown failed: {e}", file=sys.stderr)
        finally:
            self._persist_lock.release()
            # Only the persist that acquired the lock shuts down the
            # server — a guarded-out persist (concurrent /shutdown)
            # must NOT start SERVER.shutdown, or daemon_threads=True
            # would let main() exit and kill this in-flight save.
            if SERVER is not None:
                threading.Thread(target=SERVER.shutdown, daemon=True).start()

    def state_summary(self):
        model = self._snapshot[0]
        try:
            experiences = storage.experience()
        except Exception:
            experiences = []
        with self._state_lock:
            return {"experiences": len(experiences), "n_trained": self.n_trained,
                    "model_loaded": model is not None, "model_dir": self.model_dir,
                    "pool_size": self.pool.size,
                    "training_in_progress": self._training_in_progress}


# --------------------------------------------------------------------------- #
# HTTP server.
# --------------------------------------------------------------------------- #
SERVER = None
SKILL = BaoSkill()


def _drain_and_stop():
    try:
        SKILL.persist()  # persist() starts SERVER.shutdown in its finally (only if it owns the lock)
    except Exception as e:
        print(f"persist failed: {e}", file=sys.stderr)
        if SERVER is not None:
            threading.Thread(target=SERVER.shutdown, daemon=True).start()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send_json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/health"):
            self._send_json(200, {"status": "ok"})
        elif self.path.startswith("/state"):
            self._send_json(200, SKILL.state_summary())
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/optimize":
            if SKILL._shutting_down:
                self._send_json(503, {"error": "server is shutting down"})
                return
            try:
                n = int(self.headers.get("Content-Length", "0") or "0")
                raw = self.rfile.read(n).decode() if n else "{}"
                req = json.loads(raw) if raw else {}
            except Exception as e:
                self._send_json(400, {"error": f"bad request body: {e}"})
                return
            dsn = req.get("dsn")
            query = req.get("query")
            optimize_only = bool(req.get("optimize_only"))
            cfg = req.get("config")
            if isinstance(cfg, str) and cfg:
                try:
                    cfg = json.loads(cfg)
                except Exception:
                    cfg = {"model_dir": cfg}
            cfg = cfg or {}
            if not isinstance(cfg, dict):
                self._send_json(400, {"error": "config must be a JSON object"})
                return
            if not dsn or not query:
                self._send_json(400, {"error": "dsn and query are required"})
                return
            try:
                SKILL.ensure_loaded(cfg)
                result = SKILL.optimize(dsn, query, optimize_only)
            except FileNotFoundError as e:
                self._send_json(400, {"error": str(e)})
            except Exception as e:
                self._send_json(500, {"error": str(e)})
            else:
                self._send_json(200, result)
        elif self.path == "/shutdown":
            self._send_json(200, {"status": "shutting down"})
            _drain_and_stop()
        else:
            self._send_json(404, {"error": "not found"})


def _signal_handler(signum, frame):
    _drain_and_stop()


def main():
    # storage.py opens "bao.db" relative to CWD; run from bao_server/.
    if os.path.basename(SCRIPT_DIR) == "bao_server":
        os.chdir(SCRIPT_DIR)
    elif os.path.isdir(os.path.join(SCRIPT_DIR, "bao_server")):
        os.chdir(os.path.join(SCRIPT_DIR, "bao_server"))

    parser = argparse.ArgumentParser(description="Bao Query Optimizer server")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "0")))
    parser.add_argument("--train", action="store_true", help="Run a single training cycle and exit (subprocess mode).")
    parser.add_argument("--model-dir", default=None, help="Model directory (for --train mode).")
    args = parser.parse_args()

    if args.train:
        if os.path.basename(SCRIPT_DIR) != "bao_server" and os.path.isdir(os.path.join(SCRIPT_DIR, "bao_server")):
            os.chdir(os.path.join(SCRIPT_DIR, "bao_server"))
        sys.exit(run_training(args.model_dir or DEFAULT_MODEL_DIR))

    if not args.port:
        print("error: --port or PORT env required", file=sys.stderr)
        sys.exit(1)

    global SERVER
    SERVER = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    print(f"Bao skill server listening on 127.0.0.1:{args.port}", flush=True)
    try:
        SERVER.serve_forever()
    finally:
        SKILL.pool.close_all()
        SKILL._bg_pool.close_all()


if __name__ == "__main__":
    main()
