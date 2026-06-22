#!/usr/bin/env python3
"""
Bao query optimizer CLI wrapper.

Wraps the Bao tree-convolution learned optimizer (BaoRegression / BaoNet)
into a CLI tool that takes a SQL query, evaluates the 5 Bao "arms"
(GUC combinations), predicts per-arm latency with the Bao model, and emits
the optimized query with the selected arm's GUCs as `pg_hint_plan` `Set(...)`
hints prepended to the original SQL.

Using `Set(...)` hints (rather than `SET` statements) keeps each invocation
self-contained — GUCs don't leak across queries in the same session — and
lets the caller run the optimized query without superuser privileges.

Modes:
  * Online training (default): after model selection, runs EXPLAIN ANALYZE
    on the default plan and the model-selected arm's plan, records both as
    (plan, latency) experiences, and retrains when 25 new experiences
    accumulate (mirrors the chunk size used by run_queries.py).
  * Inference-only (--optimize-only): scores arms via EXPLAIN (cost only),
    no execution, no training.
"""
import argparse
import contextlib
import io
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import psycopg2

from model import BaoRegression
from reg_blocker import _ALL_OPTIONS, _arm_idx_to_hints
import storage


DEFAULT_MODEL_DIR = "bao_default_model"
DEFAULT_DB_PATH = "bao.db"
RETRAIN_THRESHOLD = 25


# ---------------------------------------------------------------------------
# Arm -> pg_hint_plan Set(...) hint conversion
# ---------------------------------------------------------------------------
def _arm_to_set_hints(arm_idx):
    """
    Convert a Bao arm index into a list of `pg_hint_plan` Set(...) hint
    strings, e.g. ["Set(enable_nestloop off)", "Set(enable_hashjoin on)", ...].

    Bao's arms are defined in reg_blocker._arm_idx_to_hints as a sequence of
    `SET <opt> TO <val>` statements: first all GUCs to off, then the desired
    ones back to on. When executed as sequential SQL statements the final
    value wins, but pg_hint_plan Set(...) hints in a single comment can't
    conflict — so we collapse to the final state per GUC.

    Arm 0 (all defaults on) emits an empty list — no hint needed.
    """
    final_state = {}
    for stmt in _arm_idx_to_hints(arm_idx):
        parts = stmt.split()
        if len(parts) >= 4 and parts[0].upper() == "SET" and parts[2].upper() == "TO":
            opt = parts[1]
            val = parts[3]
            final_state[opt] = val
    return [f"Set({opt} {val})" for opt, val in final_state.items()]


def _render_hint_comment(set_hints):
    """Build a `/*+ ... */` comment from a list of Set(...) hint strings."""
    if not set_hints:
        return ""
    return "/*+ " + " ".join(set_hints) + " */ "


# ---------------------------------------------------------------------------
# Buffer state snapshot (query-independent)
# ---------------------------------------------------------------------------
def _get_buffer_state(conn):
    """
    Build a {relation_name: block_count} dict representing the current
    PostgreSQL shared buffer cache state.

    This mirrors what the Bao C extension produces in `buffer_state()`
    (pg_extension/bao_bufferstate.h): it iterates all NBuffers and counts
    blocks per relation, excluding system tables. The result is attached to
    every plan as a top-level "Buffers" field so Bao's featurize can derive
    per-leaf buffer counts via `get_buffer_count_for_leaf`.

    The buffer state is a snapshot of the cache at planning time and is
    independent of the query being optimized — Bao uses it as a feature
    that captures how much of each relation is already cached.

    Requires the `pg_buffercache` extension to be installed on the target
    database. Falls back to an empty dict if unavailable.
    """
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
        print(f"[bao-wrapper] pg_buffercache query failed ({e}); "
              f"using empty buffer state. Install pg_buffercache for full accuracy.",
              file=sys.stderr)
        return {}

def _strip_per_node_buffers(plan_node):
    """
    Remove PostgreSQL's per-node buffer keys from an EXPLAIN plan tree.

    EXPLAIN (ANALYZE, BUFFERS) annotates every node with individual
    `Shared Hit Blocks` / `Shared Read Blocks` / etc. keys. Bao's featurize
    only consumes the top-level `Buffers` dict (which we synthesize from
    pg_buffercache), so we strip these per-node keys to avoid confusing the
    TreeBuilder.
    """
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


# ---------------------------------------------------------------------------
# EXPLAIN helpers
# ---------------------------------------------------------------------------
def _get_explain_plan(conn, sql, set_hints, with_analyze=False):
    """
    Run EXPLAIN on `sql` with the given pg_hint_plan Set(...) hints applied
    via a prepended `/*+ ... */` comment. Returns the parsed plan JSON.

    Using hint comments instead of session-level SET statements ensures
    GUC changes don't leak across queries in the same connection.
    """
    hint_comment = _render_hint_comment(set_hints)
    stmt = ("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) "
            if with_analyze else "EXPLAIN (FORMAT JSON) ")
    full_stmt = stmt + hint_comment + sql
    with conn.cursor() as cur:
        cur.execute("SET statement_timeout TO 120000")
        cur.execute(full_stmt)
        row = cur.fetchall()
        plan = row[0][0]
        if isinstance(plan, str):
            plan = json.loads(plan)
        return plan


# ---------------------------------------------------------------------------
# Model state management
# ---------------------------------------------------------------------------
def load_model(model_dir):
    if not os.path.isdir(model_dir) or not os.path.exists(os.path.join(model_dir, "nn_weights")):
        return None
    try:
        reg = BaoRegression(have_cache_data=True)
        reg.load(model_dir)
        return reg
    except Exception as e:
        print(f"[bao-wrapper] Failed to load model from {model_dir}: {e}",
              file=sys.stderr)
        return None


def train_model(model_dir, verbose=False):
    """Train a fresh model from all stored experiences and save to model_dir."""
    experiences = storage.experience()
    if not experiences:
        return None
    x = [row[0] for row in experiences]
    y = np.array([row[1] for row in experiences], dtype=np.float64)
    reg = BaoRegression(have_cache_data=True, verbose=verbose)
    with contextlib.redirect_stdout(io.StringIO()):
        reg.fit(x, y)
    reg.save(model_dir)
    return reg


# ---------------------------------------------------------------------------
# Main optimization flow
# ---------------------------------------------------------------------------
def optimize(args):
    t_start = time.time()
    model_dir = args.config or DEFAULT_MODEL_DIR

    conn = psycopg2.connect(args.dsn)
    conn.autocommit = True

    # Snapshot the buffer cache once (query-independent). Bao's featurize
    # uses this as a per-plan feature.
    buffer_state = _get_buffer_state(conn)

    # Build arm -> Set(...) hints map. Arm 0 (default) has no Set hints.
    arm_set_hints = {arm: _arm_to_set_hints(arm) for arm in range(5)}

    # Step 1: collect candidate plans from each arm via EXPLAIN (cost only).
    arm_plans = []  # list of (arm_idx, explain_json_list)
    for arm_idx in range(5):
        try:
            plan = _get_explain_plan(conn, args.query, arm_set_hints[arm_idx],
                                     with_analyze=False)
            # Strip per-node buffer keys (Bao only consumes the top-level
            # buffer dict we synthesize).
            _strip_per_node_buffers(plan[0]["Plan"])
            arm_plans.append((arm_idx, plan))
        except Exception as e:
            print(f"[bao-wrapper] Arm {arm_idx} EXPLAIN failed: {e}",
                  file=sys.stderr)
            continue

    if not arm_plans:
        return {
            "optimized_query": args.query,
            "metadata": {
                "strategy_type": "learned-arm-selection",
                "optimization_time": round(time.time() - t_start, 4),
                "estimated_impact": 0.0,
                "num_arms_evaluated": 0,
                "selected_arm": None,
                "mode": "inference-only" if args.optimize_only else "online-training",
                "error": "no candidate arms could be planned",
            },
        }

    # Step 2: score each arm with the model (or fall back to arm 0 = default).
    model = load_model(model_dir)
    arm_scores = []
    if model is not None:
        for arm_idx, plan in arm_plans:
            try:
                plan_entry = {"Plan": plan[0]["Plan"], "Buffers": buffer_state}
                pred = model.predict([plan_entry])
                arm_scores.append((arm_idx, float(pred[0][0])))
            except Exception as e:
                print(f"[bao-wrapper] Predict failed for arm {arm_idx}: {e}",
                      file=sys.stderr)

    if arm_scores:
        selected_arm = min(arm_scores, key=lambda x: x[1])[0]
        default_pred = next((s for a, s in arm_scores if a == 0), None)
        selected_pred = next((s for a, s in arm_scores if a == selected_arm), None)
        if default_pred is not None and selected_pred is not None and default_pred > 0:
            estimated_impact = max(0.0, (default_pred - selected_pred) / default_pred * 100.0)
        else:
            estimated_impact = 0.0
    else:
        selected_arm = 0
        estimated_impact = 0.0

    # Step 3: render the optimized query with the selected arm's Set(...) hints.
    set_hints = arm_set_hints[selected_arm]
    optimized_query = _render_hint_comment(set_hints) + args.query

    mode = "inference-only" if args.optimize_only else "online-training"

    # Step 4 (training mode only): execute default + selected arm plans via
    # EXPLAIN ANALYZE and record (plan, latency) experiences. Retrain if
    # threshold reached. Bao's storage/featurize expect each stored plan to
    # be the outer EXPLAIN dict {"Plan": ..., "Execution Time": ...}.
    trained = False
    if not args.optimize_only:
        try:
            for arm_idx in {0, selected_arm}:
                plan = _get_explain_plan(conn, args.query, arm_set_hints[arm_idx],
                                         with_analyze=True)
                plan_entry = plan[0]
                exec_time = plan_entry.get("Execution Time", 0.0)
                # Strip per-node buffer keys, then attach the cache snapshot.
                _strip_per_node_buffers(plan_entry["Plan"])
                plan_entry["Buffers"] = buffer_state
                storage.record_reward(plan_entry, float(exec_time), os.getpid())

            experiences = storage.experience()
            model = load_model(model_dir)
            n_trained = model.num_items_trained_on() if model else 0
            if len(experiences) - n_trained >= RETRAIN_THRESHOLD:
                print(f"[bao-wrapper] Retraining model ({len(experiences)} "
                      f"experiences, last trained on {n_trained})...",
                      file=sys.stderr)
                train_model(model_dir, verbose=False)
                trained = True
        except Exception as e:
            print(f"[bao-wrapper] Training step failed: {e}", file=sys.stderr)

    conn.close()

    return {
        "optimized_query": optimized_query,
        "metadata": {
            "strategy_type": "learned-arm-selection",
            "optimization_time": round(time.time() - t_start, 4),
            "estimated_impact": round(estimated_impact, 4),
            "num_arms_evaluated": len(arm_plans),
            "selected_arm": selected_arm,
            "mode": mode,
            "model_loaded": model is not None,
            "retrained": trained,
            "buffer_state_relations": len(buffer_state),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Bao learned query optimizer CLI wrapper.")
    parser.add_argument("--dsn", required=True,
                        help="PostgreSQL DSN (libpq connection string)")
    parser.add_argument("--query", required=True,
                        help="SQL query to optimize")
    parser.add_argument("--config", default=None,
                        help="Path to the Bao model directory "
                             "(default: ./bao_default_model)")
    parser.add_argument("--optimize-only", action="store_true",
                        help="Inference-only mode: no execution, no training.")
    args = parser.parse_args()

    # storage.py opens "bao.db" relative to CWD. Run from bao_server/ so the
    # experience DB and model directory stay colocated with the source code.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(script_dir) == "bao_server":
        os.chdir(script_dir)
    elif os.path.isdir(os.path.join(script_dir, "bao_server")):
        os.chdir(os.path.join(script_dir, "bao_server"))

    global DEFAULT_MODEL_DIR
    if args.config:
        DEFAULT_MODEL_DIR = args.config

    result = optimize(args)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
