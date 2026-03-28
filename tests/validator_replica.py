"""
OpenEnv Hackathon Validator (CLI) — Based on Real Meta Source Code.

Replicates the checks from:
  - openenv validate .       (local structure)
  - openenv validate --url   (runtime endpoints)
  - Round 1 programmatic scoring

Source: https://github.com/meta-pytorch/OpenEnv/blob/main/src/openenv/cli/_validation.py
"""

import os
import sys
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi.testclient import TestClient
from server.app import app

client = TestClient(app)


def run_validation():
    print("\n" + "=" * 70)
    print("  OpenEnv Hackathon Validator (based on real Meta source)")
    print("=" * 70)

    results = []

    def check(section, name, passed, detail=""):
        icon = "✅" if passed else "❌"
        print(f"  {icon} [{section}] {name}" + (f" — {detail}" if detail and not passed else ""))
        results.append((section, name, passed, detail))

    # --- A: Local structure (mirrors validate_multi_mode_deployment) ---
    print("\n--- A. Local Structure ---")
    check("Local", "openenv.yaml exists", os.path.exists("openenv.yaml"))
    check("Local", "pyproject.toml exists", os.path.exists("pyproject.toml"))
    check("Local", "uv.lock exists", os.path.exists("uv.lock"))

    if os.path.exists("pyproject.toml"):
        with open("pyproject.toml") as f:
            pt = f.read()
        check("Local", "[project.scripts] server with :main",
              "[project.scripts]" in pt and ":main" in pt)
        check("Local", "openenv-core dependency",
              "openenv-core" in pt or "openenv" in pt.split("[project.scripts]")[0])

    app_path = os.path.join("server", "app.py")
    if os.path.exists(app_path):
        with open(app_path) as f:
            ac = f.read()
        check("Local", "server/app.py has def main(", "def main(" in ac)
        check("Local", "server/app.py has __name__ guard", "__name__" in ac and "main()" in ac)

    check("Local", "Dockerfile exists",
          os.path.exists("Dockerfile") or os.path.exists("server/Dockerfile"))

    # --- B: Runtime endpoints (mirrors validate_running_environment) ---
    print("\n--- B. Runtime Endpoints ---")

    r = client.get("/openapi.json")
    oa_ok = False
    if r.status_code == 200:
        try:
            oa = r.json()
            oa_ok = isinstance(oa.get("info"), dict) and isinstance(oa["info"].get("version"), str)
        except Exception:
            pass
    check("Runtime", "GET /openapi.json → info.version", oa_ok)

    r = client.get("/health")
    check("Runtime", "GET /health → status: healthy",
          r.status_code == 200 and r.json().get("status") == "healthy")

    r = client.get("/metadata")
    md_ok = False
    if r.status_code == 200:
        try:
            md = r.json()
            md_ok = isinstance(md.get("name"), str) and isinstance(md.get("description"), str)
        except Exception:
            pass
    check("Runtime", "GET /metadata → name + description", md_ok,
          "404" if r.status_code == 404 else "")

    r = client.get("/schema")
    sc_ok = False
    if r.status_code == 200:
        try:
            sc = r.json()
            sc_ok = all(isinstance(sc.get(k), dict) for k in ("action", "observation", "state"))
        except Exception:
            pass
    check("Runtime", "GET /schema → action/observation/state", sc_ok,
          "404" if r.status_code == 404 else "")

    # MCP endpoint
    r = client.post("/mcp", json={})
    mcp_ok = False
    if r.status_code == 200:
        try:
            mcp_ok = r.json().get("jsonrpc") == "2.0"
        except Exception:
            pass
    check("Runtime", "POST /mcp → jsonrpc: 2.0", mcp_ok,
          f"status={r.status_code}" if not mcp_ok else "")

    # Mode consistency
    try:
        paths = client.get("/openapi.json").json().get("paths", {})
        mode_ok = "/reset" in paths and "/step" in paths and "/state" in paths
    except Exception:
        mode_ok = False
    check("Runtime", "OpenAPI paths: /reset, /step, /state", mode_ok)

    # --- C: Round 1 programmatic checks ---
    print("\n--- C. Round 1 Programmatic ---")

    r = client.post("/reset", json={"seed": 42, "options": {"difficulty": "easy"}})
    check("Round1", "POST /reset → 200 + observation",
          r.status_code == 200 and "observation" in r.json())

    r = client.post("/step", json={"decision": "allow", "reason": "test"})
    step_ok = r.status_code == 200 and all(k in r.json() for k in ("reward", "done", "observation"))
    check("Round1", "POST /step → reward/done/observation", step_ok)

    # Episode completion + score bounds
    client.post("/reset", json={"seed": 42, "options": {"difficulty": "easy"}})
    done = False
    last_r = None
    while not done:
        last_r = client.post("/step", json={"decision": "allow", "reason": "test"})
        done = last_r.json().get("done", True)
    score = last_r.json().get("observation", {}).get("episode_score", -1) if last_r else -1
    check("Round1", "Episode score in [0.0, 1.0]",
          isinstance(score, (int, float)) and 0.0 <= score <= 1.0,
          f"score={score:.4f}")

    # Post-done rejection
    r = client.post("/step", json={"decision": "allow", "reason": "post-done"})
    check("Round1", "Step after done → HTTP 400", r.status_code >= 400, f"Got {r.status_code}")

    # Invalid task_id rejection
    r = client.post("/reset", json={"options": {"task_id": "nonexistent_999"}})
    check("Round1", "Invalid task_id → HTTP 400", r.status_code >= 400, f"Got {r.status_code}")

    # Pydantic rejection
    r = client.post("/step", json={"UNKNOWN": "garbage"})
    check("Round1", "Malformed JSON → 422", r.status_code == 422, f"Got {r.status_code}")

    # Baseline
    t0 = time.time()
    r = client.get("/baseline")
    baseline_ok = r.status_code == 200 and "average_score" in r.json()
    check("Round1", "GET /baseline returns scores", baseline_ok,
          f"{time.time() - t0:.2f}s, avg={r.json().get('average_score', '?')}" if baseline_ok else "")

    # Task count
    r = client.get("/tasks")
    if r.status_code == 200:
        j = r.json()
        check("Round1", "3+ tasks across difficulties",
              j.get("total_tasks", 0) >= 3 and len(j.get("tasks_by_difficulty", {})) >= 2,
              f"total={j.get('total_tasks')}")

    # Deterministic
    client.post("/reset", json={"seed": 42})
    t1 = client.post("/step", json={"decision": "allow", "reason": "x"}).json().get("observation", {}).get("task_id")
    client.post("/reset", json={"seed": 42})
    t2 = client.post("/step", json={"decision": "allow", "reason": "x"}).json().get("observation", {}).get("task_id")
    check("Round1", "Deterministic (same seed → same task)", t1 == t2 and t1 is not None)

    # State leakage
    client.post("/reset", json={"options": {"task_id": "easy_001"}})
    client.post("/step", json={"decision": "allow", "reason": "test"})
    obs2 = client.post("/reset", json={"options": {"task_id": "easy_002"}}).json().get("observation", {})
    leak_ok = obs2.get("current_step") == 1 and not obs2.get("previous_decisions")
    check("Round1", "No state leakage between episodes", leak_ok)

    # --- Summary ---
    passed = sum(1 for _, _, p, _ in results if p)
    total = len(results)
    failed = [(s, n, d) for s, n, p, d in results if not p]

    print("\n" + "=" * 70)
    print(f"  RESULT: {passed}/{total} checks passed")
    print("=" * 70)

    if failed:
        print("\n  FAILED CHECKS:")
        for s, n, d in failed:
            print(f"    ❌ [{s}] {n}" + (f" — {d}" if d else ""))

    if passed == total:
        print("\n  ✅ ALL CHECKS PASSED — Ready for submission.")
    else:
        print(f"\n  ⚠️  {len(failed)} check(s) need attention.")


if __name__ == "__main__":
    run_validation()
