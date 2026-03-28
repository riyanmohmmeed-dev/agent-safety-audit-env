"""
OpenEnv Hackathon Validator — Based on Real Meta Source Code.

This validator replicates the ACTUAL checks performed by:
  1. `openenv validate .`          (local structure checks)
  2. `openenv validate --url ...`  (runtime endpoint checks)
  3. Hackathon Round 1 rubric      (programmatic + LLM scoring)

Source of truth:
  https://github.com/meta-pytorch/OpenEnv/blob/main/src/openenv/cli/commands/validate.py
  https://github.com/meta-pytorch/OpenEnv/blob/main/src/openenv/cli/_validation.py
"""

import streamlit as st
import time
import sys
import os
import subprocess
import glob

# Ensure the parent directory is in the Python path so we can import server.app
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi.testclient import TestClient
from server.app import app

client = TestClient(app)

st.set_page_config(page_title="OpenEnv Validator", page_icon="🔍", layout="centered")

st.title("🔍 OpenEnv Hackathon Validator")
st.caption(
    "Based on the real `openenv validate` source code from "
    "[meta-pytorch/OpenEnv](https://github.com/meta-pytorch/OpenEnv/blob/main/src/openenv/cli/_validation.py)"
)


def perform_scan():
    results = []  # list of (section, check_name, passed, details)

    st.markdown("---")

    # =================================================================
    # SECTION A: `openenv validate .` — Local structure checks
    # Mirrors: validate_multi_mode_deployment() in _validation.py
    # =================================================================
    with st.status("A. Local Structure Checks (`openenv validate .`)...", expanded=True) as status:
        local_pass = True

        # A1: openenv.yaml exists
        a1 = os.path.exists("openenv.yaml")
        results.append(("Local", "openenv.yaml exists", a1, ""))
        st.write(f"  {'✅' if a1 else '❌'} openenv.yaml exists" + ("" if a1 else " — **FATAL: Not an OpenEnv environment**"))
        if not a1:
            local_pass = False

        # A2: pyproject.toml exists
        a2 = os.path.exists("pyproject.toml")
        results.append(("Local", "pyproject.toml exists", a2, ""))
        st.write(f"  {'✅' if a2 else '❌'} pyproject.toml exists")
        if not a2:
            local_pass = False

        # A3: uv.lock exists
        a3 = os.path.exists("uv.lock")
        results.append(("Local", "uv.lock exists", a3, "Run `uv lock` to generate"))
        st.write(f"  {'✅' if a3 else '⚠️'} uv.lock exists" + ("" if a3 else " — run `uv lock`"))

        # A4: pyproject.toml has [project.scripts] with server entry containing :main
        a4 = False
        a4_detail = ""
        if a2:
            with open("pyproject.toml") as f:
                content = f.read()
            if "[project.scripts]" in content and "server" in content and ":main" in content:
                a4 = True
            else:
                a4_detail = "Missing [project.scripts] server = '...app:main'"
        results.append(("Local", "[project.scripts] server entry", a4, a4_detail))
        st.write(f"  {'✅' if a4 else '❌'} pyproject.toml has [project.scripts] server entry with :main")
        if not a4:
            local_pass = False

        # A5: server/app.py exists with def main( and __name__ guard
        a5 = False
        a5_detail = ""
        app_path = os.path.join("server", "app.py")
        if os.path.exists(app_path):
            with open(app_path) as f:
                app_content = f.read()
            has_main = "def main(" in app_content
            has_guard = "__name__" in app_content and "main()" in app_content
            if has_main and has_guard:
                a5 = True
            else:
                a5_detail = f"main()={'found' if has_main else 'MISSING'}, __name__ guard={'found' if has_guard else 'MISSING'}"
        else:
            a5_detail = "server/app.py not found"
        results.append(("Local", "server/app.py has main() + guard", a5, a5_detail))
        st.write(f"  {'✅' if a5 else '❌'} server/app.py has `def main(` and `__name__` guard")
        if not a5:
            local_pass = False

        # A6: openenv-core dependency in pyproject.toml
        a6 = False
        if a2:
            a6 = "openenv-core" in content or "openenv" in content.split("[project.scripts]")[0]
        results.append(("Local", "openenv-core dependency", a6, ""))
        st.write(f"  {'✅' if a6 else '❌'} openenv-core dependency declared")

        # A7: Dockerfile exists
        a7 = os.path.exists("Dockerfile") or os.path.exists(os.path.join("server", "Dockerfile"))
        results.append(("Local", "Dockerfile exists", a7, ""))
        st.write(f"  {'✅' if a7 else '❌'} Dockerfile exists (docker deployment mode)")

        time.sleep(0.3)
        if local_pass:
            status.update(label="A. Local Structure Checks ✅ PASSED", state="complete", expanded=False)
        else:
            status.update(label="A. Local Structure Checks ❌ FAILED", state="error", expanded=False)

    # =================================================================
    # SECTION B: `openenv validate --url` — Runtime endpoint checks
    # Mirrors: validate_running_environment() in _validation.py
    # =================================================================
    with st.status("B. Runtime Endpoint Checks (`openenv validate --url`)...", expanded=True) as status:
        runtime_pass = True

        # B1: GET /openapi.json — OpenAPI version available
        r = client.get("/openapi.json")
        b1 = False
        if r.status_code == 200:
            try:
                oa = r.json()
                if isinstance(oa.get("info"), dict) and isinstance(oa["info"].get("version"), str):
                    b1 = True
            except Exception:
                pass
        results.append(("Runtime", "GET /openapi.json returns info.version", b1, ""))
        st.write(f"  {'✅' if b1 else '❌'} GET /openapi.json returns OpenAPI info.version")
        if not b1:
            runtime_pass = False

        # B2: GET /health returns {"status": "healthy"}
        r = client.get("/health")
        b2 = False
        if r.status_code == 200:
            try:
                b2 = r.json().get("status") == "healthy"
            except Exception:
                pass
        results.append(("Runtime", "GET /health → status: healthy", b2, ""))
        st.write(f"  {'✅' if b2 else '❌'} GET /health returns `status: healthy`")
        if not b2:
            runtime_pass = False

        # B3: GET /metadata returns name and description
        r = client.get("/metadata")
        b3 = False
        b3_detail = ""
        if r.status_code == 200:
            try:
                md = r.json()
                if isinstance(md.get("name"), str) and isinstance(md.get("description"), str):
                    b3 = True
                else:
                    b3_detail = f"name={type(md.get('name')).__name__}, description={type(md.get('description')).__name__}"
            except Exception:
                b3_detail = "Non-JSON response"
        elif r.status_code == 404:
            b3_detail = "Endpoint not found (404)"
        results.append(("Runtime", "GET /metadata → name + description", b3, b3_detail))
        st.write(f"  {'✅' if b3 else '❌'} GET /metadata returns `name` and `description`" + (f" — {b3_detail}" if b3_detail else ""))
        if not b3:
            runtime_pass = False

        # B4: GET /schema returns action, observation, state schemas
        r = client.get("/schema")
        b4 = False
        b4_detail = ""
        if r.status_code == 200:
            try:
                sc = r.json()
                has_action = isinstance(sc.get("action"), dict)
                has_observation = isinstance(sc.get("observation"), dict)
                has_state = isinstance(sc.get("state"), dict)
                b4 = has_action and has_observation and has_state
                if not b4:
                    b4_detail = f"action={has_action}, observation={has_observation}, state={has_state}"
            except Exception:
                b4_detail = "Non-JSON response"
        elif r.status_code == 404:
            b4_detail = "Endpoint not found (404)"
        results.append(("Runtime", "GET /schema → action, observation, state", b4, b4_detail))
        st.write(f"  {'✅' if b4 else '❌'} GET /schema returns action, observation, state schemas" + (f" — {b4_detail}" if b4_detail else ""))
        if not b4:
            runtime_pass = False

        # B5: Mode endpoint consistency — /reset, /step, /state in OpenAPI
        b5 = False
        try:
            oa = client.get("/openapi.json").json()
            paths = oa.get("paths", {})
            has_reset = "/reset" in paths
            has_step = "/step" in paths
            has_state = "/state" in paths
            b5 = has_reset and has_step and has_state
        except Exception:
            pass
        results.append(("Runtime", "OpenAPI has /reset, /step, /state (simulation mode)", b5, ""))
        st.write(f"  {'✅' if b5 else '❌'} OpenAPI paths include /reset, /step, /state (simulation mode)")
        if not b5:
            runtime_pass = False

        time.sleep(0.3)
        if runtime_pass:
            status.update(label="B. Runtime Endpoint Checks ✅ PASSED", state="complete", expanded=False)
        else:
            status.update(label="B. Runtime Endpoint Checks ❌ FAILED", state="error", expanded=False)

    # =================================================================
    # SECTION C: Hackathon Round 1 — Programmatic Checks
    # These are hackathon-specific (not in the openenv CLI)
    # =================================================================
    with st.status("C. Round 1 Programmatic Checks...", expanded=True) as status:
        r1_pass = True

        # C1: POST /reset works
        r = client.post("/reset", json={"seed": 42, "options": {"difficulty": "easy"}})
        c1 = r.status_code == 200 and "observation" in r.json()
        results.append(("Round1", "POST /reset returns observation", c1, ""))
        st.write(f"  {'✅' if c1 else '❌'} POST /reset returns 200 with observation")
        if not c1:
            r1_pass = False

        # C2: POST /step works and returns reward + done
        r = client.post("/step", json={"decision": "allow", "reason": "test"})
        c2 = False
        if r.status_code == 200:
            j = r.json()
            c2 = "reward" in j and "done" in j and "observation" in j
        results.append(("Round1", "POST /step returns reward/done/observation", c2, ""))
        st.write(f"  {'✅' if c2 else '❌'} POST /step returns reward, done, observation")
        if not c2:
            r1_pass = False

        # C3: Episode can complete and score is in [0.0, 1.0]
        client.post("/reset", json={"seed": 42, "options": {"difficulty": "easy"}})
        done = False
        last_r = None
        while not done:
            last_r = client.post("/step", json={"decision": "allow", "reason": "test"})
            done = last_r.json().get("done", True)
        c3 = False
        if last_r:
            score = last_r.json().get("observation", {}).get("episode_score", -1)
            c3 = isinstance(score, (int, float)) and 0.0 <= score <= 1.0
        results.append(("Round1", "Episode score in [0.0, 1.0]", c3, f"score={score}" if last_r else ""))
        st.write(f"  {'✅' if c3 else '❌'} Episode completes with score in [0.0, 1.0]" + (f" (score={score:.4f})" if c3 else ""))
        if not c3:
            r1_pass = False

        # C4: POST /step after done → HTTP 400 (strict RL compliance)
        r = client.post("/step", json={"decision": "allow", "reason": "post-done"})
        c4 = r.status_code >= 400
        results.append(("Round1", "POST /step after done → 400", c4, f"Got {r.status_code}"))
        st.write(f"  {'✅' if c4 else '❌'} POST /step after episode done returns HTTP {r.status_code}")
        if not c4:
            r1_pass = False

        # C5: Invalid task_id → HTTP 400
        r = client.post("/reset", json={"options": {"task_id": "nonexistent_xyz_999"}})
        c5 = r.status_code >= 400
        results.append(("Round1", "Invalid task_id → 400", c5, f"Got {r.status_code}"))
        st.write(f"  {'✅' if c5 else '❌'} Invalid task_id returns HTTP {r.status_code}")
        if not c5:
            r1_pass = False

        # C6: Pydantic rejects malformed JSON
        r = client.post("/step", json={"UNKNOWN_FIELD": "garbage"})
        c6 = r.status_code == 422
        results.append(("Round1", "Malformed step JSON → 422", c6, f"Got {r.status_code}"))
        st.write(f"  {'✅' if c6 else '❌'} Malformed JSON rejected with 422")
        if not c6:
            r1_pass = False

        # C7: GET /baseline returns scores
        r = client.get("/baseline")
        c7 = False
        c7_detail = ""
        if r.status_code == 200:
            j = r.json()
            c7 = "average_score" in j and "episodes" in j
            c7_detail = f"avg={j.get('average_score', '?')}, episodes={len(j.get('episodes', []))}"
        results.append(("Round1", "GET /baseline runs and returns scores", c7, c7_detail))
        st.write(f"  {'✅' if c7 else '❌'} GET /baseline returns scores" + (f" ({c7_detail})" if c7 else ""))
        if not c7:
            r1_pass = False

        # C8: 3+ tasks across multiple difficulties
        r = client.get("/tasks")
        c8 = False
        c8_detail = ""
        if r.status_code == 200:
            j = r.json()
            total = j.get("total_tasks", 0)
            diffs = j.get("tasks_by_difficulty", {})
            c8 = total >= 3 and len(diffs) >= 2
            c8_detail = f"total={total}, difficulties={list(diffs.keys())}"
        results.append(("Round1", "3+ tasks with difficulty range", c8, c8_detail))
        st.write(f"  {'✅' if c8 else '❌'} 3+ tasks with difficulty range" + (f" ({c8_detail})" if c8 else ""))
        if not c8:
            r1_pass = False

        # C9: Deterministic — same seed produces same task
        client.post("/reset", json={"seed": 42})
        t1 = client.post("/step", json={"decision": "allow", "reason": "x"}).json().get("observation", {}).get("task_id")
        client.post("/reset", json={"seed": 42})
        t2 = client.post("/step", json={"decision": "allow", "reason": "x"}).json().get("observation", {}).get("task_id")
        c9 = t1 == t2 and t1 is not None
        results.append(("Round1", "Deterministic with same seed", c9, f"seed=42 → {t1} == {t2}"))
        st.write(f"  {'✅' if c9 else '❌'} Deterministic: same seed → same task_id ({t1})")
        if not c9:
            r1_pass = False

        # C10: State leakage — reset produces clean state
        client.post("/reset", json={"options": {"task_id": "easy_001"}})
        client.post("/step", json={"decision": "allow", "reason": "test"})
        obs2 = client.post("/reset", json={"options": {"task_id": "easy_002"}}).json().get("observation", {})
        c10 = (
            obs2.get("current_step") == 1
            and not obs2.get("previous_decisions")
            and obs2.get("task_id") == "easy_002"
        )
        results.append(("Round1", "No state leakage between episodes", c10, ""))
        st.write(f"  {'✅' if c10 else '❌'} No state leakage between episodes")
        if not c10:
            r1_pass = False

        time.sleep(0.3)
        if r1_pass:
            status.update(label="C. Round 1 Programmatic Checks ✅ PASSED", state="complete", expanded=False)
        else:
            status.update(label="C. Round 1 Programmatic Checks ❌ FAILED", state="error", expanded=False)

    # =================================================================
    # SECTION D: Docker Build
    # =================================================================
    with st.status("D. Docker Build...", expanded=True) as status:
        d1 = False
        d1_detail = ""
        try:
            dres = subprocess.run(
                ["docker", "build", "-t", "openenv_test", ".", "-q"],
                capture_output=True, text=True, timeout=300
            )
            d1 = dres.returncode == 0
            if not d1:
                d1_detail = dres.stderr[-200:] if dres.stderr else "Unknown error"
        except FileNotFoundError:
            d1_detail = "Docker not installed or not running"
        except subprocess.TimeoutExpired:
            d1_detail = "Docker build timed out (>5min)"

        results.append(("Docker", "docker build succeeds", d1, d1_detail))
        st.write(f"  {'✅' if d1 else '⚠️'} docker build" + (f" — {d1_detail}" if d1_detail else ""))

        # Check Dockerfile for USER (non-root for HF Spaces)
        d2 = False
        if os.path.exists("Dockerfile"):
            with open("Dockerfile") as f:
                d2 = "USER " in f.read()
        results.append(("Docker", "Dockerfile has USER (non-root)", d2, ""))
        st.write(f"  {'✅' if d2 else '❌'} Dockerfile drops to non-root USER")

        if d1 and d2:
            status.update(label="D. Docker Build ✅ PASSED", state="complete", expanded=False)
        elif d1_detail:
            status.update(label="D. Docker Build ⚠️ SKIPPED", state="complete", expanded=False)
        else:
            status.update(label="D. Docker Build ❌ FAILED", state="error", expanded=False)

    # =================================================================
    # SUMMARY
    # =================================================================
    st.markdown("---")

    passed = sum(1 for _, _, p, _ in results if p)
    total = len(results)
    failed = [(s, n, d) for s, n, p, d in results if not p]

    col1, col2 = st.columns(2)
    col1.metric("Checks Passed", f"{passed}/{total}")
    col2.metric("Checks Failed", f"{len(failed)}")

    if len(failed) == 0:
        st.success("✅ **ALL CHECKS PASSED** — Ready for hackathon submission.")
        st.balloons()
    elif len(failed) <= 2:
        st.warning(f"⚠️ **{len(failed)} check(s) failed** — Minor issues to fix.")
    else:
        st.error(f"❌ **{len(failed)} checks failed** — Needs attention before submission.")

    if failed:
        st.markdown("### Failed Checks")
        for section, name, detail in failed:
            st.write(f"- **[{section}]** {name}" + (f" — `{detail}`" if detail else ""))


if st.button("▶ RUN VALIDATOR", type="primary"):
    perform_scan()
