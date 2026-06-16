import base64
import inspect
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from ai_coach import process_video

st.set_page_config(page_title="AI Golf Coach", layout="wide")
st.title("AI Golf Coach")
st.caption("Upload two MP4 files and compare analyzed outputs side-by-side.")

def run_analysis(
    uploaded_file,
    side_label: str,
    model: str,
    analysis_mode: str,
) -> dict:
    work_dir = Path("media")
    work_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(uploaded_file.name).name
    input_video_path = work_dir / f"{side_label}_{safe_name}"
    output_video_path = work_dir / f"{input_video_path.stem}_output.mp4"
    csv_output_path = work_dir / f"{input_video_path.stem}_metrics.csv"

    input_video_path.write_bytes(uploaded_file.getbuffer())

    base_kwargs = {
        "input_video_path": str(input_video_path),
        "output_video_path": str(output_video_path),
        "csv_output_path": str(csv_output_path),
        "model": model,
        "model_path": None,
        "show_preview": False,
    }

    params = inspect.signature(process_video).parameters
    if "analysis_mode" in params:
        summary = process_video(**base_kwargs, analysis_mode=analysis_mode)
    else:
        raise RuntimeError(
            "Loaded ai_coach.process_video does not support analysis_mode. "
            "Restart Streamlit to load the latest ai_coach.py."
        )

    return {
        "source_name": safe_name,
        "output_video_path": str(output_video_path),
        "csv_output_path": str(csv_output_path),
        "summary": summary,
    }


def render_result_panel(title: str, result: dict, key_prefix: str) -> None:
    output_video_path = Path(result["output_video_path"])
    csv_output_path = Path(result["csv_output_path"])
    summary = result["summary"]

    st.markdown(f"### {title}: {result['source_name']}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Frames", f"{summary['total_frames']}")
    c2.metric("Head Moving Ratio", f"{summary['moving_ratio_percent']:.2f}%")
    c3.metric("Shoulder Tilt Deviation", f"{summary['max_shoulder_tilt_deviation_deg']:.2f} deg")

    c4, _ = st.columns(2)
    c4.metric("Max Lateral Deviation", f"{summary['max_lateral_deviation_px']} px")
    st.caption(f"Analysis mode: {summary.get('analysis_mode', 'unknown')}")

    st.caption(f"Output codec: {summary.get('video_codec', 'unknown')}")

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            label="Download Processed Video",
            data=output_video_path.read_bytes(),
            file_name=output_video_path.name,
            mime="video/mp4",
            use_container_width=True,
            key=f"{key_prefix}_video_dl",
        )
    with d2:
        st.download_button(
            label="Download Metrics CSV",
            data=csv_output_path.read_bytes(),
            file_name=csv_output_path.name,
            mime="text/csv",
            use_container_width=True,
            key=f"{key_prefix}_csv_dl",
        )


def render_synced_dual_player(left_video_path: Path, right_video_path: Path) -> None:
    left_b64 = base64.b64encode(left_video_path.read_bytes()).decode("utf-8")
    right_b64 = base64.b64encode(right_video_path.read_bytes()).decode("utf-8")

    html = f"""
    <div style="padding:8px;border:1px solid #ddd;border-radius:10px;background:#fafafa;box-sizing:border-box;">
        <div style="display:flex;flex-wrap:wrap;gap:10px;margin-bottom:10px;justify-content:center;">
            <button id="syncBtn" style="padding:8px 14px;cursor:pointer;">Sync</button>
            <button id="startBtn" style="padding:8px 14px;cursor:pointer;">Start</button>
            <button id="stopBtn" style="padding:8px 14px;cursor:pointer;">Pause</button>
            <button id="repeatBtn" style="padding:8px 14px;cursor:pointer;">Repeat: Off</button>
            <button id="speed25Btn" style="padding:8px 14px;cursor:pointer;">25%</button>
            <button id="speed50Btn" style="padding:8px 14px;cursor:pointer;">50%</button>
            <button id="speed75Btn" style="padding:8px 14px;cursor:pointer;">75%</button>
            <button id="speed100Btn" style="padding:8px 14px;cursor:pointer;">100%</button>
        </div>

        <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
            <div>
                <div style="font:600 13px sans-serif;margin-bottom:4px;">Left Output</div>
                <video id="leftVideo" style="width:100%;height:auto;display:block;" preload="metadata" playsinline>
                    <source src="data:video/mp4;base64,{left_b64}" type="video/mp4" />
                </video>
            </div>
            <div>
                <div style="font:600 13px sans-serif;margin-bottom:4px;">Right Output</div>
                <video id="rightVideo" style="width:100%;height:auto;display:block;" preload="metadata" playsinline>
                    <source src="data:video/mp4;base64,{right_b64}" type="video/mp4" />
                </video>
            </div>
        </div>
    </div>

    <script>
        const left = document.getElementById('leftVideo');
        const right = document.getElementById('rightVideo');
        const syncBtn = document.getElementById('syncBtn');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const repeatBtn = document.getElementById('repeatBtn');
        const speed25Btn = document.getElementById('speed25Btn');
        const speed50Btn = document.getElementById('speed50Btn');
        const speed75Btn = document.getElementById('speed75Btn');
        const speed100Btn = document.getElementById('speed100Btn');
        let repeatEnabled = false;

        function setFrameHeight() {{
            const h = Math.max(document.body.scrollHeight, document.documentElement.scrollHeight);
            window.parent.postMessage({{
                isStreamlitMessage: true,
                type: 'streamlit:setFrameHeight',
                height: h
            }}, '*');
        }}

        function syncOnce() {{
            if (!Number.isFinite(left.currentTime) || !Number.isFinite(right.currentTime)) return;
            const t = Math.min(left.currentTime, right.currentTime);
            left.currentTime = t;
            right.currentTime = t;
        }}

        function playBoth() {{
            syncOnce();
            left.play();
            right.play();
        }}

        function restartBoth() {{
            left.currentTime = 0;
            right.currentTime = 0;
            left.play();
            right.play();
            stopBtn.textContent = 'Pause';
        }}

        function stopBoth() {{
            left.pause();
            right.pause();
            stopBtn.textContent = 'Continue';
        }}

        function togglePauseContinue() {{
            if (left.paused && right.paused) {{
                playBoth();
                stopBtn.textContent = 'Pause';
                return;
            }}
            stopBoth();
        }}

        function setSpeed(rate) {{
            left.playbackRate = rate;
            right.playbackRate = rate;
        }}

        syncBtn.addEventListener('click', syncOnce);
        startBtn.addEventListener('click', restartBoth);
        stopBtn.addEventListener('click', togglePauseContinue);
        repeatBtn.addEventListener('click', () => {{
            repeatEnabled = !repeatEnabled;
            repeatBtn.textContent = repeatEnabled ? 'Repeat: On' : 'Repeat: Off';
        }});
        speed25Btn.addEventListener('click', () => setSpeed(0.25));
        speed50Btn.addEventListener('click', () => setSpeed(0.5));
        speed75Btn.addEventListener('click', () => setSpeed(0.75));
        speed100Btn.addEventListener('click', () => setSpeed(1.0));

        function onVideoEnded() {{
            if (!repeatEnabled) return;
            left.currentTime = 0;
            right.currentTime = 0;
            playBoth();
            stopBtn.textContent = 'Pause';
        }}

        left.addEventListener('ended', onVideoEnded);
        right.addEventListener('ended', onVideoEnded);
        left.addEventListener('loadedmetadata', setFrameHeight);
        right.addEventListener('loadedmetadata', setFrameHeight);
        window.addEventListener('resize', setFrameHeight);
        setTimeout(setFrameHeight, 50);

        setInterval(() => {{
            if (left.paused || right.paused) return;
            const drift = Math.abs(left.currentTime - right.currentTime);
            if (drift > 0.08) {{
                syncOnce();
            }}
        }}, 300);
    </script>
    """

    components.html(html, height=900, scrolling=False)


if "compare_ready" not in st.session_state:
    st.session_state.compare_ready = False
if "left_result" not in st.session_state:
    st.session_state.left_result = None
if "right_result" not in st.session_state:
    st.session_state.right_result = None

left_col, right_col = st.columns(2)
with left_col:
    left_uploaded = st.file_uploader("Left video (e.g. 01.mp4)", type=["mp4"], key="left_upload")
with right_col:
    right_uploaded = st.file_uploader("Right video (e.g. goodswing.mp4 or 02.mp4)", type=["mp4"], key="right_upload")

control_col1, control_col2 = st.columns(2)
with control_col1:
    model = st.selectbox("Pose model", ["lite", "full", "heavy"], index=1)
with control_col2:
    play_clicked = st.button("Play Both Videos without Analysis", type="secondary", use_container_width=True)
    analysis_clicked = st.button("Run Selected Analyses", type="primary", use_container_width=True)

analysis_label = st.radio(
    "Analysis option (single trace for clarity)",
    ["head_positon", "v_arm", "swing_arc"],
    horizontal=True,
)

analysis_mode_map = {
    "head_positon": "head_position",
    "v_arm": "v_arm",
    "swing_arc": "swing_arc",
}
analysis_mode = analysis_mode_map[analysis_label]

if play_clicked:
    if left_uploaded is None or right_uploaded is None:
        st.error("Please upload both left and right MP4 files.")
    else:
        st.session_state.compare_ready = True
        st.session_state.left_result = {
            "output_video_path": str(Path("media") / f"left_{left_uploaded.name}"), 
            "csv_output_path": str(Path("media") / f"left_{left_uploaded.name}_metrics.csv"),
        }
        st.session_state.right_result = {
            "output_video_path": str(Path("media") / f"right_{right_uploaded.name}"),
            "csv_output_path": str(Path("media") / f"right_{right_uploaded.name}_metrics.csv"),
        }
        left_video_path = Path(st.session_state.left_result["output_video_path"])
        right_video_path = Path(st.session_state.right_result["output_video_path"])
        render_synced_dual_player(left_video_path, right_video_path)
        
if analysis_clicked:
    if left_uploaded is None or right_uploaded is None:
        st.error("Please upload both left and right MP4 files.")
    else:
        with st.spinner("Processing both videos..."):
            left_result = run_analysis(
                left_uploaded,
                "left",
                model,
                analysis_mode,
            )
            right_result = run_analysis(
                right_uploaded,
                "right",
                model,
                analysis_mode,
            )

        st.session_state.compare_ready = True
        st.session_state.left_result = left_result
        st.session_state.right_result = right_result

if st.session_state.compare_ready:
    left_result = st.session_state.left_result
    right_result = st.session_state.right_result

    if not left_result or not right_result:
        st.session_state.compare_ready = False
        st.error("Comparison state is invalid. Please run analysis again.")
    else:
        left_output = Path(left_result["output_video_path"])
        right_output = Path(right_result["output_video_path"])
        left_csv = Path(left_result["csv_output_path"])
        right_csv = Path(right_result["csv_output_path"])

        if not all([left_output.exists(), right_output.exists(), left_csv.exists(), right_csv.exists()]):
            st.session_state.compare_ready = False
            st.error("One or more output files are missing. Please run analysis again.")
        else:
            st.success("Side-by-side analysis complete")

            st.subheader("Synchronized Comparison Player")
            render_synced_dual_player(left_output, right_output)

            p_left, p_right = st.columns(2)
            with p_left:
                render_result_panel("Left", left_result, "left")
            with p_right:
                render_result_panel("Right", right_result, "right")
