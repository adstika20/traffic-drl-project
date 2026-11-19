import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from environment.traffic_rl_environment import SingleIntersectionTrafficEnv, EmpiricalDemandGenerator

st.set_page_config(page_title="Adaptive Traffic Signal Control", page_icon="🚦", layout="wide")
st.title("🚦 Adaptive Traffic Signal Control")
st.caption("Visualisasi 2D persimpangan dua fase — DQN vs PPO vs Fixed-Time")

MODEL_DIR = ROOT / "models"
DQN_PATH = MODEL_DIR / "dqn_traffic_signal_20000.zip"
PPO_PATH = MODEL_DIR / "ppo_traffic_signal_20000.zip"
DATA_PATH = ROOT / "data" / "smart_traffic_management_dataset.csv"

with st.sidebar:
    st.header("Pengaturan simulasi")
    policy = st.selectbox("Kebijakan sinyal", ["DQN", "PPO", "Fixed-Time"])
    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=101, step=1)
    fixed_green_steps = st.slider("Fixed-Time: durasi green (langkah 10 detik)", 3, 9, 4)
    animation_speed = st.slider("Kecepatan animasi", 0.03, 0.30, 0.10, 0.01)
    st.caption("Satu langkah simulasi = 10 detik. Kecepatan animasi hanya mengatur jeda tampilan.")
    uploaded = st.file_uploader("Dataset traffic_volume (CSV)", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
    if "traffic_volume" not in df.columns:
        st.error("Dataset harus memiliki kolom traffic_volume.")
        st.stop()
    demand_values = df["traffic_volume"].dropna().to_numpy()
elif DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)
    demand_values = df["traffic_volume"].dropna().to_numpy()
else:
    st.error("Dataset empiris tidak ditemukan. Letakkan smart_traffic_management_dataset.csv di folder data/.")
    st.stop()

if len(demand_values) == 0:
    st.error("Kolom traffic_volume tidak memiliki data valid.")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Model DQN", "Siap" if DQN_PATH.exists() else "Tidak ditemukan")
c2.metric("Model PPO", "Siap" if PPO_PATH.exists() else "Tidak ditemukan")
c3.metric("Dataset empiris", f"{len(demand_values):,} baris")

def draw_intersection(ax, q_ns, q_ew, phase, step, arrivals, departures):
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect("equal")
    ax.set_facecolor("#dce8d5"); ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)
    road = "#454b54"
    ax.add_patch(Rectangle((38, 0), 24, 100, color=road, zorder=1))
    ax.add_patch(Rectangle((0, 38), 100, 24, color=road, zorder=1))
    for y in range(2, 100, 8):
        ax.plot([50, 50], [y, y+4], color="white", lw=1.2, alpha=.75, zorder=2)
    for x in range(2, 100, 8):
        ax.plot([x, x+4], [50, 50], color="white", lw=1.2, alpha=.75, zorder=2)
    ax.plot([38, 62], [68, 68], color="white", lw=2.5, zorder=3)
    ax.plot([38, 62], [32, 32], color="white", lw=2.5, zorder=3)
    ax.plot([32, 32], [38, 62], color="white", lw=2.5, zorder=3)
    ax.plot([68, 68], [38, 62], color="white", lw=2.5, zorder=3)
    ns_color = "#20c766" if phase == 0 else "#ef4444"
    ew_color = "#20c766" if phase == 1 else "#ef4444"
    ax.add_patch(Circle((74, 74), 4, color=ns_color, ec="white", lw=1.2, zorder=8))
    ax.add_patch(Circle((26, 74), 4, color=ew_color, ec="white", lw=1.2, zorder=8))
    ax.text(74, 82, "NS", ha="center", fontsize=9, weight="bold")
    ax.text(26, 82, "EW", ha="center", fontsize=9, weight="bold")
    colors = ["#f59e0b", "#38bdf8", "#a78bfa", "#fb7185", "#f8fafc"]
    def car(x, y, horizontal=False, idx=0):
        w, h = (7, 3.8) if horizontal else (3.8, 7)
        ax.add_patch(Rectangle((x-w/2, y-h/2), w, h, facecolor=colors[idx % len(colors)],
                               edgecolor="#111827", lw=.7, zorder=6))
    for i in range(min(int(np.ceil(q_ns/2)), 12)): car(44, 91-i*6, idx=i)
    for i in range(min(int(q_ns//2), 12)): car(56, 9+i*6, idx=i+1)
    for i in range(min(int(np.ceil(q_ew/2)), 12)): car(9+i*6, 44, horizontal=True, idx=i+2)
    for i in range(min(int(q_ew//2), 12)): car(91-i*6, 56, horizontal=True, idx=i+3)
    ax.text(50, 97, f"Utara | antrean NS: {int(q_ns)}", ha="center", va="top",
            color="#111827", fontsize=9, weight="bold")
    ax.text(50, 3, "Selatan", ha="center", va="bottom", color="white", fontsize=9)
    ax.text(3, 50, "Barat", ha="left", va="bottom", color="white", fontsize=9, rotation=90)
    ax.text(97, 50, "Timur", ha="right", va="top", color="white", fontsize=9, rotation=270)
    ax.text(50, 29, f"Antrean EW: {int(q_ew)}", ha="center", va="center",
            color="white", fontsize=9, weight="bold")
    ax.set_title(f"Persimpangan 2D • Langkah {step}/180 • Kedatangan: {arrivals} • Terlayani: {departures}",
                 fontsize=10, pad=8)

if st.button("▶ Mulai simulasi kendaraan", type="primary", use_container_width=True):
    try:
        from stable_baselines3 import DQN, PPO
        if policy == "DQN":
            if not DQN_PATH.exists(): st.error("File model DQN tidak ditemukan."); st.stop()
            model = DQN.load(str(DQN_PATH), device="cpu")
        elif policy == "PPO":
            if not PPO_PATH.exists(): st.error("File model PPO tidak ditemukan."); st.stop()
            model = PPO.load(str(PPO_PATH), device="cpu")
        else:
            model = None
        generator = EmpiricalDemandGenerator(demand_values=demand_values, profile_length=180, seed=int(seed))
        env = SingleIntersectionTrafficEnv(demand_generator=generator, min_green_seconds=30, seed=int(seed))
        obs, _ = env.reset(seed=int(seed))
        st.subheader(f"Simulasi berjalan: {policy}")
        signal_col, step_col, queue_col = st.columns(3)
        signal_view, step_view, queue_view = signal_col.empty(), step_col.empty(), queue_col.empty()
        scene = st.empty()
        progress = st.progress(0)
        terminated = truncated = False
        logs = []
        while not (terminated or truncated):
            if policy == "Fixed-Time":
                action = (1-env.current_phase if env.green_elapsed_seconds >= fixed_green_steps*10 else env.current_phase)
            else:
                action, _ = model.predict(obs, deterministic=True); action = int(action)
            obs, reward, terminated, truncated, info = env.step(action)
            step = len(logs)+1
            q_ns, q_ew, phase = info["queue_NS"], info["queue_EW"], info["current_phase"]
            signal_view.metric("Fase aktif", "North–South 🟢" if phase == 0 else "East–West 🟢")
            step_view.metric("Waktu simulasi", f"{step*10} detik")
            queue_view.metric("Total antrean", f"{q_ns+q_ew} kendaraan")
            fig, ax = plt.subplots(figsize=(7, 6))
            draw_intersection(ax, q_ns, q_ew, phase, step, info["arrivals"], info["departures"])
            scene.pyplot(fig, use_container_width=False); plt.close(fig)
            progress.progress(step/180)
            logs.append({"step":step,"queue_NS":q_ns,"queue_EW":q_ew,"total_queue":q_ns+q_ew,
                         "phase":"NS" if phase==0 else "EW","arrivals":info["arrivals"],
                         "departures":info["departures"],"reward":reward,
                         "switch_blocked":info["switch_blocked"]})
            time.sleep(animation_speed)
        result = pd.DataFrame(logs); metrics = info["episode_metrics"]
        st.success("Simulasi selesai.")
        a,b,c,d = st.columns(4)
        a.metric("Rata-rata antrean", f'{metrics["avg_queue"]:.2f}')
        b.metric("Throughput", f'{metrics["throughput"]} kendaraan')
        c.metric("Pergantian fase", metrics["phase_switches"])
        d.metric("Antrean akhir", f'{metrics["final_queue"]} kendaraan')
        st.subheader("Riwayat antrean")
        st.line_chart(result.set_index("step")[["queue_NS","queue_EW","total_queue"]])
        with st.expander("Log simulasi"):
            st.dataframe(result, use_container_width=True)
            st.download_button("Unduh log CSV", result.to_csv(index=False).encode(),
                               f"{policy.lower()}_traffic_animation_log.csv","text/csv")
    except Exception as e:
        st.exception(e)
else:
    st.info("Pilih kebijakan, lalu klik **Mulai simulasi kendaraan**.")
    st.markdown("""
    **Catatan interpretasi:** ikon mobil adalah representasi visual panjang antrean pada environment berbasis queue.
    Pembagian antrean NS/EW ke sisi pendekatan jalan hanya untuk tampilan, bukan data arah kendaraan.
    Ini bukan simulator mikroskopik seperti SUMO; DQN/PPO tetap beroperasi pada environment antrean abstrak.
    """)
