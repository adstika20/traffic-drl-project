import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from environment.traffic_rl_environment import SingleIntersectionTrafficEnv, EmpiricalDemandGenerator

st.set_page_config(page_title="Adaptive Traffic Signal Simulation", page_icon="🚦", layout="wide")
st.title("🚦 Adaptive Traffic Signal Simulation")
st.caption("Perbandingan kebijakan DQN, PPO, dan Fixed-Time pada satu persimpangan abstrak dua fase.")

MODEL_DIR = ROOT / "models"
DQN_PATH = MODEL_DIR / "dqn_traffic_signal_20000.zip"
PPO_PATH = MODEL_DIR / "ppo_traffic_signal_20000.zip"
DATA_PATH = ROOT / "data" / "smart_traffic_management_dataset.csv"

with st.sidebar:
    st.header("Pengaturan simulasi")
    policy_label = st.selectbox("Kebijakan sinyal", ["DQN", "PPO", "Fixed-Time"])
    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=101, step=1)
    fixed_green = st.slider("Durasi green Fixed-Time (kelipatan 10 detik)", 1, 6, 3)
    st.caption("Environment menggunakan minimum green 30 detik. Setiap langkah simulasi mewakili 10 detik.")
    uploaded_data = st.file_uploader("Dataset traffic_volume (CSV, opsional)", type=["csv"])

st.subheader("Status berkas")
c1, c2, c3 = st.columns(3)
c1.metric("Model DQN", "Tersedia" if DQN_PATH.exists() else "Belum ada")
c2.metric("Model PPO", "Tersedia" if PPO_PATH.exists() else "Belum ada")
c3.metric("Dataset empiris", "Tersedia" if (DATA_PATH.exists() or uploaded_data is not None) else "Belum ada")

demand_values = None
if uploaded_data is not None:
    try:
        df_upload = pd.read_csv(uploaded_data)
        if "traffic_volume" not in df_upload.columns:
            st.error("CSV harus memiliki kolom bernama traffic_volume.")
            st.stop()
        demand_values = df_upload["traffic_volume"].dropna().to_numpy()
    except Exception as e:
        st.error(f"Gagal membaca dataset: {e}")
        st.stop()
elif DATA_PATH.exists():
    df_data = pd.read_csv(DATA_PATH)
    if "traffic_volume" in df_data.columns:
        demand_values = df_data["traffic_volume"].dropna().to_numpy()

if demand_values is None or len(demand_values) == 0:
    st.warning("Dataset empiris belum tersedia. Simulasi menggunakan demand sintetis sementara. Hasil ini bukan replikasi distribusi dataset penelitian.")
    rng_tmp = np.random.default_rng(int(seed))
    demand_values = rng_tmp.integers(50, 999, size=2000)

if st.button("▶ Jalankan simulasi", type="primary", use_container_width=True):
    try:
        from stable_baselines3 import DQN, PPO

        env = SingleIntersectionTrafficEnv(
            demand_generator=EmpiricalDemandGenerator(demand_values=demand_values),
            min_green_seconds=30,
            seed=int(seed)
        )
        obs, info = env.reset(seed=int(seed))

        model = None
        if policy_label == "DQN":
            if not DQN_PATH.exists():
                st.error(f"Model tidak ditemukan: {DQN_PATH}")
                st.stop()
            model = DQN.load(str(DQN_PATH), device="cpu")
        elif policy_label == "PPO":
            if not PPO_PATH.exists():
                st.error(f"Model tidak ditemukan: {PPO_PATH}")
                st.stop()
            model = PPO.load(str(PPO_PATH), device="cpu")

        rows = []
        terminated = truncated = False
        while not (terminated or truncated):
            if policy_label == "Fixed-Time":
                # Bergantian setelah fixed_green langkah green yang memenuhi minimum green.
                action = 1 - int(obs[2]) if env.green_elapsed_seconds >= fixed_green * 10 else int(obs[2])
            else:
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)

            obs, reward, terminated, truncated, info = env.step(action)
            rows.append({
                "step": len(rows) + 1,
                "time_min": len(rows) * 10 / 60,
                "queue_NS": info["queue_NS"],
                "queue_EW": info["queue_EW"],
                "total_queue": info["queue_NS"] + info["queue_EW"],
                "phase": "NS" if info["current_phase"] == 0 else "EW",
                "arrivals": info["arrivals"],
                "departures": info["departures"],
                "reward": reward,
                "switch_blocked": info["switch_blocked"]
            })

        result = pd.DataFrame(rows)
        metrics = info["episode_metrics"]
        a, b, c, d = st.columns(4)
        a.metric("Rata-rata antrean", f'{metrics["avg_queue"]:.2f} kendaraan')
        b.metric("Throughput", f'{metrics["throughput"]} kendaraan')
        c.metric("Pergantian fase", f'{metrics["phase_switches"]}')
        d.metric("Antrean akhir", f'{metrics["final_queue"]} kendaraan')

        st.subheader("Dinamika antrean")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(result["time_min"], result["queue_NS"], label="Antrean North–South")
        ax.plot(result["time_min"], result["queue_EW"], label="Antrean East–West")
        ax.plot(result["time_min"], result["total_queue"], label="Total antrean", linewidth=2)
        ax.set_xlabel("Waktu simulasi (menit)")
        ax.set_ylabel("Jumlah kendaraan")
        ax.grid(alpha=0.25)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Status fase sinyal")
        st.line_chart(result.set_index("time_min")["phase"].map({"NS": 0, "EW": 1}))
        st.caption("0 = fase North–South; 1 = fase East–West.")

        with st.expander("Detail metrik episode"):
            st.json(metrics)
        with st.expander("Log setiap langkah"):
            st.dataframe(result, use_container_width=True)
            st.download_button(
                "Unduh log simulasi CSV",
                data=result.to_csv(index=False).encode("utf-8"),
                file_name=f"traffic_simulation_{policy_label.lower()}_seed_{seed}.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.exception(e)
else:
    st.info("Atur kebijakan dan seed pada panel kiri, kemudian klik **Jalankan simulasi**.")
    st.markdown("""
    **Cakupan simulasi**
    - Satu persimpangan abstrak dengan dua fase: North–South dan East–West.
    - DQN dan PPO menggunakan model hasil pelatihan yang disimpan pada folder `models/`.
    - Fixed-Time menjadi baseline pembanding.
    - Satu episode terdiri dari 180 langkah; satu langkah mewakili 10 detik.
    """)
