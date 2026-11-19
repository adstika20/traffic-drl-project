# Traffic DRL Streamlit App

A single-page Streamlit application for running one episode of an abstract two-phase intersection simulation using DQN, PPO, or Fixed-Time.

## Required project files
- `app.py`
- `environment/traffic_rl_environment.py`
- `models/dqn_traffic_signal_20000.zip`
- `models/ppo_traffic_signal_20000.zip`
- `data/smart_traffic_management_dataset.csv` (recommended)

The two model ZIP files and original CSV are not included in this starter archive. Copy them from your existing Google Drive project ZIP / dataset source into the matching folders.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Important methodological note
If the empirical dataset is absent, the app uses a temporary synthetic array between 50 and 998 only to make the interface runnable. This is NOT the empirical demand distribution used in the experiment. Upload the original CSV through the sidebar for empirical-demand simulation.

The environment is abstract and queue-based, not a microscopic traffic simulator or a safety-certified controller.
