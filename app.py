import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import nnodely as nn

# -----------------------------------------------------------
# App Configuration
# -----------------------------------------------------------
st.set_page_config(
    page_title="nnodely Demo",
    layout="wide",
)

# -----------------------------------------------------------
# Title Component
# -----------------------------------------------------------
st.image("assets/nnodely_logo.png", width=250)
st.title("nnodely Demo: Structured Neural Networks for Mechanical Systems")
st.markdown("---")

# -----------------------------------------------------------
# 1. Modeling Component
# -----------------------------------------------------------
st.header("1. Modeling Mechanical Systems")

st.subheader("Inputs")
inputs = st.text_area("Enter system inputs (comma separated)", "force, velocity")

st.subheader("Outputs")
outputs = st.text_area("Enter system outputs (comma separated)", "displacement")

st.subheader("Relations")
relations = st.multiselect(
    "Select relations",
    ["Newton's Law", "Hooke's Law", "Damping Relation", "Energy Conservation"],
    default=["Newton's Law"]
)

if st.button("Save Model Specification"):
    st.success("Model specification saved ✅")
    st.json({
        "inputs": [i.strip() for i in inputs.split(",") if i.strip()],
        "outputs": [o.strip() for o in outputs.split(",") if o.strip()],
        "relations": relations
    })

st.markdown("---")

# -----------------------------------------------------------
# 2. Structure Component
# -----------------------------------------------------------
st.header("2. Neural Network Structure")

# Placeholder: simple matplotlib visualization
fig, ax = plt.subplots()
ax.set_title("Network Structure (placeholder)")
circle1 = plt.Circle((0.3, 0.5), 0.05, color="blue")
circle2 = plt.Circle((0.5, 0.5), 0.05, color="green")
circle3 = plt.Circle((0.7, 0.5), 0.05, color="red")
ax.add_artist(circle1)
ax.add_artist(circle2)
ax.add_artist(circle3)
ax.plot([0.35, 0.45], [0.5, 0.5], "k-")
ax.plot([0.55, 0.65], [0.5, 0.5], "k-")
ax.axis("off")
st.pyplot(fig)

st.markdown("---")

# -----------------------------------------------------------
# 3. Training Component
# -----------------------------------------------------------
st.header("3. Train the Model")

col1, col2 = st.columns(2)

with col1:
    epochs = st.number_input("Epochs", min_value=1, max_value=5000, value=100)
    lr = st.number_input("Learning Rate", min_value=1e-6, max_value=1.0, value=0.001, format="%.6f")

with col2:
    batch_size = st.number_input("Batch Size", min_value=1, max_value=1024, value=32)
    optimizer = st.selectbox("Optimizer", ["SGD", "Adam", "RMSprop"])

if st.button("Start Training"):
    st.info("Training started... (placeholder)")
    # Placeholder: insert nnodely training call here
    # results = nnodely.train_model(...)
    # You can add a progress bar to simulate training

st.markdown("---")

# -----------------------------------------------------------
# 4. Visualization & Export Component
# -----------------------------------------------------------
st.header("4. Results, Visualization & Export")

# Placeholder: fake training loss curve
epochs_arr = np.arange(1, 101)
loss = np.exp(-epochs_arr / 20) + np.random.normal(0, 0.02, size=len(epochs_arr))

fig, ax = plt.subplots()
ax.plot(epochs_arr, loss, label="Training Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
st.pyplot(fig)

if st.button("Export Model"):
    st.success("Model exported successfully! ✅")
    # Placeholder: insert nnodely export function here
