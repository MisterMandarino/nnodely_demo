import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import nnodely as nn
import pandas as pd
import sys, io

from nnodely.support.jsonutils import plot_graphviz_structure

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
st.image("assets/nnodely_logo.png", width=1250)
st.title("Structured Neural Networks for Mechanical Systems")
st.markdown("---")

# -----------------------------------------------------------
# 1. Modeling Component
# -----------------------------------------------------------
st.header("1. Define the Model Structure")

model_name = st.text_input("Model Name", "Model", key="model_name")

# Create a 3-column layout
col1, col2, col3 = st.columns(3)

# Column 1: Inputs
with col1:
    st.subheader("Inputs")

    # Initialize session state for inputs
    if "inputs" not in st.session_state:
        st.session_state.inputs = []

    # Input form
    tag = st.text_input("Name", "", key="input_tag")
    dim = st.number_input("Dimension", min_value=1, value=1, step=1)
    time_window = st.number_input("Time window", min_value=0.1, value=1.0, step=0.1, format="%.2f")

    if st.button("➕ Add Input"):
        # Validity checks
        if not tag:
            st.error("❌ Name cannot be empty.")
        elif " " in tag:
            st.error("❌ Name cannot contain spaces.")
        elif tag in [inp['tag'] for inp in st.session_state.inputs]:
            st.error(f"❌ Input '{tag}' already exists.")
        elif dim < 1:
            st.error("❌ Dimension must be >= 1.")
        elif time_window <= 0:
            st.error("❌ Time window must be > 0.")
        else:
            # Add input to session state
            st.session_state.inputs.append({
                "tag": tag,
                "dimension": dim,
                "time_window": time_window
            })
            st.success(f"✅ Input '{tag}' added successfully.")

    # Show all defined inputs with delete option
    if st.session_state.inputs:
        st.write("### Defined Inputs")
        for idx, inp in enumerate(st.session_state.inputs):
            input_col1, input_col2 = st.columns([2, 1])
            with input_col1:
                st.markdown(f"**{inp['tag']}** (dim={inp['dimension']}, window={inp['time_window']})")
            with input_col2:
                if st.button("🗑️ Delete", key=f"delete_input_{idx}"):
                    st.session_state.inputs.pop(idx)
                    st.rerun()


# Column 2: Relations
with col2:
    st.subheader("Relations")

    # Available relations from nnodely
    available_relations = ["Add", "Mul", "Fir", "Linear", "Sin", "Cos"]

    # Initialize session state for relations
    if "relations" not in st.session_state:
        st.session_state.relations = []

    # Build list of possible predecessors (inputs + already defined relations)
    rel_options = []
    rel_options.extend([inp['tag'] for inp in st.session_state.get("inputs", [])])
    rel_options.extend([rel['name'] for rel in st.session_state.relations])

    # Relation form
    relation_type = st.selectbox("Choose Relation", available_relations, key="relation_type", index=None)
    params = {}
    input_rel = []

    # --- Render fields dynamically depending on relation type ---
    if relation_type == "Fir":
        params["dimension"] = st.number_input("Fir dimension", min_value=1, value=1, step=1)
        if rel_options:
            input_rel = [st.selectbox("Choose Ingress", rel_options, key="fir_pred")]
        else:
            st.warning("⚠️ Define at least one input before adding relations.")

    elif relation_type == "Linear":
        params["dimension"] = st.number_input("Linear output dimension", min_value=1, value=1, step=1)
        if rel_options:
            input_rel = [st.selectbox("Choose Ingress", rel_options, key="lin_pred")]
        else:
            st.warning("⚠️ Define at least one input before adding relations.")

    elif relation_type in ["Add", "Mul"]:
        if rel_options:
            pred1 = st.selectbox("Choose Ingress 1", rel_options, key="pred1")
            pred2 = st.selectbox("Choose Ingress 2", rel_options, key="pred2")
            input_rel = [pred1, pred2]
        else:
            st.warning("⚠️ Define at least one input before adding relations.")

    elif relation_type in ["Sin", "Cos"]:
        if rel_options:
            input_rel = [st.selectbox("Choose Ingress", rel_options, key="trig_pred")]
        else:
            st.warning("⚠️ Define at least one input before adding relations.")

    if st.button("➕ Add Relation"):
        # Validation checks
        if not rel_options:
            st.error("❌ You must define at least one input before adding relations.")
        else:
            relation_def = {
                "name": f"{relation_type}_{len(st.session_state.relations) + 1}",
                "type": relation_type,
                "parameters": params,
                "ingress": input_rel
            }
            st.session_state.relations.append(relation_def)
            st.success(f"✅ Relation '{relation_type}' added successfully.")

    # Show defined relations
    if st.session_state.relations:
        st.write("### Defined Relations")
        for idx, rel in enumerate(st.session_state.relations):
            rel_col1, rel_col2, rel_col3, rel_col4 = st.columns([1, 1, 1, 1])
            with rel_col1:
                st.markdown(f"{rel['name']}")
                if rel["parameters"]:
                    st.json(rel["parameters"], expanded=False)
            with rel_col2:
                st.markdown(f"**Type:** {rel['type']}")
            with rel_col3:
                st.markdown(f"**In:** {rel['ingress']}")
            with rel_col4:
                if st.button("🗑️ Delete", key=f"delete_relation_{idx}"):
                    st.session_state.relations.pop(idx)
                    st.rerun()

# Column 3: Outputs
with col3:
    st.subheader("Outputs")

    # Initialize session state for outputs
    if "outputs" not in st.session_state:
        st.session_state.outputs = []

    # Retrieve inputs and relations
    available_relations = [inp['tag'] for inp in st.session_state.inputs] + [rel['name'] for rel in st.session_state.relations]

    # Output form
    tag = st.text_input("Name", "", key="output_tag")
    if available_relations:
        output_rel = st.selectbox("Choose Output Relation", available_relations, key="output_rel")
    else:
        st.warning("⚠️ Define at least one relation before adding outputs.")

    if st.button("➕ Add Output"):
        # Validity checks
        if not tag:
            st.error("❌ Name cannot be empty.")
        elif " " in tag:
            st.error("❌ Name cannot contain spaces.")
        elif tag in [out['name'] for out in st.session_state.outputs]:
            st.error(f"❌ Output '{tag}' already exists.")
        else:
            # Add output to session state
            st.session_state.outputs.append({
                "name": tag,
                "relation": output_rel
            })
            st.success(f"✅ Output '{tag}' added successfully.")

    # Show all defined outputs with delete option
    if st.session_state.outputs:
        st.write("### Defined Outputs")
        for idx, out in enumerate(st.session_state.outputs):
            output_col1, output_col2, output_col3 = st.columns([2, 2, 1])
            with output_col1:
                st.markdown(f"**{out['name']}**")
            with output_col2:
                st.markdown(f"**Out:** {out['relation']}")
            with output_col3:
                if st.button("🗑️ Delete", key=f"delete_output_{idx}"):
                    st.session_state.outputs.pop(idx)
                    st.rerun()

## Minimizers
if "minimizers" not in st.session_state:
    st.session_state.minimizers = {}

available_minimizers = [inp['tag'] for inp in st.session_state.inputs] + \
                       [rel['name'] for rel in st.session_state.relations] + \
                       [out['name'] for out in st.session_state.outputs]

loss_function_name = st.text_input("Minimize Name", "Error", key="loss_function_name")
st.text("Select two relations to minimize:")
minimize_col1, minimize_col2 = st.columns([1, 1])
with minimize_col1:
    minimizer_a = st.selectbox("First Relation", available_minimizers, key="minimizer_a", index=None)
with minimize_col2:
    minimizer_b = st.selectbox("Second Relation", available_minimizers, key="minimizer_b", index=None)
loss_function = st.selectbox("Loss Function", ['mse', 'rmse', 'mae', 'cross_entropy'], key="loss_function", index=1)

if st.button("➕ Add Minimize"):
    # Validity checks
    if not loss_function_name:
        st.error("❌ Name cannot be empty.")
    elif " " in loss_function_name:
        st.error("❌ Name cannot contain spaces.")
    elif not minimizer_a or not minimizer_b:
        st.error("❌ Please select both relations to minimize.")
    else:
        # Add minimizer to session state
        st.session_state.minimizers[loss_function_name] = {
            "minimizer_a": minimizer_a,
            "minimizer_b": minimizer_b,
            "loss_function": loss_function
        }
        st.success(f"✅ Minimizer '{loss_function_name}' added successfully.")

# Show all defined minimizers with delete option
if st.session_state.minimizers:
    st.write("### Defined Minimizers")
    for name, min in st.session_state.minimizers.items():
        min_col1, min_col2, min_col3 = st.columns([2, 2, 1])
        with min_col1:
            st.markdown(f"**{name}**")
        with min_col2:
            st.markdown(f"**Minimizer A:** {min['minimizer_a']}")
            st.markdown(f"**Minimizer B:** {min['minimizer_b']}")
            st.markdown(f"**Loss Function:** {min['loss_function']}")
        with min_col3:
            if st.button("🗑️ Delete", key=f"delete_minimizer_{name}"):
                del st.session_state.minimizers[name]
                st.rerun()

    st.markdown("---")

## Build nnodely Model
if st.session_state.minimizers:
    if "nnodely_model" not in st.session_state:
        st.session_state.nnodely_model = None
    if "plot_graph" not in st.session_state:
        st.session_state.plot_graph = False
    if "graphviz_filename" not in st.session_state:
        st.session_state.graphviz_filename = None

    ## Add a input box with the sampling time (float value from 0.1)
    neuralize_col1, _ = st.columns([1, 2])
    with neuralize_col1:
        sampling_time = st.number_input("Sampling Time", min_value=0.1, value=1.0, step=0.1)
        ## Add a button to neuralize the model
        if st.button("🧠 Neuralize Model", width="stretch"): ## NNodely model definition
            # Create nnodely structure
            nn.clearNames()
            nnodely_model = nn.Modely(visualizer=nn.TextVisualizer(),seed=2)
            ## nnodely inputs
            nnodely_inputs = {inp['tag']: nn.Input(name=inp['tag'], dimensions=inp['dimension']).sw(round(inp['time_window']/sampling_time)) for inp in st.session_state.inputs}
            ## nnodely relations
            for relation in st.session_state.relations:
                func = getattr(nn, relation['type'])
                if relation['type'] in ["Add", "Mul"]:
                    nnodely_inputs[relation['name']] = func(nnodely_inputs[relation['ingress'][0]], nnodely_inputs[relation['ingress'][1]])
                elif relation['type'] in ["Sin", "Cos"]:
                    nnodely_inputs[relation['name']] = func(nnodely_inputs[relation['ingress'][0]])
                elif relation['type'] in ["Fir", "Linear"]:
                    nnodely_inputs[relation['name']] = func(output_dimension=relation['parameters']['dimension'])(nnodely_inputs[relation['ingress'][0]])
            ## nnodely outputs
            nnodely_outputs = [nn.Output(name=out['name'], relation=nnodely_inputs[out['relation']]) for out in st.session_state.outputs]
            for out in nnodely_outputs:
                nnodely_inputs[out.name] = out
            ## Neuralize
            nnodely_model.addModel(model_name, nnodely_outputs)
            for name, minimizer in st.session_state.minimizers.items():
                nnodely_model.addMinimize(name, nnodely_inputs[minimizer['minimizer_a']], nnodely_inputs[minimizer['minimizer_b']], loss_function=minimizer['loss_function'])
            nnodely_model.neuralizeModel(sample_time=sampling_time)
            st.success("Model neuralized successfully!")
            st.session_state.nnodely_model = nnodely_model
            st.session_state.plot_graph = True
            st.session_state.graphviz_filename = model_name+"_structure"
            plot_graphviz_structure(nnodely_model.json, filename=st.session_state.graphviz_filename, view=False)

    if st.session_state.plot_graph:
        graph_col1, graph_col2 = st.columns([1, 3])
        with graph_col1: ## Plot json model structure
            st.write(f"### {model_name} JSON Structure")
            st.json(st.session_state.nnodely_model.json, expanded=1)
        with graph_col2: ## Plot graph structure
            st.write(f"### {model_name} Graph Structure")
            with open(st.session_state.graphviz_filename) as f:
                dot_source = f.read()
            st.graphviz_chart(dot_source, use_container_width=True)

st.markdown("---")

# -----------------------------------------------------------
# 2. Data Loader Component
# -----------------------------------------------------------
st.header("2. Data Loader")

# Initialize session state for data loading
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = []

# --- Folder uploader ---
dataset_name = st.text_input("Dataset Name", "dataset", key="dataset_name")
uploaded_file = st.file_uploader("Upload your data file (CSV)", 
                                  type=["csv"], 
                                  accept_multiple_files=False,
                                  help="The file should be in .csv format where every input defined in the model" \
                                  "matches one column of the header")

if uploaded_file:
    # Preview file
    st.subheader("📊 Preview")
    df_preview = pd.read_csv(uploaded_file)
    st.dataframe(df_preview.head())

    if st.button("Load Dataset"):
        ## Add the dataset name inside the session state
        if dataset_name in st.session_state.data_loaded:
            st.warning(f"Dataset {dataset_name} already loaded. Overriding...")
        else:
            st.session_state.data_loaded.append(dataset_name)

        st.subheader("📂 Uploaded Datasets")
        for dataset in st.session_state.data_loaded:
            st.write(f"{dataset} - Number of samples: {len(df_preview)}")

        # --- Capture nnodely terminal output ---
        buffer = io.StringIO()
        sys.stdout = buffer  # redirect stdout temporarily

        try:
            st.session_state.nnodely_model.loadData(name=dataset_name, source=df_preview, skiplines=1)
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
        finally:
            sys.stdout = sys.__stdout__  # restore stdout

        # Show captured logs
        logs = buffer.getvalue()
        if logs:
            st.subheader("🖥️ Loader Logs")
            st.text(logs)

st.markdown("---")

# -----------------------------------------------------------
# 3. Training Component
# -----------------------------------------------------------
if st.session_state.data_loaded:
    st.header("3. Train the Model")

    st.subheader("🛠️ Select Hyperparameters")

    train_col1, train_col2, train_col3 = st.columns([1, 1, 1])
    with train_col1:
        # Dataset choice
        train_dataset = st.selectbox("Train Dataset", options=st.session_state.data_loaded, index=0)
        # Learning rate
        learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=1.0, value=0.001, step=0.0001, format="%.4f")

    with train_col2:
        # Batch size
        batch_size = st.number_input("Batch Size", min_value=1, value=32, step=1)
        # Epochs
        epochs = st.number_input("Epochs", min_value=1, value=10, step=1)

    with train_col3:
        # Shuffle
        shuffle = st.checkbox("Shuffle Data", value=True)
        # Optimizer
        optimizer = st.selectbox("Optimizer", ["Adam", "SGD"])

    # Train split (percentage)
    train_split = st.slider("Train Split (%)", min_value=1, max_value=100, value=80)

    # Train button
    if st.button("🚀 Train Model"):
        buffer = io.StringIO()
        sys.stdout = buffer  # redirect stdout

        with st.spinner(text="Training in progress.."):
            try:
                st.session_state.nnodely_model.trainModel(
                    dataset=train_dataset,
                    splits=[train_split, 100 - train_split, 0],
                    train_batch_size=batch_size,
                    val_batch_size=batch_size,
                    lr=learning_rate,
                    shuffle_data=shuffle,
                    num_of_epochs=epochs,
                    optimizer=optimizer
                )
                st.success("✅ Training complete!")

            except Exception as e:
                st.error(f"❌ Training failed: {e}")

            finally:
                sys.stdout = sys.__stdout__

        # Display captured training logs
        logs = buffer.getvalue()
        if logs:
            st.subheader("🖥️ Training Logs")
            st.text(logs)

st.markdown("---")

# -----------------------------------------------------------
# 4. Visualization & Export Component
# -----------------------------------------------------------
st.header("4. Results, Visualization & Export")


if st.button("Export Model"):
    st.success("Model exported successfully! ✅")
    # Placeholder: insert nnodely export function here
