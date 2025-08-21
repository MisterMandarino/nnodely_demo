# nnodely_demo

🚀 **nnodely_demo** is an interactive [Streamlit](https://streamlit.io/) application showcasing the capabilities of **[nnodely](https://github.com/tonegas/nnodely.git)** — a new framework for modeling **mechanical systems** with **structured neural networks**.  

This demo allows users to:
- Define **mechanical systems** by selecting **inputs, outputs, and relations**.
- Choose **minimization functions** and **optimizers**.
- Train and validate structured neural networks with `nnodely`.
- Export trained models for further use.

---

## 📸 Demo Preview
*(Add screenshot/gif of your Streamlit app here once running)*

---

## ✨ Features
- 🔧 **System Modeling** – Build mechanical models by specifying variables and relations.  
- ⚙️ **Configurable Training** – Select loss functions, optimization algorithms, and training parameters.  
- 📊 **Validation & Visualization** – Validate models and view training metrics interactively.  
- 💾 **Model Export** – Save and export trained models for downstream applications.  

---

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/MisterMandarino/nnodely_demo.git
   cd nnodely_demo
   ```
2. Create and activate a virtual environment:
   Mac/Linux:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

   Windows:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Install nnodely:
   ```bash
   pip install nnodely
   ```

## ▶️ Usage

Run the Streamlit demo:
```bash
streamlit run app.py
```

## 📂 Project Structure

```bash
nnodely_demo/
│── app.py              # Streamlit demo app
│── requirements.txt    # Dependencies
│── README.md           # Project documentation
│── /assets             # (optional) images/screenshots
└── /notebooks          # (optional) experiments or examples
```

## 📖 Documentation

For full details on the `nnodely` framework — including concepts, tutorials, and API reference — please visit the official documentation:  

👉 [nnodely Documentation](https://nnodely.readthedocs.io/en/latest/)
