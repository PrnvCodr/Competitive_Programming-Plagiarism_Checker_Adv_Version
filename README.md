# 🛡️ Competitive Programming Plagiarism Checker

A powerful, multi-algorithm plagiarism detection system built with Streamlit, designed specifically for **competitive programming submissions**. With 12 cutting-edge algorithms, rich visualizations, and robust analysis features, this tool helps educators and administrators detect code similarity with high accuracy and transparency.

---

## 🚀 Features

- 🧠 **12 Advanced Algorithms** (MOSS-inspired, Token, Structure, Edit Distance, etc.)
- 📊 **Interactive Visualizations** with Plotly, Matplotlib & NetworkX
- 🗃️ **Multi-Format Reports** (CSV, JSON, Markdown)
- 💡 **Risk Assessment System** with Heatmaps & Network Graphs
- 🔒 **Privacy-First Design**: All code processed locally
- 📂 **Rich Metadata Extraction** (Cyclomatic complexity, function count, etc.)
- 🧩 **Custom Configuration System** for thresholds, preprocessing, and more

[🔗 Live Demo]([https://cpu-scheduler-visualizer-final.vercel.app/](https://competitiveprogramming-plagiarismcheckeradvversion-uuhprt5wpal.streamlit.app/))
---

## 🧰 Algorithms Implemented

| Algorithm                     | Purpose                                               |
|------------------------------|--------------------------------------------------------|
| Advanced MOSS (Mock)         | Combines fingerprinting, tokens, and structure         |
| Token Frequency Analysis     | Identifies similarity in keyword/operator frequencies |
| Structure-Based Comparison   | Uses control structures like if/for/while             |
| Winnowing Fingerprinting     | k-gram based token hashing                            |
| Edit Distance (Levenshtein)  | Character-level difference computation                |
| Longest Common Subsequence   | Token sequence comparison                             |
| Jaccard Similarity           | Token set similarity                                  |
| Cosine Similarity (TF-IDF)   | Vectorized token comparison                           |
| N-Gram Analysis              | Pattern matching using sliding windows                |
| Control Flow Analysis        | Detects logic-level similarity                        |
| Variable Renaming Detection  | Normalizes identifiers for fair comparison            |
| Comment Removal Analysis     | Ignores comment-level obfuscation                     |

---

## 🛠️ Tech Stack

- **Frontend/UI**: Streamlit
- **Data Processing**: Python (Regex, Pandas, NumPy)
- **Visualization**: Plotly, Matplotlib, Seaborn, NetworkX
- **Machine Learning Utilities**: Scikit-learn

---

## 📁 Tabs in UI

1. **Analysis Results** – Summary scores, suspicious pair highlighting
2. **Visualizations** – Similarity heatmaps, graphs, histograms
3. **Code Comparison** – Side-by-side viewer with unified diffs
4. **Detailed Report** – Executive summaries & stats
5. **File Explorer** – Browse code, metadata, includes
6. **Export Results** – Download as CSV, JSON, or Markdown

---

## ⚙️ Configuration Options

- Similarity thresholds (default: `0.5`)
- Select which algorithms to apply
- Enable/disable:
  - Comment stripping
  - Whitespace normalization
  - Variable normalization
- File limits:
  - Min size: 100 bytes
  - Max size: 100 KB
  - Extensions: `.cpp`, `.cc`, `.cxx`, `.c`

---

## 📊 Visualization Examples

- 📍 **Similarity Heatmap** (interactive)
- 📈 **Histogram** of similarity scores
- 🧬 **Control Flow Graph** (NetworkX)
- 🧠 **Risk Distribution Pie Chart**

---

## 📦 Export Options

- `CSV`: Easy spreadsheet filtering
- `JSON`: For API integration
- `Markdown`: Readable reports
- **Custom Reports**: Select fields, snippets, metadata

---

## 📂 File Handling

- Accepts: `.cpp`, `.cc`, `.cxx`, `.c`, or `.zip`
- Supports:
  - Drag & drop
  - Multi-file or zip upload
  - UTF-8 encoding auto-detection
  - Metadata: LOC, complexity, function count

---

## 🚦 Risk Assessment

| Risk Level     | Similarity Score     |
|----------------|----------------------|
| 🟥 High Risk    | ≥ 80%                |
| 🟧 Medium Risk  | 60% – 79%            |
| 🟨 Low Risk     | Threshold – 59%      |
| 🟩 Safe         | < Threshold (e.g., 50%) |

---

## ⚡ Performance Optimizations

- Caching of results and preprocessing
- Lazy loading of files and ZIP streaming
- Early termination for highly dissimilar files
- Regex and vectorized operation optimizations

---

## 🧪 Testing Strategy

- ✅ Unit & integration tests for all algorithms
- 📉 Benchmarking for performance
- 🧬 False positive & obfuscation variant validation
- 🖥️ UI and file upload stress testing

---

## 🔐 Privacy & Security

- Local analysis only (no cloud upload)
- Session-cleared file storage
- Input sanitization and file validation
- Encoding fallback and safe rendering

---

## 🛠️ System Requirements

- Python ≥ 3.8
- RAM: 4GB minimum (8GB recommended)
- Browser: Chrome 90+, Firefox 88+, Safari 14+

Install dependencies:
```bash
pip install -r requirements.txt
