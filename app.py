import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import hashlib
import re
from collections import Counter, defaultdict
from itertools import combinations
import io
import base64
from datetime import datetime
import zipfile
import tempfile
import os

# Configure Streamlit page
st.set_page_config(
    page_title="Competitive Programming Plagiarism Checker",
    page_icon="âš¡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI (add this to your existing CSS section)
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .suspicious-card {
        background: #fff5f5;
        border-left: 4px solid #e53e3e;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .safe-card {
        background: #f0fff4;
        border-left: 4px solid #38a169;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    /* Fix for tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 10px 10px 0 0;
        padding: 0.5rem 1rem;
        color: #333333 !important;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #667eea !important;
        color: white !important;
        font-weight: 600;
    }
    
    /* Fix for button styling */
    .stButton > button {
        background-color: #667eea;
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stButton > button:hover {
        background-color: #5a67d8;
        color: white !important;
    }
    
    /* Fix for download button styling */
    .stDownloadButton > button {
        background-color: #38a169 !important;
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    /* Fix for dataframe styling - IMPORTANT FOR RESULTS VISIBILITY */
    .stDataFrame {
        background-color: white !important;
    }
    
    .stDataFrame > div {
        background-color: white !important;
    }
    
    /* Fix dataframe table styling */
    .stDataFrame table {
        background-color: white !important;
        color: #333333 !important;
    }
    
    .stDataFrame th {
        background-color: #f8f9fa !important;
        color: #333333 !important;
        font-weight: 600;
        border-bottom: 2px solid #dee2e6 !important;
    }
    
    .stDataFrame td {
        background-color: white !important;
        color: #333333 !important;
        border-bottom: 1px solid #dee2e6 !important;
    }
    
    /* Fix for styled dataframe */
    .stDataFrame [data-testid="stDataFrame"] {
        background-color: white !important;
    }
    
    /* Ensure text is visible in all elements */
    .stMarkdown, .stText {
        color: #2d3748 !important;
    }
    
    /* Fix metric containers */
    [data-testid="metric-container"] {
        background-color: white !important;
        border: 1px solid #e2e8f0 !important;
        padding: 1rem !important;
        border-radius: 10px !important;
        color: #333333 !important;
    }
    
    [data-testid="metric-container"] > div {
        color: #333333 !important;
    }
    
    /* Fix for expander content */
    .streamlit-expanderContent {
        background-color: white !important;
        color: #333333 !important;
    }
    
    /* Fix for code blocks */
    .stCodeBlock {
        background-color: #f8f9fa !important;
    }
    
    /* Fix for any remaining dark backgrounds */
    .main .block-container {
        background-color: white !important;
    }
</style>
""", unsafe_allow_html=True)

class CompetitiveProgrammingPlagiarismDetector:
    def __init__(self):
        self.files = {}
        self.file_metadata = {}
        self.similarity_scores = {}
        self.algorithms = [
            "Advanced MOSS (Mock)",
            "Token Frequency Analysis", 
            "Structure-based Comparison",
            "Winnowing Fingerprinting",
            "Edit Distance (Levenshtein)",
            "Longest Common Subsequence",
            "Jaccard Similarity",
            "Cosine Similarity",
            "N-Gram Analysis",
            "Control Flow Analysis",
            "Variable Renaming Detection",
            "Comment Removal Analysis"
        ]
        
    def add_file(self, filename, content, file_size):
        """Add a C++ file with metadata"""
        self.files[filename] = content
        self.file_metadata[filename] = {
            'size': file_size,
            'lines': len(content.split('\n')),
            'functions': self.count_functions(content),
            'complexity': self.calculate_complexity(content),
            'includes': self.extract_includes(content),
            'upload_time': datetime.now()
        }
    
    def preprocess_cpp_code(self, code):
        """Advanced C++ code preprocessing"""
        # Remove single-line comments
        code = re.sub(r'//.*', '', code)
        # Remove multi-line comments
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        # Remove extra whitespace
        code = re.sub(r'\s+', ' ', code)
        # Remove string literals (to focus on logic)
        code = re.sub(r'"[^"]*"', '""', code)
        code = re.sub(r"'[^']*'", "''", code)
        return code.strip()
    
    def normalize_variables(self, code):
        """Normalize variable names to detect variable renaming"""
        # Extract variable declarations
        var_pattern = r'\b(int|long|char|float|double|string|bool)\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        variables = re.findall(var_pattern, code)
        
        # Replace variables with generic names
        normalized_code = code
        for i, (type_name, var_name) in enumerate(variables):
            normalized_code = re.sub(r'\b' + var_name + r'\b', f'var{i}', normalized_code)
        
        return normalized_code
    
    def tokenize_cpp_code(self, code):
        """Advanced C++ tokenization"""
        # C++ keywords and operators
        cpp_tokens = re.findall(r'''
            \b(?:int|long|char|float|double|string|bool|void|if|else|for|while|do|switch|case|break|continue|return|class|struct|public|private|protected|namespace|using|include|define|ifdef|ifndef|endif)\b|
            [{}();,\[\]<>=!&|+\-*/%.]+|
            \b[a-zA-Z_][a-zA-Z0-9_]*\b|
            \b\d+\b
        ''', code, re.VERBOSE)
        
        return [token.lower() for token in cpp_tokens if token.strip()]
    
    def extract_includes(self, code):
        """Extract #include statements"""
        includes = re.findall(r'#include\s*[<"][^>"]*[>"]', code)
        return includes
    
    def count_functions(self, code):
        """Count function definitions"""
        # Simple function counting (can be improved)
        function_pattern = r'\b\w+\s+\w+\s*\([^)]*\)\s*{'
        return len(re.findall(function_pattern, code))
    
    def calculate_complexity(self, code):
        """Calculate cyclomatic complexity approximation"""
        complexity_keywords = ['if', 'else', 'for', 'while', 'do', 'switch', 'case', '&&', '||']
        complexity = 1  # Base complexity
        
        for keyword in complexity_keywords:
            complexity += len(re.findall(r'\b' + keyword + r'\b', code, re.IGNORECASE))
        
        return complexity
    
    def extract_control_flow(self, code):
        """Extract control flow structure"""
        control_structures = []
        
        # Find control flow patterns
        patterns = {
            'if': r'\bif\s*\(',
            'for': r'\bfor\s*\(',
            'while': r'\bwhile\s*\(',
            'switch': r'\bswitch\s*\(',
            'function': r'\b\w+\s+\w+\s*\([^)]*\)\s*{'
        }
        
        for structure, pattern in patterns.items():
            matches = re.finditer(pattern, code)
            for match in matches:
                control_structures.append((structure, match.start()))
        
        return sorted(control_structures, key=lambda x: x[1])
    
    def winnowing_fingerprint_advanced(self, tokens, k=7, w=5):
        """Advanced winnowing with better parameters for C++"""
        if len(tokens) < k:
            return set()
        
        # Generate k-grams
        kgrams = []
        for i in range(len(tokens) - k + 1):
            kgram = ' '.join(tokens[i:i+k])
            hash_val = hash(kgram) % (2**20)  # Larger hash space
            kgrams.append((i, hash_val, kgram))
        
        # Winnowing: select minimum hash in each window
        fingerprints = set()
        for i in range(len(kgrams) - w + 1):
            window = kgrams[i:i+w]
            min_hash = min(window, key=lambda x: x[1])
            fingerprints.add((min_hash[0], min_hash[1]))
        
        return fingerprints
    
    def advanced_moss_similarity(self, code1, code2):
        """Advanced MOSS-like similarity calculation"""
        # Preprocess both codes
        proc1 = self.preprocess_cpp_code(code1)
        proc2 = self.preprocess_cpp_code(code2)
        
        # Normalize variables
        norm1 = self.normalize_variables(proc1)
        norm2 = self.normalize_variables(proc2)
        
        # Tokenize
        tokens1 = self.tokenize_cpp_code(norm1)
        tokens2 = self.tokenize_cpp_code(norm2)
        
        # Multiple similarity measures
        similarities = []
        
        # 1. Winnowing fingerprints
        fp1 = self.winnowing_fingerprint_advanced(tokens1)
        fp2 = self.winnowing_fingerprint_advanced(tokens2)
        if fp1 or fp2:
            fp_sim = len(fp1.intersection(fp2)) / len(fp1.union(fp2)) if fp1.union(fp2) else 0
            similarities.append(fp_sim * 0.3)
        
        # 2. Token sequence similarity
        seq_sim = self.sequence_similarity(tokens1, tokens2)
        similarities.append(seq_sim * 0.25)
        
        # 3. Control flow similarity
        cf1 = self.extract_control_flow(proc1)
        cf2 = self.extract_control_flow(proc2)
        cf_sim = self.control_flow_similarity(cf1, cf2)
        similarities.append(cf_sim * 0.2)
        
        # 4. Structure similarity
        struct_sim = self.structure_similarity(code1, code2)
        similarities.append(struct_sim * 0.25)
        
        return sum(similarities)
    
    def sequence_similarity(self, seq1, seq2):
        """Calculate sequence similarity using LCS"""
        if not seq1 or not seq2:
            return 0
        
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_length = dp[m][n]
        return lcs_length / max(m, n)
    
    def control_flow_similarity(self, cf1, cf2):
        """Compare control flow structures"""
        if not cf1 or not cf2:
            return 0
        
        # Extract just the structure types
        struct1 = [item[0] for item in cf1]
        struct2 = [item[0] for item in cf2]
        
        return self.sequence_similarity(struct1, struct2)
    
    def structure_similarity(self, code1, code2):
        """Compare overall code structure"""
        # Count different code elements
        elements = ['if', 'for', 'while', 'switch', 'function', '{', '}', '(', ')']
        
        counts1 = {}
        counts2 = {}
        
        for element in elements:
            # Escape special regex characters
            if element in ['(', ')', '{', '}', '[', ']', '+', '*', '?', '^', '$', '|', '\\']:
                escaped_element = re.escape(element)
                pattern = r'\b' + escaped_element + r'\b' if element.isalnum() else escaped_element
            else:
                pattern = r'\b' + element + r'\b'
            
            counts1[element] = len(re.findall(pattern, code1, re.IGNORECASE))
            counts2[element] = len(re.findall(pattern, code2, re.IGNORECASE))
        
        # Calculate cosine similarity of counts
        vec1 = list(counts1.values())
        vec2 = list(counts2.values())
        
        if sum(vec1) == 0 or sum(vec2) == 0:
            return 0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        
        return dot_product / (magnitude1 * magnitude2)

    
    def calculate_similarity(self, file1, file2, algorithm):
        """Calculate similarity using specified algorithm"""
        code1, code2 = self.files[file1], self.files[file2]
        
        if algorithm == "Advanced MOSS (Mock)":
            return self.advanced_moss_similarity(code1, code2)
        elif algorithm == "Token Frequency Analysis":
            return self.token_frequency_similarity(code1, code2)
        elif algorithm == "Structure-based Comparison":
            return self.structure_similarity(code1, code2)
        elif algorithm == "Winnowing Fingerprinting":
            return self.winnowing_similarity(code1, code2)
        elif algorithm == "Edit Distance (Levenshtein)":
            return self.levenshtein_similarity(code1, code2)
        elif algorithm == "Longest Common Subsequence":
            return self.lcs_similarity(code1, code2)
        elif algorithm == "Jaccard Similarity":
            return self.jaccard_similarity(code1, code2)
        elif algorithm == "Cosine Similarity":
            return self.cosine_similarity(code1, code2)
        elif algorithm == "N-Gram Analysis":
            return self.ngram_similarity(code1, code2)
        elif algorithm == "Control Flow Analysis":
            return self.control_flow_analysis(code1, code2)
        elif algorithm == "Variable Renaming Detection":
            return self.variable_renaming_detection(code1, code2)
        elif algorithm == "Comment Removal Analysis":
            return self.comment_removal_analysis(code1, code2)
        else:
            return 0
    
    def token_frequency_similarity(self, code1, code2):
        """Token frequency based similarity"""
        tokens1 = self.tokenize_cpp_code(self.preprocess_cpp_code(code1))
        tokens2 = self.tokenize_cpp_code(self.preprocess_cpp_code(code2))
        
        freq1 = Counter(tokens1)
        freq2 = Counter(tokens2)
        
        all_tokens = set(tokens1 + tokens2)
        similarity = 0
        
        for token in all_tokens:
            f1, f2 = freq1.get(token, 0), freq2.get(token, 0)
            similarity += min(f1, f2)
        
        total_tokens = max(len(tokens1), len(tokens2))
        return similarity / total_tokens if total_tokens > 0 else 0
    
    def winnowing_similarity(self, code1, code2):
        """Winnowing fingerprint similarity"""
        tokens1 = self.tokenize_cpp_code(self.preprocess_cpp_code(code1))
        tokens2 = self.tokenize_cpp_code(self.preprocess_cpp_code(code2))
        
        fp1 = self.winnowing_fingerprint_advanced(tokens1)
        fp2 = self.winnowing_fingerprint_advanced(tokens2)
        
        if not fp1 and not fp2:
            return 1.0
        if not fp1 or not fp2:
            return 0.0
        
        intersection = len(fp1.intersection(fp2))
        union = len(fp1.union(fp2))
        
        return intersection / union
    
    def levenshtein_similarity(self, code1, code2):
        """Levenshtein distance similarity"""
        proc1 = self.preprocess_cpp_code(code1)
        proc2 = self.preprocess_cpp_code(code2)
        
        if len(proc1) < len(proc2):
            proc1, proc2 = proc2, proc1
        
        if len(proc2) == 0:
            return 0
        
        previous_row = list(range(len(proc2) + 1))
        for i, c1 in enumerate(proc1):
            current_row = [i + 1]
            for j, c2 in enumerate(proc2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        distance = previous_row[-1]
        max_len = max(len(proc1), len(proc2))
        return 1 - (distance / max_len) if max_len > 0 else 0
    
    def lcs_similarity(self, code1, code2):
        """Longest Common Subsequence similarity"""
        proc1 = self.preprocess_cpp_code(code1)
        proc2 = self.preprocess_cpp_code(code2)
        
        return self.sequence_similarity(list(proc1), list(proc2))
    
    def jaccard_similarity(self, code1, code2):
        """Jaccard similarity on token sets"""
        tokens1 = set(self.tokenize_cpp_code(self.preprocess_cpp_code(code1)))
        tokens2 = set(self.tokenize_cpp_code(self.preprocess_cpp_code(code2)))
        
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        return intersection / union if union > 0 else 0
    
    def cosine_similarity(self, code1, code2):
        """Cosine similarity using TF-IDF"""
        proc1 = self.preprocess_cpp_code(code1)
        proc2 = self.preprocess_cpp_code(code2)
        
        vectorizer = TfidfVectorizer(token_pattern=r'\b\w+\b')
        try:
            tfidf_matrix = vectorizer.fit_transform([proc1, proc2])
            similarity_matrix = cosine_similarity(tfidf_matrix)
            return similarity_matrix[0][1]
        except:
            return 0
    
    def ngram_similarity(self, code1, code2, n=3):
        """N-gram similarity analysis"""
        tokens1 = self.tokenize_cpp_code(self.preprocess_cpp_code(code1))
        tokens2 = self.tokenize_cpp_code(self.preprocess_cpp_code(code2))
        
        if len(tokens1) < n or len(tokens2) < n:
            return 0
        
        ngrams1 = set(' '.join(tokens1[i:i+n]) for i in range(len(tokens1) - n + 1))
        ngrams2 = set(' '.join(tokens2[i:i+n]) for i in range(len(tokens2) - n + 1))
        
        intersection = len(ngrams1.intersection(ngrams2))
        union = len(ngrams1.union(ngrams2))
        
        return intersection / union if union > 0 else 0
    
    def control_flow_analysis(self, code1, code2):
        """Control flow pattern analysis"""
        cf1 = self.extract_control_flow(code1)
        cf2 = self.extract_control_flow(code2)
        
        return self.control_flow_similarity(cf1, cf2)
    
    def variable_renaming_detection(self, code1, code2):
        """Detect variable renaming plagiarism"""
        norm1 = self.normalize_variables(self.preprocess_cpp_code(code1))
        norm2 = self.normalize_variables(self.preprocess_cpp_code(code2))
        
        tokens1 = self.tokenize_cpp_code(norm1)
        tokens2 = self.tokenize_cpp_code(norm2)
        
        return self.sequence_similarity(tokens1, tokens2)
    
    def comment_removal_analysis(self, code1, code2):
        """Analyze similarity after removing comments"""
        # Remove comments and compare
        clean1 = self.preprocess_cpp_code(code1)
        clean2 = self.preprocess_cpp_code(code2)
        
        # Also compare original vs cleaned versions
        orig_sim = self.sequence_similarity(
            self.tokenize_cpp_code(code1),
            self.tokenize_cpp_code(code2)
        )
        
        clean_sim = self.sequence_similarity(
            self.tokenize_cpp_code(clean1),
            self.tokenize_cpp_code(clean2)
        )
        
        # If similarity increases significantly after cleaning, it's suspicious
        return max(orig_sim, clean_sim)
    
    def compare_all_files(self, algorithm, threshold=0.5):
        """Compare all file pairs"""
        results = []
        filenames = list(self.files.keys())
        
        for i, file1 in enumerate(filenames):
            for j, file2 in enumerate(filenames[i+1:], i+1):
                similarity = self.calculate_similarity(file1, file2, algorithm)
                
                # Determine risk level
                if similarity >= 0.8:
                    risk = "ðŸ”´ High Risk"
                elif similarity >= 0.6:
                    risk = "ðŸŸ¡ Medium Risk"
                elif similarity >= threshold:
                    risk = "ðŸŸ  Low Risk"
                else:
                    risk = "ðŸŸ¢ Safe"
                
                results.append({
                    'File 1': file1,
                    'File 2': file2,
                    'Similarity': similarity,
                    'Percentage': f"{similarity * 100:.2f}%",
                    'Risk Level': risk,
                    'Status': 'Suspicious' if similarity >= threshold else 'OK',
                    'Algorithm': algorithm
                })
        
        return sorted(results, key=lambda x: x['Similarity'], reverse=True)

def create_advanced_visualizations(detector, results, algorithm):
    """Create advanced visualizations"""
    if not results:
        return None, None, None
    
    # 1. Interactive Similarity Matrix
    filenames = list(detector.files.keys())
    n = len(filenames)
    matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 1.0
            elif i < j:
                similarity = detector.calculate_similarity(filenames[i], filenames[j], algorithm)
                matrix[i][j] = similarity
                matrix[j][i] = similarity
    
    fig_heatmap = px.imshow(
        matrix,
        x=filenames,
        y=filenames,
        color_continuous_scale='Reds',
        title=f'Similarity Matrix - {algorithm}',
        labels=dict(color="Similarity Score")
    )
    fig_heatmap.update_layout(width=800, height=600)
    
    # 2. Risk Distribution
    risk_counts = Counter([r['Risk Level'] for r in results])
    fig_risk = px.pie(
        values=list(risk_counts.values()),
        names=list(risk_counts.keys()),
        title="Risk Level Distribution",
        color_discrete_map={
            "ðŸ”´ High Risk": "#e53e3e",
            "ðŸŸ¡ Medium Risk": "#d69e2e", 
            "ðŸŸ  Low Risk": "#dd6b20",
            "ðŸŸ¢ Safe": "#38a169"
        }
    )
    
    # 3. Similarity Score Distribution
    similarities = [r['Similarity'] for r in results]
    fig_dist = px.histogram(
        x=similarities,
        nbins=20,
        title="Similarity Score Distribution",
        labels={'x': 'Similarity Score', 'y': 'Count'}
    )
    fig_dist.add_vline(x=0.5, line_dash="dash", line_color="red", 
                       annotation_text="Threshold")
    
    return fig_heatmap, fig_risk, fig_dist

def generate_detailed_report(detector, results, algorithm, threshold):
    """Generate detailed analysis report"""
    total_files = len(detector.files)
    total_comparisons = len(results)
    suspicious_pairs = sum(1 for r in results if r['Status'] == 'Suspicious')
    
    report = f"""
# ðŸ† Competitive Programming Plagiarism Analysis Report

## ðŸ“Š Executive Summary
- **Analysis Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Algorithm Used**: {algorithm}
- **Detection Threshold**: {threshold:.2f}
- **Files Analyzed**: {total_files}
- **Total Comparisons**: {total_comparisons}
- **Suspicious Pairs**: {suspicious_pairs}
- **Detection Rate**: {(suspicious_pairs/total_comparisons*100):.1f}%

## ðŸ” File Analysis Details
"""
    
    for filename, metadata in detector.file_metadata.items():
        report += f"""
### ðŸ“„ {filename}
- **Size**: {metadata['size']} bytes
- **Lines of Code**: {metadata['lines']}
- **Functions**: {metadata['functions']}
- **Complexity Score**: {metadata['complexity']}
- **Includes**: {len(metadata['includes'])}
"""
    
    report += "\n## ðŸš¨ High-Risk Comparisons\n"
    
    high_risk = [r for r in results if r['Similarity'] >= 0.8]
    for result in high_risk[:10]:  # Top 10
        report += f"- **{result['File 1']}** â†” **{result['File 2']}**: {result['Percentage']} similarity\n"
    
    return report

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>âš¡ Competitive Programming Plagiarism Checker</h1>
        <p>Advanced C++ Code Similarity Detection for Competitive Programming</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize detector
    if 'detector' not in st.session_state:
        st.session_state.detector = CompetitiveProgrammingPlagiarismDetector()
    
    # Sidebar
    with st.sidebar:
        st.header("ðŸ”§ Configuration Panel")
        
        # File upload section
        st.subheader("ðŸ“ Upload C++ Files")
        uploaded_files = st.file_uploader(
            "Choose C++ files",
            type=['cpp', 'cc', 'cxx', 'c'],
            accept_multiple_files=True,
            help="Upload multiple C++ source files for comparison"
        )
        
        # Bulk upload option
        st.subheader("ðŸ“¦ Bulk Upload (ZIP)")
        zip_file = st.file_uploader(
            "Upload ZIP file containing C++ files",
            type=['zip'],
            help="Upload a ZIP file containing multiple C++ files"
        )
        
        # Algorithm selection
        st.subheader("ðŸ§  Detection Algorithm")
        algorithm = st.selectbox(
            "Select Algorithm",
            st.session_state.detector.algorithms,
            index=0,
            help="Choose the plagiarism detection algorithm"
        )
        
        # Advanced settings
        st.subheader("âš™ï¸ Advanced Settings")
        threshold = st.slider(
            "Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Minimum similarity to flag as suspicious"
        )
        
        min_file_size = st.number_input(
            "Minimum File Size (bytes)",
            min_value=0,
            value=100,
            help="Ignore files smaller than this size"
        )
        
        max_file_size = st.number_input(
            "Maximum File Size (KB)",
            min_value=1,
            value=100,
            help="Ignore files larger than this size"
        )
        
        # Analysis options
        st.subheader("ðŸ” Analysis Options")
        ignore_includes = st.checkbox("Ignore #include statements", value=True)
        ignore_comments = st.checkbox("Ignore comments", value=True)
        normalize_whitespace = st.checkbox("Normalize whitespace", value=True)
        detect_variable_renaming = st.checkbox("Detect variable renaming", value=True)
    
    # Process uploaded files
    if uploaded_files or zip_file:
        st.session_state.detector = CompetitiveProgrammingPlagiarismDetector()
        
        files_to_process = []
        
        # Handle regular file uploads
        if uploaded_files:
            files_to_process.extend(uploaded_files)
        
        # Handle ZIP file upload
        if zip_file:
            with tempfile.TemporaryDirectory() as temp_dir:
                zip_path = os.path.join(temp_dir, "uploaded.zip")
                with open(zip_path, "wb") as f:
                    f.write(zip_file.read())
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    for file_info in zip_ref.filelist:
                        if file_info.filename.endswith(('.cpp', '.cc', '.cxx', '.c')):
                            content = zip_ref.read(file_info.filename).decode('utf-8', errors='ignore')
                            # Create a mock uploaded file object
                            class MockFile:
                                def __init__(self, name, content, size):
                                    self.name = name
                                    self.content = content
                                    self.size = size
                                def read(self):
                                    return self.content.encode('utf-8')
                            
                            mock_file = MockFile(file_info.filename, content, file_info.file_size)
                            files_to_process.append(mock_file)
        
        # Process all files
        processed_count = 0
        skipped_count = 0
        
        for uploaded_file in files_to_process:
            try:
                if hasattr(uploaded_file, 'read'):
                    content = uploaded_file.read().decode('utf-8', errors='ignore')
                else:
                    content = uploaded_file.content
                
                file_size = len(content.encode('utf-8'))
                
                # File validation
                if file_size < min_file_size:
                    st.sidebar.warning(f"âš ï¸ {uploaded_file.name} too small ({file_size} bytes)")
                    skipped_count += 1
                    continue
                elif file_size > max_file_size * 1024:
                    st.sidebar.warning(f"âš ï¸ {uploaded_file.name} too large ({file_size/1024:.1f} KB)")
                    skipped_count += 1
                    continue
                elif len(content.strip()) == 0:
                    st.sidebar.warning(f"âš ï¸ {uploaded_file.name} is empty")
                    skipped_count += 1
                    continue
                
                st.session_state.detector.add_file(uploaded_file.name, content, file_size)
                processed_count += 1
                
            except Exception as e:
                st.sidebar.error(f"âŒ Error processing {uploaded_file.name}: {str(e)}")
                skipped_count += 1
    
    # Display file statistics
    file_count = len(st.session_state.detector.files)
    
    if file_count > 0:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("ðŸ“ Files Loaded", file_count)
        with col2:
            if 'processed_count' in locals():
                st.metric("âœ… Processed", processed_count)
        with col3:
            if 'skipped_count' in locals():
                st.metric("â­ï¸ Skipped", skipped_count)
    
    if file_count < 2:
        st.info("ðŸ‘† Please upload at least 2 C++ files to start plagiarism detection.")
        
        # Show sample code for demonstration
        with st.expander("ðŸ“– View Sample Analysis"):
            st.markdown("""
            ### Sample C++ Code Comparison
            
            **Features of this plagiarism checker:**
            - ðŸ” **12 Advanced Algorithms** including MOSS simulation
            - ðŸŽ¯ **Variable Renaming Detection** 
            - ðŸ“Š **Control Flow Analysis**
            - ðŸ”„ **Structure-based Comparison**
            - ðŸ“ˆ **Interactive Visualizations**
            - ðŸ“‹ **Detailed Reports**
            - ðŸš€ **Optimized for Competitive Programming**
            """)
        return
    
    # Main analysis tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "ðŸ“Š Analysis Results", 
        "ðŸ“ˆ Visualizations", 
        "ðŸ” Code Comparison", 
        "ðŸ“‹ Detailed Report",
        "ðŸ“ File Explorer",
        "ðŸ“¥ Export Results"
    ])
    
    # Calculate results
    with st.spinner(f"ðŸ”„ Analyzing {file_count} files using {algorithm}..."):
        results = st.session_state.detector.compare_all_files(algorithm, threshold)
    
    with tab1:
        st.header("ðŸ“Š Plagiarism Analysis Results")
        
        if results:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <h3>ðŸ” Total Comparisons</h3>
                    <h2>{}</h2>
                </div>
                """.format(len(results)), unsafe_allow_html=True)
            
            with col2:
                suspicious_count = sum(1 for r in results if r['Status'] == 'Suspicious')
                st.markdown("""
                <div class="metric-card">
                    <h3>ðŸš¨ Suspicious Pairs</h3>
                    <h2>{}</h2>
                </div>
                """.format(suspicious_count), unsafe_allow_html=True)
            
            with col3:
                avg_similarity = np.mean([r['Similarity'] for r in results])
                st.markdown("""
                <div class="metric-card">
                    <h3>ðŸ“Š Average Similarity</h3>
                    <h2>{:.1f}%</h2>
                </div>
                """.format(avg_similarity * 100), unsafe_allow_html=True)
            
            with col4:
                max_similarity = max([r['Similarity'] for r in results])
                st.markdown("""
                <div class="metric-card">
                    <h3>âš ï¸ Highest Similarity</h3>
                    <h2>{:.1f}%</h2>
                </div>
                """.format(max_similarity * 100), unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Results table with enhanced styling
            df = pd.DataFrame(results)
            
            # Color coding function
            # Color coding function with better contrast
            def highlight_risk(row):
                if row['Similarity'] >= 0.8:
                    return ['background-color: #fed7d7; color: #742a2a; font-weight: bold;'] * len(row)
                elif row['Similarity'] >= 0.6:
                    return ['background-color: #fef5e7; color: #744210; font-weight: bold;'] * len(row)
                elif row['Similarity'] >= threshold:
                    return ['background-color: #feebc8; color: #7c2d12; font-weight: bold;'] * len(row)
                else:
                    return ['background-color: #f0fff4; color: #22543d; font-weight: bold;'] * len(row)

            
            styled_df = df.style.apply(highlight_risk, axis=1)
            st.dataframe(styled_df, use_container_width=True, height=400)
            
            # Top suspicious pairs
            st.subheader("ðŸ”´ Most Suspicious Pairs")
            top_suspicious = [r for r in results if r['Similarity'] >= 0.7][:5]
            
            for result in top_suspicious:
                if result['Similarity'] >= 0.8:
                    card_class = "suspicious-card"
                else:
                    card_class = "safe-card"
                
                st.markdown(f"""
                <div class="{card_class}">
                    <h4>{result['File 1']} â†” {result['File 2']}</h4>
                    <p><strong>Similarity:</strong> {result['Percentage']} | <strong>Risk:</strong> {result['Risk Level']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        else:
            st.info("No results to display.")
    
    with tab2:
        st.header("ðŸ“ˆ Advanced Visualizations")
        
        if results and len(results) > 0:
            fig_heatmap, fig_risk, fig_dist = create_advanced_visualizations(
                st.session_state.detector, results, algorithm
            )
            
            # Interactive similarity matrix
            st.subheader("ðŸ”¥ Interactive Similarity Heatmap")
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("ðŸŽ¯ Risk Level Distribution")
                st.plotly_chart(fig_risk, use_container_width=True)
            
            with col2:
                st.subheader("ðŸ“Š Similarity Score Distribution")
                st.plotly_chart(fig_dist, use_container_width=True)
            
            # Network graph for high similarities
            st.subheader("ðŸ•¸ï¸ Similarity Network")
            high_sim_results = [r for r in results if r['Similarity'] >= threshold]
            
            if high_sim_results:
                G = nx.Graph()
                
                # Add nodes
                for filename in st.session_state.detector.files.keys():
                    G.add_node(filename)
                
                # Add edges
                for result in high_sim_results:
                    G.add_edge(result['File 1'], result['File 2'], 
                             weight=result['Similarity'],
                             similarity=result['Percentage'])
                
                # Create network visualization
                pos = nx.spring_layout(G, k=3, iterations=50)
                
                edge_x = []
                edge_y = []
                edge_info = []
                
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                    edge_info.append(f"{edge[0]} â†” {edge[1]}: {G[edge[0]][edge[1]]['similarity']}")
                
                edge_trace = go.Scatter(x=edge_x, y=edge_y,
                                      line=dict(width=2, color='red'),
                                      hoverinfo='none',
                                      mode='lines')
                
                node_x = []
                node_y = []
                node_text = []
                
                for node in G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    node_text.append(node)
                
                node_trace = go.Scatter(x=node_x, y=node_y,
                                      mode='markers+text',
                                      hoverinfo='text',
                                      text=node_text,
                                      textposition="middle center",
                                      marker=dict(size=50,
                                                color='lightblue',
                                                line=dict(width=2, color='black')))
                
                fig_network = go.Figure(data=[edge_trace, node_trace],
                                      layout=go.Layout(
                                        title='Similarity Network (Above Threshold)',
                                        titlefont_size=16,
                                        showlegend=False,
                                        hovermode='closest',
                                        margin=dict(b=20,l=5,r=5,t=40),
                                        annotations=[ dict(
                                            text="Connections show similarities above threshold",
                                            showarrow=False,
                                            xref="paper", yref="paper",
                                            x=0.005, y=-0.002 ) ],
                                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
                
                st.plotly_chart(fig_network, use_container_width=True)
            else:
                st.info("No similarities above threshold to display in network.")
        
        else:
            st.info("No data available for visualization.")
    
    with tab3:
        st.header("ðŸ” Side-by-Side Code Comparison")
        
        if file_count >= 2:
            filenames = list(st.session_state.detector.files.keys())
            
            col1, col2 = st.columns(2)
            with col1:
                file1 = st.selectbox("Select First File", filenames, key="compare_file1")
            with col2:
                file2 = st.selectbox("Select Second File", filenames, key="compare_file2")
            
            if file1 != file2:
                # Calculate similarity with multiple algorithms
                st.subheader("ðŸŽ¯ Multi-Algorithm Analysis")
                
                similarities = {}
                for alg in st.session_state.detector.algorithms[:6]:  # Top 6 algorithms
                    sim = st.session_state.detector.calculate_similarity(file1, file2, alg)
                    similarities[alg] = sim
                
                # Display similarities in a nice format
                cols = st.columns(3)
                for i, (alg, sim) in enumerate(similarities.items()):
                    with cols[i % 3]:
                        color = "ðŸ”´" if sim >= 0.8 else "ðŸŸ¡" if sim >= 0.6 else "ðŸŸ¢"
                        st.metric(f"{color} {alg}", f"{sim:.3f}", f"{sim*100:.1f}%")
                
                # File metadata comparison
                st.subheader("ðŸ“Š File Metadata Comparison")
                meta1 = st.session_state.detector.file_metadata[file1]
                meta2 = st.session_state.detector.file_metadata[file2]
                
                comparison_df = pd.DataFrame({
                    file1: [meta1['size'], meta1['lines'], meta1['functions'], meta1['complexity']],
                    file2: [meta2['size'], meta2['lines'], meta2['functions'], meta2['complexity']]
                }, index=['Size (bytes)', 'Lines of Code', 'Functions', 'Complexity'])
                
                st.dataframe(comparison_df, use_container_width=True)
                
                # Side-by-side code display
                st.subheader("ðŸ“„ Code Comparison")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**ðŸ“„ {file1}**")
                    st.code(st.session_state.detector.files[file1], language='cpp', line_numbers=True)
                
                with col2:
                    st.markdown(f"**ðŸ“„ {file2}**")
                    st.code(st.session_state.detector.files[file2], language='cpp', line_numbers=True)
                
                # Detailed diff analysis
                st.subheader("ðŸ” Detailed Difference Analysis")
                
                code1_lines = st.session_state.detector.files[file1].splitlines()
                code2_lines = st.session_state.detector.files[file2].splitlines()
                
                diff = list(difflib.unified_diff(code1_lines, code2_lines, 
                                               fromfile=file1, tofile=file2, lineterm=''))
                
                if diff:
                    diff_text = '\n'.join(diff)
                    st.code(diff_text, language='diff')
                else:
                    st.success("âœ… Files are identical!")
            
            else:
                st.warning("âš ï¸ Please select different files to compare.")
    
    with tab4:
        st.header("ðŸ“‹ Detailed Analysis Report")
        
        if results:
            report = generate_detailed_report(
                st.session_state.detector, results, algorithm, threshold
            )
            
            st.markdown(report)
            
            # Advanced statistics
            st.subheader("ðŸ“ˆ Statistical Analysis")
            
            similarities = [r['Similarity'] for r in results]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Similarity", f"{np.mean(similarities):.3f}")
            with col2:
                st.metric("Std Deviation", f"{np.std(similarities):.3f}")
            with col3:
                st.metric("Median Similarity", f"{np.median(similarities):.3f}")
            
            # Percentile analysis
            percentiles = [25, 50, 75, 90, 95, 99]
            perc_values = [np.percentile(similarities, p) for p in percentiles]
            
            perc_df = pd.DataFrame({
                'Percentile': [f"{p}th" for p in percentiles],
                'Similarity Score': [f"{v:.3f}" for v in perc_values]
            })
            
            st.subheader("ðŸ“Š Percentile Analysis")
            st.dataframe(perc_df, use_container_width=True)
        
        else:
            st.info("No results available for detailed report.")
    
    with tab5:
        st.header("ðŸ“ File Explorer & Metadata")
        
        if st.session_state.detector.files:
            for filename, content in st.session_state.detector.files.items():
                with st.expander(f"ðŸ“„ {filename}"):
                    metadata = st.session_state.detector.file_metadata[filename]
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown("**ðŸ“Š File Statistics:**")
                        st.write(f"â€¢ Size: {metadata['size']} bytes")
                        st.write(f"â€¢ Lines: {metadata['lines']}")
                        st.write(f"â€¢ Functions: {metadata['functions']}")
                        st.write(f"â€¢ Complexity: {metadata['complexity']}")
                        st.write(f"â€¢ Includes: {len(metadata['includes'])}")
                        st.write(f"â€¢ Uploaded: {metadata['upload_time'].strftime('%H:%M:%S')}")
                        
                        if metadata['includes']:
                            st.markdown("**ðŸ“š Includes:**")
                            for include in metadata['includes']:
                                st.code(include, language='cpp')
                    
                                with col2:
                                    st.markdown("**ðŸ“„ Code Preview:**")
                                    preview_lines = content.split('\n')[:20]
                                    preview = '\n'.join(preview_lines)
                                    if len(content.split('\n')) > 20:
                                        lines_count = len(content.split('\n')) - 20
                                        preview += f"\n... ({lines_count} more lines)"
                                    st.code(preview, language='cpp')

        else:
            st.info("No files uploaded yet.")
    
    with tab6:
        st.header("ðŸ“¥ Export & Download Results")
        
        if results:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("ðŸ“Š Export Options")
                
                # CSV Export
                df = pd.DataFrame(results)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="ðŸ“Š Download CSV Report",
                    data=csv,
                    file_name=f"plagiarism_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                # JSON Export
                import json
                json_data = json.dumps(results, indent=2, default=str)
                st.download_button(
                    label="ðŸ“‹ Download JSON Report",
                    data=json_data,
                    file_name=f"plagiarism_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
                
                # Detailed Report
                detailed_report = generate_detailed_report(
                    st.session_state.detector, results, algorithm, threshold
                )
                st.download_button(
                    label="ðŸ“„ Download Detailed Report",
                    data=detailed_report,
                    file_name=f"detailed_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
            
            with col2:
                st.subheader("ðŸ“ˆ Export Settings")
                
                include_metadata = st.checkbox("Include file metadata", value=True)
                include_code_snippets = st.checkbox("Include code snippets", value=False)
                only_suspicious = st.checkbox("Only suspicious pairs", value=False)
                
                if st.button("ðŸŽ¯ Generate Custom Report"):
                    custom_results = results
                    if only_suspicious:
                        custom_results = [r for r in results if r['Status'] == 'Suspicious']
                    
                    custom_report = f"""
# Custom Plagiarism Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Algorithm: {algorithm}
Threshold: {threshold}

## Results Summary
Total Comparisons: {len(custom_results)}
"""
                    
                    for result in custom_results:
                        custom_report += f"""
### {result['File 1']} â†” {result['File 2']}
- Similarity: {result['Percentage']}
- Risk Level: {result['Risk Level']}
- Status: {result['Status']}
"""
                        
                        if include_code_snippets:
                            custom_report += f"""
#### Code Snippets:
**{result['File 1']}:**

**{result['File 2']}:**

"""
                    
                    st.download_button(
                        label="ðŸ“‹ Download Custom Report",
                        data=custom_report,
                        file_name=f"custom_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
        
        else:
            st.info("No results available for export.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <h3>âš¡ Competitive Programming Plagiarism Checker</h3>
        <p>Advanced C++ code similarity detection with 12 sophisticated algorithms</p>
        <p>Built for competitive programming â€¢ Optimized for accuracy â€¢ Enhanced UI</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
