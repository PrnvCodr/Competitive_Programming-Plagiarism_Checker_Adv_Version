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
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for beautiful UI
st.markdown("""
<style>
/* Import Google Fonts - via @import in CSS */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

/* Base font and background */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen,
      Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif !important;
}

.main .block-container {
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    min-height: 100vh;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Main header styling */
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 3rem 2rem;
    border-radius: 20px;
    color: white;
    text-align: center;
    margin-bottom: 3rem;
    box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.main-header h1 {
    font-size: 3.5rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.02em;
    text-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

.main-header p {
    font-size: 1.4rem;
    font-weight: 400;
    margin-top: 1rem;
    opacity: 0.95;
    letter-spacing: 0.01em;
}

/* Enhanced metric card styling */
.metric-card {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    padding: 2rem 2.5rem;
    border-radius: 18px;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.08);
    border-left: 6px solid #667eea;
    color: #1a202c;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    border: 1px solid rgba(102, 126, 234, 0.1);
    position: relative;
    overflow: hidden;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #667eea, #764ba2);
}

.metric-card:hover {
    transform: translateY(-8px) scale(1.02);
    box-shadow: 0 25px 50px rgba(102, 126, 234, 0.25);
}

.metric-card h3 {
    font-size: 0.9rem;
    font-weight: 600;
    color: #64748b;
    margin: 0 0 0.5rem 0;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.metric-card h2 {
    font-size: 2.5rem;
    font-weight: 700;
    color: #1e293b;
    margin: 0;
}

/* Suspicious and safe cards with enhanced styling */
.suspicious-card {
    background: linear-gradient(135deg, #fef2f2 0%, #fecaca 20%, #fff5f5 100%);
    border-left: 6px solid #dc2626;
    padding: 2rem 2.5rem;
    border-radius: 18px;
    margin: 1rem 0;
    color: #7f1d1d;
    box-shadow: 0 10px 25px rgba(220, 38, 38, 0.15);
    font-weight: 600;
    border: 1px solid rgba(220, 38, 38, 0.2);
}

.safe-card {
    background: linear-gradient(135deg, #f0fdf4 0%, #bbf7d0 20%, #f0fff4 100%);
    border-left: 6px solid #16a34a;
    padding: 2rem 2.5rem;
    border-radius: 18px;
    margin: 1rem 0;
    color: #14532d;
    box-shadow: 0 10px 25px rgba(22, 163, 74, 0.15);
    font-weight: 600;
    border: 1px solid rgba(22, 163, 74, 0.2);
}

/* Enhanced tabs styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(255, 255, 255, 0.8);
    padding: 0.5rem;
    border-radius: 16px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 12px;
    padding: 1rem 2rem;
    color: #64748b !important;
    font-weight: 600;
    font-size: 1.1rem;
    transition: all 0.3s ease;
    border: 2px solid transparent;
    position: relative;
}

.stTabs [data-baseweb="tab"]:hover {
    background: rgba(102, 126, 234, 0.1);
    color: #667eea !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    font-weight: 700;
    box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
    transform: translateY(-2px);
}

/* Enhanced button styling */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white !important;
    border: none;
    border-radius: 14px;
    padding: 1rem 2.5rem;
    font-weight: 700;
    font-size: 1.1rem;
    cursor: pointer;
    box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    text-transform: none;
    letter-spacing: 0.02em;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
    box-shadow: 0 12px 30px rgba(90, 103, 216, 0.6);
    transform: translateY(-3px);
}

/* Enhanced download button styling */
.stDownloadButton > button {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    color: white !important;
    border: none;
    border-radius: 14px !important;
    padding: 1rem 2.5rem;
    font-weight: 700;
    font-size: 1.1rem;
    cursor: pointer;
    box-shadow: 0 8px 20px rgba(16, 185, 129, 0.4);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
    box-shadow: 0 12px 30px rgba(5, 150, 105, 0.6);
    transform: translateY(-3px);
}

/* Enhanced dataframe styling */
.stDataFrame {
    background: white !important;
    border-radius: 16px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
    padding: 1.5rem !important;
    margin: 1.5rem 0;
    border: 1px solid #e2e8f0;
}

.stDataFrame > div {
    background: white !important;
    border-radius: 12px;
}

.stDataFrame table {
    background: white !important;
    color: #334155 !important;
    font-size: 1rem;
    border-collapse: separate !important;
    border-spacing: 0 !important;
}

.stDataFrame th {
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%) !important;
    color: #475569 !important;
    font-weight: 700;
    border: none !important;
    padding: 1.2rem 1.5rem !important;
    text-align: left;
    font-size: 0.95rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.stDataFrame th:first-child {
    border-radius: 12px 0 0 0;
}

.stDataFrame th:last-child {
    border-radius: 0 12px 0 0;
}

.stDataFrame td {
    background: white !important;
    color: #334155 !important;
    border-bottom: 1px solid #f1f5f9 !important;
    padding: 1.2rem 1.5rem !important;
    font-size: 0.95rem;
    vertical-align: middle;
}

.stDataFrame tbody tr:hover td {
    background: #f8fafc !important;
}

/* Enhanced sidebar styling */
.css-1d391kg {
    background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
    border-right: 1px solid #e2e8f0;
    box-shadow: 4px 0 12px rgba(0, 0, 0, 0.05);
}

/* Enhanced expander styling */
.streamlit-expanderContent {
    background: white !important;
    color: #334155 !important;
    border-radius: 0 0 16px 16px;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
    border: 1px solid #e2e8f0;
    padding: 2rem !important;
}

/* Enhanced code block styling */
.stCodeBlock {
    background: #f8fafc !important;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    padding: 1.5rem !important;
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace !important;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.05);
}

/* Enhanced metric container styling */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%) !important;
    border: 1px solid #e2e8f0 !important;
    padding: 2rem !important;
    border-radius: 18px !important;
    color: #1e293b !important;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.08) !important;
    transition: all 0.3s ease !important;
}

[data-testid="metric-container"]:hover {
    transform: translateY(-4px);
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.12) !important;
}

[data-testid="metric-container"] > div {
    color: #1e293b !important;
    font-weight: 600;
}

[data-testid="metric-container"] [data-testid="metric-value"] {
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    color: #0f172a !important;
}

[data-testid="metric-container"] [data-testid="metric-label"] {
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    color: #64748b !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

/* Enhanced selectbox and input styling */
.stSelectbox > div > div {
    background: white;
    border-radius: 12px;
    border: 2px solid #e2e8f0;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    transition: all 0.3s ease;
}

.stSelectbox > div > div:focus-within {
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}

/* File uploader styling */
.stFileUploader {
    border-radius: 16px;
    border: 2px dashed #cbd5e1;
    background: rgba(248, 250, 252, 0.5);
    padding: 2rem;
    transition: all 0.3s ease;
}

.stFileUploader:hover {
    border-color: #667eea;
    background: rgba(102, 126, 234, 0.05);
}

/* Footer styling */
.footer {
    text-align: center;
    padding: 3rem 2rem;
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    border-radius: 20px;
    margin-top: 3rem;
    color: #64748b;
    border: 1px solid #e2e8f0;
}

.footer h3 {
    font-size: 1.8rem;
    font-weight: 700;
    color: #334155;
    margin-bottom: 1rem;
}

.footer p {
    font-size: 1.1rem;
    line-height: 1.6;
    margin: 0.5rem 0;
}

/* Responsive design */
@media (max-width: 768px) {
    .main-header h1 {
        font-size: 2.5rem;
    }
    
    .main-header {
        padding: 2rem 1.5rem;
    }
    
    .metric-card {
        padding: 1.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.8rem 1.5rem;
        font-size: 1rem;
    }
}

/* Animations */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.metric-card, .suspicious-card, .safe-card {
    animation: fadeInUp 0.6s ease-out;
}
</style>
""", unsafe_allow_html=True)

# Your existing CompetitiveProgrammingPlagiarismDetector class
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
            fingerprints.add((min_hash, min_hash[1]))
        
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
        """Compare control flow structures - FIXED BUG"""
        if not cf1 or not cf2:
            return 0
        
        # Extract just the structure types - FIXED
        struct1 = [item[0] for item in cf1]
        struct2 = [item for item in cf2]  # FIXED: was [item for item in cf2]
        
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

    def cosine_similarity_method(self, code1, code2):
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
                    risk = "High Risk"
                elif similarity >= 0.6:
                    risk = "Medium Risk"
                elif similarity >= threshold:
                    risk = "Low Risk"
                else:
                    risk = "Safe"
                
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

# Enhanced visualization functions
def create_enhanced_similarity_matrix(detector, results, algorithm):
    """Create enhanced interactive similarity matrix with annotations"""
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
    
    # Create annotations matrix for text display
    annotation_matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append("SAME")
            else:
                row.append(f"{matrix[i][j]:.3f}")
        annotation_matrix.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=filenames,
        y=filenames,
        colorscale=[
            [0, '#2d5016'],      # Dark green for low similarity
            [0.3, '#65a30d'],    # Medium green
            [0.5, '#facc15'],    # Yellow for medium similarity
            [0.7, '#f97316'],    # Orange for high similarity
            [1, '#dc2626']       # Red for very high similarity
        ],
        text=annotation_matrix,
        texttemplate="%{text}",
        textfont={"size": 12, "color": "white"},
        hoverongaps=False,
        hovertemplate='<b>%{y}</b> vs <b>%{x}</b><br>Similarity: %{z:.3f}<extra></extra>'
    ))
    
    # FIXED: Use correct title format
    fig.update_layout(
        title=dict(text=f"📊 Enhanced Similarity Matrix - {algorithm}", font=dict(size=16)),
        xaxis_title="Files",
        yaxis_title="Files",
        margin=dict(t=80, b=40, l=40, r=40)
    )
    
    return fig

def create_risk_gauges(results):
    """Create individual gauge charts for top risky pairs"""
    top_risks = sorted(results, key=lambda x: x['Similarity'], reverse=True)[:6]
    
    if len(top_risks) < 6:
        # Fill with empty data if we have fewer than 6 results
        while len(top_risks) < 6:
            top_risks.append({
                'File 1': 'No Data',
                'File 2': 'No Data',
                'Similarity': 0,
                'Percentage': '0.00%'
            })
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[f"{r['File 1'][:10]}... vs {r['File 2'][:10]}..." for r in top_risks],
        specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
               [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]]
    )
    
    positions = [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3)]
    
    for i, (result, pos) in enumerate(zip(top_risks, positions)):
        similarity_percent = result['Similarity'] * 100
        
        # Determine color based on risk
        if similarity_percent >= 80:
            color = "red"
        elif similarity_percent >= 60:
            color = "orange"
        elif similarity_percent >= 40:
            color = "yellow"
        else:
            color = "green"
        
        fig.add_trace(go.Indicator(
            mode = "gauge+number",
            value = similarity_percent,
            title = {'text': f"Similarity %"},
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 40], 'color': "lightgray"},
                    {'range': [40, 60], 'color': "lightyellow"},
                    {'range': [60, 80], 'color': "lightcoral"},
                    {'range': [80, 100], 'color': "lightpink"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ), row=pos[0], col=pos[1])
    
    fig.update_layout(
        height=500, 
        title=dict(text="⚡ Risk Assessment Gauges", font=dict(size=16)),
        margin=dict(t=80, b=40, l=40, r=40)
    )
    return fig

def create_file_complexity_radar(detector):
    """Create radar chart showing file complexity metrics"""
    files = list(detector.files.keys())
    
    fig = go.Figure()
    
    for filename in files[:5]:  # Show top 5 files
        metadata = detector.file_metadata[filename]
        
        # Normalize metrics for radar chart
        max_lines = max([m['lines'] for m in detector.file_metadata.values()])
        max_functions = max([m['functions'] for m in detector.file_metadata.values()])
        max_complexity = max([m['complexity'] for m in detector.file_metadata.values()])
        max_size = max([m['size'] for m in detector.file_metadata.values()])
        max_includes = max([len(m['includes']) for m in detector.file_metadata.values()])
        
        # Normalize to 0-100 scale
        normalized_metrics = [
            (metadata['lines'] / max_lines * 100) if max_lines > 0 else 0,
            (metadata['functions'] / max_functions * 100) if max_functions > 0 else 0,
            (metadata['complexity'] / max_complexity * 100) if max_complexity > 0 else 0,
            (metadata['size'] / max_size * 100) if max_size > 0 else 0,
            (len(metadata['includes']) / max_includes * 100) if max_includes > 0 else 0
        ]
        
        fig.add_trace(go.Scatterpolar(
            r=normalized_metrics + [normalized_metrics[0]],  # Close the shape
            theta=['Lines of Code', 'Functions', 'Complexity', 'File Size', 'Includes', 'Lines of Code'],
            fill='toself',
            name=filename[:15] + "..." if len(filename) > 15 else filename,
            opacity=0.7
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title=dict(text="🎯 File Complexity Analysis", font=dict(size=16)),
        margin=dict(t=80, b=40, l=40, r=40)
    )
    
    return fig

def create_similarity_trends(results):
    """Create similarity trend analysis"""
    # Sort results by average file size or alphabetically
    sorted_results = sorted(results, key=lambda x: (x['File 1'], x['File 2']))
    
    x_labels = [f"{r['File 1'][:8]}...vs...{r['File 2'][:8]}..." for r in sorted_results]
    similarities = [r['Similarity'] * 100 for r in sorted_results]
    
    # Create colors based on risk levels
    colors = []
    for sim in similarities:
        if sim >= 80:
            colors.append('red')
        elif sim >= 60:
            colors.append('orange')
        elif sim >= 40:
            colors.append('yellow')
        else:
            colors.append('green')
    
    fig = go.Figure()
    
    # Add line plot
    fig.add_trace(go.Scatter(
        x=list(range(len(x_labels))),
        y=similarities,
        mode='lines+markers',
        name='Similarity Trend',
        line=dict(color='blue', width=2),
        marker=dict(size=8, color=colors, line=dict(color='black', width=1))
    ))
    
    # Add threshold line
    fig.add_hline(y=50, line_dash="dash", line_color="red", 
                  annotation_text="Threshold (50%)")
    
    fig.update_layout(
        title=dict(text="📈 Similarity Trend Analysis", font=dict(size=16)),
        xaxis_title="File Pairs",
        yaxis_title="Similarity Percentage",
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(x_labels))),
            ticktext=x_labels,
            tickangle=45
        ),
        height=500,
        margin=dict(t=80, b=100, l=40, r=40)
    )
    
    return fig

def create_algorithm_comparison_sunburst(detector, file_pairs):
    """Create sunburst chart comparing algorithm results"""
    algorithms = detector.algorithms[:6]  # Top 6 algorithms
    
    data = []
    
    # Calculate average similarities for each algorithm
    for algorithm in algorithms:
        avg_similarity = 0
        count = 0
        
        for file1, file2 in file_pairs:
            similarity = detector.calculate_similarity(file1, file2, algorithm)
            avg_similarity += similarity
            count += 1
        
        avg_similarity = avg_similarity / count if count > 0 else 0
        
        # Categorize performance
        if avg_similarity >= 0.7:
            category = "High Detection"
        elif avg_similarity >= 0.4:
            category = "Medium Detection"
        else:
            category = "Low Detection"
        
        data.append({
            'ids': algorithm,
            'labels': algorithm,
            'parents': "",
            'values': avg_similarity * 100
        })
        
        data.append({
            'ids': f"{algorithm}-{category}",
            'labels': f"{avg_similarity*100:.1f}%",
            'parents': algorithm,
            'values': avg_similarity * 100
        })
    
    fig = go.Figure(go.Sunburst(
        ids=[d['ids'] for d in data],
        labels=[d['labels'] for d in data],
        parents=[d['parents'] for d in data],
        values=[d['values'] for d in data],
        branchvalues="total"
    ))
    
    fig.update_layout(
        title=dict(text="☀️ Algorithm Performance Sunburst", font=dict(size=16)),
        height=600,
        margin=dict(t=80, b=40, l=40, r=40)
    )
    
    return fig

def create_statistics_dashboard(detector, results):
    """Create comprehensive statistics dashboard"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['File Size Distribution', 'Complexity vs Lines', 
                       'Risk Level Pie Chart', 'Similarity Distribution'],
        specs=[[{"type": "histogram"}, {"type": "scatter"}],
               [{"type": "pie"}, {"type": "histogram"}]]
    )
    
    # File size distribution
    file_sizes = [metadata['size'] for metadata in detector.file_metadata.values()]
    fig.add_trace(go.Histogram(x=file_sizes, name="File Sizes"), row=1, col=1)
    
    # Complexity vs Lines scatter
    complexities = [metadata['complexity'] for metadata in detector.file_metadata.values()]
    lines = [metadata['lines'] for metadata in detector.file_metadata.values()]
    filenames = list(detector.file_metadata.keys())
    
    fig.add_trace(go.Scatter(
        x=lines, y=complexities, mode='markers+text',
        text=[f[:8] + "..." if len(f) > 8 else f for f in filenames],
        textposition="top center",
        name="Complexity vs Lines"
    ), row=1, col=2)
    
    # Risk level pie chart
    risk_counts = {}
    for result in results:
        risk = result['Risk Level']
        risk_counts[risk] = risk_counts.get(risk, 0) + 1
    
    fig.add_trace(go.Pie(
        labels=list(risk_counts.keys()),
        values=list(risk_counts.values()),
        name="Risk Levels"
    ), row=2, col=1)
    
    # Similarity distribution
    similarities = [r['Similarity'] * 100 for r in results]
    fig.add_trace(go.Histogram(x=similarities, name="Similarities"), row=2, col=2)
    
    fig.update_layout(
        height=700, 
        title=dict(text="📊 Comprehensive Statistics Dashboard", font=dict(size=16)),
        margin=dict(t=80, b=40, l=40, r=40)
    )
    
    return fig

def create_enhanced_metrics(results):
    """Enhanced metric cards with visual indicators"""
    suspicious_count = sum(1 for r in results if r['Status'] == 'Suspicious')
    avg_similarity = np.mean([r['Similarity'] for r in results]) if results else 0
    max_similarity = max([r['Similarity'] for r in results]) if results else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Total Comparisons</h3>
            <h2>{len(results)}</h2>
            <div style="width: 100%; background-color: #e2e8f0; border-radius: 10px; height: 8px; margin-top: 10px;">
                <div style="width: 100%; background: linear-gradient(90deg, #667eea, #764ba2); height: 8px; border-radius: 10px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        danger_width = (suspicious_count / len(results)) * 100 if results else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3>⚠️ Suspicious Pairs</h3>
            <h2>{suspicious_count}</h2>
            <div style="width: 100%; background-color: #e2e8f0; border-radius: 10px; height: 8px; margin-top: 10px;">
                <div style="width: {danger_width}%; background: linear-gradient(90deg, #dc2626, #ef4444); height: 8px; border-radius: 10px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_width = avg_similarity * 100
        st.markdown(f"""
        <div class="metric-card">
            <h3>📈 Average Similarity</h3>
            <h2>{avg_similarity * 100:.1f}%</h2>
            <div style="width: 100%; background-color: #e2e8f0; border-radius: 10px; height: 8px; margin-top: 10px;">
                <div style="width: {avg_width}%; background: linear-gradient(90deg, #10b981, #059669); height: 8px; border-radius: 10px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        max_width = max_similarity * 100
        st.markdown(f"""
        <div class="metric-card">
            <h3>🔥 Highest Similarity</h3>
            <h2>{max_similarity * 100:.1f}%</h2>
            <div style="width: 100%; background-color: #e2e8f0; border-radius: 10px; height: 8px; margin-top: 10px;">
                <div style="width: {max_width}%; background: linear-gradient(90deg, #f59e0b, #d97706); height: 8px; border-radius: 10px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

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
    fig_heatmap.update_layout(margin=dict(t=80, b=40, l=40, r=40))
    
    # 2. Risk Distribution
    risk_counts = Counter([r['Risk Level'] for r in results])
    fig_risk = px.pie(
        values=list(risk_counts.values()),
        names=list(risk_counts.keys()),
        title="Risk Level Distribution",
        color_discrete_map={
            "High Risk": "#e53e3e",
            "Medium Risk": "#d69e2e", 
            "Low Risk": "#dd6b20",
            "Safe": "#38a169"
        }
    )
    fig_risk.update_layout(margin=dict(t=80, b=40, l=40, r=40))
    
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
    fig_dist.update_layout(margin=dict(t=80, b=40, l=40, r=40))
    
    return fig_heatmap, fig_risk, fig_dist

def generate_detailed_report(detector, results, algorithm, threshold):
    """Generate detailed analysis report"""
    total_files = len(detector.files)
    total_comparisons = len(results)
    suspicious_pairs = sum(1 for r in results if r['Status'] == 'Suspicious')
    
    report = f"""
# Competitive Programming Plagiarism Analysis Report

## Executive Summary
- **Analysis Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Algorithm Used**: {algorithm}
- **Detection Threshold**: {threshold:.2f}
- **Files Analyzed**: {total_files}
- **Total Comparisons**: {total_comparisons}
- **Suspicious Pairs**: {suspicious_pairs}
- **Detection Rate**: {(suspicious_pairs/total_comparisons*100):.1f}%

## File Analysis Details
"""
    
    for filename, metadata in detector.file_metadata.items():
        report += f"""
### {filename}
- **Size**: {metadata['size']} bytes
- **Lines of Code**: {metadata['lines']}
- **Functions**: {metadata['functions']}
- **Complexity Score**: {metadata['complexity']}
- **Includes**: {len(metadata['includes'])}
"""
    
    report += "\n## High-Risk Comparisons\n"
    
    high_risk = [r for r in results if r['Similarity'] >= 0.8]
    for result in high_risk[:10]:  # Top 10
        report += f"- **{result['File 1']}** → **{result['File 2']}**: {result['Percentage']} similarity\n"
    
    return report

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Competitive Programming Plagiarism Checker</h1>
        <p>Advanced C++ Code Similarity Detection for Competitive Programming</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize detector
    if 'detector' not in st.session_state:
        st.session_state.detector = CompetitiveProgrammingPlagiarismDetector()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration Panel")
        
        # File upload section
        st.subheader("📁 Upload C++ Files")
        uploaded_files = st.file_uploader(
            "Choose C++ files",
            type=['cpp', 'cc', 'cxx', 'c'],
            accept_multiple_files=True,
            help="Upload multiple C++ source files for comparison"
        )
        
        # Bulk upload option
        st.subheader("📦 Bulk Upload (ZIP)")
        zip_file = st.file_uploader(
            "Upload ZIP file containing C++ files",
            type=['zip'],
            help="Upload a ZIP file containing multiple C++ files"
        )
        
        # Algorithm selection
        st.subheader("🔍 Detection Algorithm")
        algorithm = st.selectbox(
            "Select Algorithm",
            st.session_state.detector.algorithms,
            index=0,
            help="Choose the plagiarism detection algorithm"
        )
        
        # Advanced settings
        st.subheader("⚙️ Advanced Settings")
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
        st.subheader("🎛️ Analysis Options")
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
        
        # Process all files with progress bar
        processed_count = 0
        skipped_count = 0
        
        if files_to_process:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, uploaded_file in enumerate(files_to_process):
                try:
                    progress = (idx + 1) / len(files_to_process)
                    progress_bar.progress(progress)
                    status_text.text(f'Processing {uploaded_file.name}...')
                    
                    if hasattr(uploaded_file, 'read'):
                        content = uploaded_file.read().decode('utf-8', errors='ignore')
                    else:
                        content = uploaded_file.content
                    
                    file_size = len(content.encode('utf-8'))
                    
                    # File validation
                    if file_size < min_file_size:
                        st.sidebar.warning(f"Warning: {uploaded_file.name} too small ({file_size} bytes)")
                        skipped_count += 1
                        continue
                    elif file_size > max_file_size * 1024:
                        st.sidebar.warning(f"Warning: {uploaded_file.name} too large ({file_size/1024:.1f} KB)")
                        skipped_count += 1
                        continue
                    elif len(content.strip()) == 0:
                        st.sidebar.warning(f"Warning: {uploaded_file.name} is empty")
                        skipped_count += 1
                        continue
                    
                    st.session_state.detector.add_file(uploaded_file.name, content, file_size)
                    processed_count += 1

                except Exception as e:
                    st.sidebar.error(f"Error processing {uploaded_file.name}: {str(e)}")
                    skipped_count += 1
            
            progress_bar.empty()
            status_text.empty()
    
    # Display file statistics
    file_count = len(st.session_state.detector.files)
    
    if file_count > 0:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Files Loaded", file_count)
        with col2:
            if 'processed_count' in locals():
                st.metric("Processed", processed_count)
        with col3:
            if 'skipped_count' in locals():
                st.metric("Skipped", skipped_count)
    
    if file_count < 2:
        st.info("Please upload at least 2 C++ files to start plagiarism detection.")
        
        # Show sample code for demonstration
        with st.expander("View Sample Analysis"):
            st.markdown("""
            ### Sample C++ Code Comparison
            
            **Features of this plagiarism checker:**
            - **12 Advanced Algorithms** including MOSS simulation
            - **Variable Renaming Detection** 
            - **Control Flow Analysis**
            - **Structure-based Comparison**
            - **Interactive Visualizations**
            - **Detailed Reports**
            - **Optimized for Competitive Programming**
            """)
        return
    
    # Main analysis tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Analysis Results", 
        "📈 Enhanced Visualizations", 
        "🔍 Code Comparison", 
        "📄 Detailed Report",
        "📂 File Explorer",
        "💾 Export Results"
    ])
    
    # Calculate results
    with st.spinner(f"Analyzing {file_count} files using {algorithm}..."):
        results = st.session_state.detector.compare_all_files(algorithm, threshold)
    
    with tab1:
        st.header("🎯 Plagiarism Analysis Results")
        
        if results:
            # Enhanced summary metrics
            create_enhanced_metrics(results)
            
            st.markdown("---")
            
            # Results table with enhanced styling
            df = pd.DataFrame(results)
            
            # Color coding function
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
            st.subheader("🚨 Most Suspicious Pairs")
            top_suspicious = [r for r in results if r['Similarity'] >= 0.7][:5]
            
            for result in top_suspicious:
                if result['Similarity'] >= 0.8:
                    card_class = "suspicious-card"
                else:
                    card_class = "safe-card"
                
                st.markdown(f"""
                <div class="{card_class}">
                    <h4>{result['File 1']} → {result['File 2']}</h4>
                    <p><strong>Similarity:</strong> {result['Percentage']} | <strong>Risk:</strong> {result['Risk Level']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        else:
            st.info("No results to display.")
    
    with tab2:
        st.header("📊 Enhanced Visualizations Dashboard")
        
        if results and len(results) > 0:
            # Enhanced similarity matrix
            st.subheader("🔥 Enhanced Interactive Similarity Matrix")
            fig_enhanced_matrix = create_enhanced_similarity_matrix(
                st.session_state.detector, results, algorithm
            )
            st.plotly_chart(fig_enhanced_matrix, use_container_width=True)
            
            # Risk assessment gauges
            st.subheader("⚡ Risk Assessment Gauges")
            fig_gauges = create_risk_gauges(results)
            st.plotly_chart(fig_gauges, use_container_width=True)
            
            # File complexity radar
            st.subheader("🎯 File Complexity Analysis")
            fig_radar = create_file_complexity_radar(st.session_state.detector)
            st.plotly_chart(fig_radar, use_container_width=True)
            
            # Similarity trends
            st.subheader("📈 Similarity Trend Analysis")
            fig_trends = create_similarity_trends(results)
            st.plotly_chart(fig_trends, use_container_width=True)
            
            # Algorithm comparison sunburst
            st.subheader("☀️ Algorithm Performance Sunburst")
            file_pairs = [(r['File 1'], r['File 2']) for r in results[:10]]  # Top 10 pairs
            fig_sunburst = create_algorithm_comparison_sunburst(
                st.session_state.detector, file_pairs
            )
            st.plotly_chart(fig_sunburst, use_container_width=True)
            
            # Statistics dashboard
            st.subheader("📊 Comprehensive Statistics Dashboard")
            fig_dashboard = create_statistics_dashboard(
                st.session_state.detector, results
            )
            st.plotly_chart(fig_dashboard, use_container_width=True)
            
            # Network graph for high similarities
            st.subheader("🕸️ Similarity Network Graph")
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
                    # FIXED: Correct way to access edge attributes in NetworkX
                    try:
                        similarity = G.edges[edge]['similarity']
                        edge_info.append(f"{edge[0]} → {edge[1]}: {similarity}")
                    except KeyError:
                        # Fallback if edge doesn't exist in expected direction
                        similarity = G.edges.get((edge[1], edge[0]), {}).get('similarity', "N/A")
                        edge_info.append(f"{edge[0]} → {edge[1]}: {similarity}")
                
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

                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode="markers+text",
                    text=node_text,
                    textposition="middle center",
                    hoverinfo="text",
                    marker=dict(size=20, color="lightblue",
                                line=dict(width=2, color="black"))
                )

                # FIXED: Use correct title format in layout
                fig_network = go.Figure(
                    data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=dict(text="🕸️ Similarity Network Graph", font=dict(size=16)),
                        showlegend=False,
                        hovermode="closest",
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="Nodes are files; edges show similarity ≥ threshold",
                            showarrow=False, xref="paper", yref="paper",
                            x=0.005, y=-0.002, xanchor="left", yanchor="bottom",
                            font=dict(size=12))],
                        xaxis=dict(showgrid=False, zeroline=False,
                                   showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False,
                                   showticklabels=False)
                    )
                )
                st.plotly_chart(fig_network, use_container_width=True)
            else:
                st.info("No high-similarity connections to plot.")

    # ------------- TAB 3 — CODE COMPARISON -----------------
    with tab3:
        st.header("🔍 Code Comparison")

        if len(st.session_state.detector.files) >= 2:
            c1, c2 = st.columns(2)
            with c1:
                file1 = st.selectbox("Select First File",
                                     list(st.session_state.detector.files.keys()),
                                     key="file1_select")
            with c2:
                file2 = st.selectbox("Select Second File",
                                     list(st.session_state.detector.files.keys()),
                                     key="file2_select")

            if file1 and file2 and file1 != file2:
                st.subheader(f"📋 {file1} ↔ {file2}")

                algo_rows = []
                for algo in st.session_state.detector.algorithms:
                    sim = st.session_state.detector.calculate_similarity(file1, file2, algo)
                    algo_rows.append(dict(
                        Algorithm=algo,
                        Similarity=f"{sim*100:.2f}%",
                        Risk=("High" if sim >= .8 else
                              "Medium" if sim >= .6 else
                              "Low" if sim >= .4 else "Safe")
                    ))
                st.dataframe(pd.DataFrame(algo_rows), use_container_width=True)

                colL, colR = st.columns(2)
                with colL:
                    st.subheader(file1)
                    st.code(st.session_state.detector.files[file1], language="cpp")
                with colR:
                    st.subheader(file2)
                    st.code(st.session_state.detector.files[file2], language="cpp")
        else:
            st.info("Upload at least two files to compare.")

    # ------------- TAB 4 — DETAILED REPORT -----------------
    with tab4:
        st.header("📄 Detailed Report")
        if results:
            md_report = generate_detailed_report(st.session_state.detector,
                                                 results, algorithm, threshold)
            st.markdown(md_report)
            st.download_button(
                "📥 Download Report (Markdown)",
                md_report,
                file_name=f"plagiarism_report_{datetime.now():%Y%m%d_%H%M%S}.md",
                mime="text/markdown")
        else:
            st.info("Run an analysis first.")

    # ------------- TAB 5 — FILE EXPLORER -------------------
    with tab5:
        st.header("📂 File Explorer")
        if st.session_state.detector.files:
            explorer_df = pd.DataFrame([
                dict(Filename=f,
                     Size=meta["size"],
                     Lines=meta["lines"],
                     Functions=meta["functions"],
                     Complexity=meta["complexity"],
                     Includes=len(meta["includes"]))
                for f, meta in st.session_state.detector.file_metadata.items()
            ])
            st.dataframe(explorer_df, use_container_width=True)

            view_file = st.selectbox("View file content",
                                     list(st.session_state.detector.files.keys()))
            st.code(st.session_state.detector.files[view_file], language="cpp")

            if st.button("🗑️ Clear All Files"):
                st.session_state.detector = CompetitiveProgrammingPlagiarismDetector()
                st.success("Cleared; reload to start fresh.")
                st.rerun()
        else:
            st.info("No files yet.")

    # ------------- TAB 6 — EXPORT RESULTS ------------------
    with tab6:
        st.header("💾 Export Results")
        if results:
            data_df = pd.DataFrame(results)
            st.download_button("📊 Download CSV",
                               data_df.to_csv(index=False),
                               file_name="results.csv",
                               mime="text/csv")
        else:
            st.info("Nothing to export yet.")

    # --------------------- FOOTER ---------------------------
    st.markdown("---")
    st.markdown(
        "<div class='footer'>© 2024 • Competitive Programming Plagiarism Checker</div>",
        unsafe_allow_html=True)


if __name__ == "__main__":
    main()
