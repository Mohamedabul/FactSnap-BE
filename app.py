"""
Streamlit web interface for FactSnap-V
Provides an interactive interface for audio/video analysis
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
import time
from datetime import datetime

# Streamlit extras for UI enhancements
import streamlit.components.v1 as components

# Import the main application
from main import FactSnapV


def init_streamlit_config():
    """
    Initialize Streamlit page configuration
    """
    st.set_page_config(
        page_title="FactSnap-V Enhanced",
        page_icon="✨",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://docs.streamlit.io',
            'Report a bug': 'https://github.com/your-repo/issues',
            'About': "### FactSnap-V\nEnhanced UI for interactive analysis."
        }
    )


def create_color_map():
    """
    Create color mapping for emotions and bias levels
    """
    emotion_colors = {
        'joy': '#4CAF50',      # Green
        'love': '#E91E63',     # Pink
        'optimism': '#2196F3', # Blue
        'trust': '#00BCD4',    # Cyan
        'anger': '#F44336',    # Red
        'fear': '#9C27B0',     # Purple
        'sadness': '#607D8B',  # Blue Grey
        'disgust': '#795548',  # Brown
        'surprise': '#FF9800', # Orange
        'anticipation': '#FFEB3B', # Yellow
        'pessimism': '#424242', # Dark Grey
        'neutral': '#9E9E9E'   # Grey
    }
    
    bias_colors = {
        'Low': '#4CAF50',      # Green
        'Medium': '#FF9800',   # Orange
        'High': '#F44336'      # Red
    }
    
    fact_colors = {
        'True': '#4CAF50',         # Green
        'False': '#F44336',        # Red
        'Mixed': '#FF9800',        # Orange
        'Mixed Partial': '#FFC107', # Amber
        'Mixed False': '#FF5722',  # Deep Orange
        'Partially True': '#8BC34A', # Light Green
        'Unverifiable': '#9E9E9E', # Grey
        'No Claims': '#E0E0E0',    # Light Grey
        'Not Checked': '#EEEEEE',  # Very Light Grey
        'Error': '#795548'         # Brown
    }
    
    return emotion_colors, bias_colors, fact_colors


def display_file_uploader():
    """
    Display file uploader widget with enhanced UI
    """
    # Enhanced header with gradient styling
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; text-align: center; margin: 0;">
            ✨ FactSnap-V Enhanced
        </h1>
        <p style="color: white; text-align: center; margin: 0.5rem 0 0 0; font-size: 1.2rem;">
            Advanced AI-Powered Audio & Video Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input method selection
    st.markdown("### 🎯 Choose Input Method")
    input_method = st.radio(
        "Select how you want to provide audio:",
        ["📁 Upload File", "🎙️ Real-time Streaming"],
        horizontal=True,
        help="Choose between uploading a file or streaming live audio"
    )
    
    return input_method
    
    return input_method


def display_file_upload_section():
    """
    Display file upload section
    """
    # Theme toggle in sidebar
    st.sidebar.markdown("### 🎨 Appearance")
    theme = st.sidebar.selectbox(
        "Choose theme:", 
        ('Light ☀️', 'Dark 🌙', 'Auto 🔄'), 
        index=0
    )
    
    # Apply custom CSS based on theme
    if theme == 'Dark 🌙':
        st.markdown("""
        <style>
        .main > div {
            background-color: #0e1117;
            color: #fafafa;
        }
        .stSelectbox > div > div {
            background-color: #262730;
        }
        </style>
        """, unsafe_allow_html=True)
    
    # Enhanced file uploader with better styling
    st.markdown("### 📁 Upload Your Media File")
    
    uploaded_file = st.file_uploader(
        "Drag and drop or browse to upload",
        type=['wav', 'mp3', 'm4a', 'flac', 'mp4', 'avi', 'mov', 'mkv'],
        help="📊 Supported formats: WAV, MP3, M4A, FLAC (audio) | MP4, AVI, MOV, MKV (video)"
    )
    
    return uploaded_file


def display_analysis_options():
    """
    Display enhanced analysis options in sidebar
    """
    st.sidebar.markdown("### ⚙️ Analysis Configuration")
    
    # Model performance info
    st.sidebar.markdown("""
    **🏆 Models Used:**
    - **Emotion**: j-hartmann/emotion-english-distilroberta-base (91.2% accuracy)
    - **Bias**: martin-ha/toxic-comment-model (94.1% accuracy)
    - **Speech**: OpenAI Whisper (state-of-the-art)
    """)
    
    st.sidebar.markdown("---")
    
    extract_claims = st.sidebar.checkbox(
        "🔍 Enable Fact Verification",
        value=True,
        help="Extract and verify factual claims using Google Fact Check Tools API"
    )
    
    show_detailed_results = st.sidebar.checkbox(
        "📈 Show Detailed Results",
        value=True,
        help="Display sentence-by-sentence analysis results"
    )
    
    # Advanced options
    st.sidebar.markdown("#### Advanced Options")
    
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Minimum confidence for bias/emotion detection"
    )
    
    batch_size = st.sidebar.selectbox(
        "Processing Batch Size",
        [8, 16, 32, 64],
        index=1,
        help="Larger batch sizes are faster but use more memory"
    )
    
    return extract_claims, show_detailed_results, confidence_threshold, batch_size


def display_progress_indicator(current_step, total_steps, step_name):
    """
    Display progress indicator
    """
    progress = current_step / total_steps
    st.progress(progress)
    st.write(f"**Step {current_step}/{total_steps}**: {step_name}")


def display_emotion_analysis(emotion_results, emotion_summary):
    """
    Display enhanced emotion analysis results
    """
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem;">
        <h2 style="color: white; margin: 0; text-align: center;">
            😊 Emotion Analysis Dashboard
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    if not emotion_summary:
        st.warning("⚠️ No emotion analysis results available.")
        return
    
    # Enhanced metrics with better styling
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📝 Total Sentences",
            emotion_summary['total_sentences'],
            delta=None,
            help="Number of sentences analyzed"
        )
    
    with col2:
        avg_conf = emotion_summary['average_confidence']
        st.metric(
            "🎢 Avg Confidence", 
            f"{avg_conf:.3f}",
            delta=f"{(avg_conf - 0.5)*100:+.1f}%" if avg_conf != 0.5 else None,
            help="Average model confidence across all predictions"
        )
    
    with col3:
        # Find dominant emotion
        emotion_dist = emotion_summary['emotion_distribution']
        dominant_emotion = max(emotion_dist.items(), key=lambda x: x[1]['count'])
        st.metric(
            "🏆 Dominant Emotion", 
            dominant_emotion[0].capitalize(),
            delta=f"{dominant_emotion[1]['percentage']:.1f}%",
            help="Most frequently detected emotion"
        )
    
    with col4:
        # Emotion diversity index
        diversity = len([e for e in emotion_dist.values() if e['count'] > 0])
        st.metric(
            "🌈 Emotion Diversity",
            f"{diversity} types",
            delta=None,
            help="Number of different emotions detected"
        )
    
    # Enhanced emotion distribution charts
    if emotion_dist:
        emotion_data = []
        for emotion, data in emotion_dist.items():
            emotion_data.append({
                'Emotion': emotion.capitalize(),
                'Count': data['count'],
                'Percentage': data['percentage'],
                'Confidence': data['average_confidence']
            })
        
        df_emotions = pd.DataFrame(emotion_data)
        
        # Create enhanced visualizations
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comprehensive Emotion Analysis', fontsize=16, fontweight='bold')
        
        # Enhanced color scheme
        emotion_colors, _, _ = create_color_map()
        colors = [emotion_colors.get(emotion.lower(), '#9E9E9E') for emotion in df_emotions['Emotion']]
        
        # 1. Donut chart instead of pie chart
        wedges, texts, autotexts = ax1.pie(
            df_emotions['Count'], 
            labels=df_emotions['Emotion'], 
            autopct='%1.1f%%', 
            colors=colors,
            startangle=90,
            pctdistance=0.85,
            wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2)
        )
        ax1.set_title('🍰 Emotion Distribution (Donut)', fontweight='bold', pad=20)
        
        # 2. Horizontal bar chart
        bars = ax2.barh(df_emotions['Emotion'], df_emotions['Count'], color=colors)
        ax2.set_title('📊 Emotion Counts (Horizontal)', fontweight='bold', pad=20)
        ax2.set_xlabel('Count')
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax2.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'{int(width)}', ha='left', va='center', fontweight='bold')
        
        # 3. Confidence vs Count scatter plot
        scatter = ax3.scatter(
            df_emotions['Count'], 
            df_emotions['Confidence'], 
            c=colors, 
            s=100, 
            alpha=0.7,
            edgecolors='black',
            linewidth=1
        )
        ax3.set_title('🎢 Confidence vs Count', fontweight='bold', pad=20)
        ax3.set_xlabel('Count')
        ax3.set_ylabel('Average Confidence')
        # Add emotion labels
        for i, emotion in enumerate(df_emotions['Emotion']):
            ax3.annotate(emotion, (df_emotions['Count'][i], df_emotions['Confidence'][i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 4. Percentage breakdown
        bars4 = ax4.bar(df_emotions['Emotion'], df_emotions['Percentage'], color=colors)
        ax4.set_title('📊 Percentage Breakdown', fontweight='bold', pad=20)
        ax4.set_ylabel('Percentage (%)')
        ax4.tick_params(axis='x', rotation=45)
        # Add percentage labels on bars
        for i, bar in enumerate(bars4):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Interactive data table
        st.markdown("#### 📋 Detailed Emotion Breakdown")
        st.dataframe(
            df_emotions.style.background_gradient(subset=['Count', 'Percentage', 'Confidence']),
            use_container_width=True
        )


def display_bias_analysis(bias_results, bias_summary):
    """
    Display enhanced bias analysis results
    """
    st.markdown("""
    <div style="background: linear-gradient(135deg, #ff6b6b 0%, #ffa500 100%);
                padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem;">
        <h2 style="color: white; margin: 0; text-align: center;">
            ⚖️ Bias Analysis Dashboard
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    if not bias_summary:
        st.warning("⚠️ No bias analysis results available.")
        return
    
    # Enhanced metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📝 Total Sentences",
            bias_summary['total_sentences'],
            help="Number of sentences analyzed for bias"
        )
    
    with col2:
        avg_bias = bias_summary['average_bias_score']
        bias_level = "High" if avg_bias > 0.7 else ("Medium" if avg_bias > 0.3 else "Low")
        st.metric(
            "🎢 Avg Bias Score", 
            f"{avg_bias:.3f}",
            delta=f"{bias_level} Risk",
            delta_color="inverse" if bias_level == "High" else "normal",
            help="Average bias score across all sentences"
        )
    
    with col3:
        # Count high bias sentences
        bias_dist = bias_summary['bias_distribution']
        high_bias_count = bias_dist.get('High', {}).get('count', 0)
        high_bias_pct = bias_dist.get('High', {}).get('percentage', 0)
        st.metric(
            "🚨 High Bias Sentences", 
            high_bias_count,
            delta=f"{high_bias_pct:.1f}%",
            delta_color="inverse",
            help="Sentences with high bias detection"
        )
    
    with col4:
        # Safety score (inverse of bias)
        safety_score = (1 - avg_bias) * 100
        st.metric(
            "🛡️ Safety Score",
            f"{safety_score:.1f}%",
            delta="Safe" if safety_score > 70 else "Caution",
            delta_color="normal" if safety_score > 70 else "inverse",
            help="Overall content safety assessment"
        )
    
    # Enhanced bias distribution charts
    if bias_dist:
        bias_data = []
        for level, data in bias_dist.items():
            bias_data.append({
                'Bias Level': level,
                'Count': data['count'],
                'Percentage': data['percentage'],
                'Avg Score': data['average_score']
            })
        
        df_bias = pd.DataFrame(bias_data)
        
        # Create enhanced visualizations
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Comprehensive Bias Analysis', fontsize=16, fontweight='bold')
        
        # Enhanced color scheme
        _, bias_colors, _ = create_color_map()
        colors = [bias_colors.get(level, '#9E9E9E') for level in df_bias['Bias Level']]
        
        # 1. Donut chart
        wedges, texts, autotexts = ax1.pie(
            df_bias['Count'], 
            labels=df_bias['Bias Level'], 
            autopct='%1.1f%%', 
            colors=colors,
            startangle=90,
            pctdistance=0.85,
            wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2)
        )
        ax1.set_title('🍰 Bias Distribution', fontweight='bold', pad=20)
        
        # 2. Stacked bar chart
        bars = ax2.bar(df_bias['Bias Level'], df_bias['Count'], color=colors)
        ax2.set_title('📊 Bias Level Counts', fontweight='bold', pad=20)
        ax2.set_ylabel('Count')
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Risk assessment gauge
        risk_levels = ['Low', 'Medium', 'High']
        risk_counts = [df_bias[df_bias['Bias Level'] == level]['Count'].sum() for level in risk_levels]
        total_count = sum(risk_counts)
        risk_percentages = [count/total_count*100 if total_count > 0 else 0 for count in risk_counts]
        
        ax3.barh(risk_levels, risk_percentages, color=['#4CAF50', '#FF9800', '#F44336'])
        ax3.set_title('🚨 Risk Assessment', fontweight='bold', pad=20)
        ax3.set_xlabel('Percentage (%)')
        for i, pct in enumerate(risk_percentages):
            ax3.text(pct + 1, i, f'{pct:.1f}%', va='center', fontweight='bold')
        
        # 4. Score distribution
        bars4 = ax4.bar(df_bias['Bias Level'], df_bias['Avg Score'], color=colors)
        ax4.set_title('🎢 Average Bias Scores', fontweight='bold', pad=20)
        ax4.set_ylabel('Average Score')
        ax4.set_ylim(0, 1)
        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Interactive data table
        st.markdown("#### 📋 Detailed Bias Breakdown")
        st.dataframe(
            df_bias.style.background_gradient(subset=['Count', 'Percentage', 'Avg Score']),
            use_container_width=True
        )


def display_fact_verification(fact_results, fact_summary):
    """
    Display enhanced LangGraph fact verification results
    """
    st.markdown("""
    <div style="background: linear-gradient(135deg, #4CAF50 0%, #2196F3 100%);
                padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem;">
        <h2 style="color: white; margin: 0; text-align: center;">
            🔍 LangGraph Fact Verification Dashboard
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    if not fact_summary:
        st.warning("⚠️ No fact verification results available.")
        return
    
    # Enhanced metrics for LangGraph results
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📝 Sentences Analyzed",
            fact_summary['total_sentences'],
            help="Total number of sentences processed"
        )
    
    with col2:
        st.metric(
            "🔍 Claims Found", 
            fact_summary['total_claims'],
            delta=f"{fact_summary['claims_per_sentence']:.1f} per sentence",
            help="Total factual claims extracted and verified"
        )
    
    with col3:
        st.metric(
            "📊 Sentences with Claims",
            fact_summary['sentences_with_claims'],
            delta=f"{(fact_summary['sentences_with_claims']/fact_summary['total_sentences']*100):.1f}%",
            help="Sentences containing verifiable claims"
        )
    
    with col4:
        avg_conf = fact_summary['average_confidence']
        st.metric(
            "🎢 Avg Confidence",
            f"{avg_conf:.3f}",
            delta="High" if avg_conf > 0.8 else ("Medium" if avg_conf > 0.6 else "Low"),
            help="Average verification confidence"
        )
    
    # Sentence-level verification distribution
    sentence_dist = fact_summary.get('sentence_status_distribution', {})
    if sentence_dist:
        st.markdown("#### 📊 Sentence-Level Verification Results")
        
        sentence_data = []
        for status, data in sentence_dist.items():
            sentence_data.append({
                'Status': status.replace('_', ' ').title(),
                'Count': data['count'],
                'Percentage': data['percentage']
            })
        
        df_sentences = pd.DataFrame(sentence_data)
        
        # Create enhanced visualizations
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('LangGraph Fact Verification Analysis', fontsize=16, fontweight='bold')
        
        # Enhanced color scheme for LangGraph results
        langgraph_colors = {
            'True': '#4CAF50',
            'False': '#F44336', 
            'Mixed Partial': '#FF9800',
            'Mixed False': '#FF5722',
            'Mixed': '#FFC107',
            'Unverifiable': '#9E9E9E',
            'No Claims': '#E0E0E0',
            'Error': '#795548'
        }
        colors = [langgraph_colors.get(status, '#9E9E9E') for status in df_sentences['Status']]
        
        # 1. Donut chart for sentence status
        wedges, texts, autotexts = ax1.pie(
            df_sentences['Count'], 
            labels=df_sentences['Status'], 
            autopct='%1.1f%%', 
            colors=colors,
            startangle=90,
            pctdistance=0.85,
            wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2)
        )
        ax1.set_title('🍰 Sentence Verification Status', fontweight='bold', pad=20)
        
        # 2. Horizontal bar chart
        bars = ax2.barh(df_sentences['Status'], df_sentences['Count'], color=colors)
        ax2.set_title('📊 Verification Counts', fontweight='bold', pad=20)
        ax2.set_xlabel('Count')
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax2.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'{int(width)}', ha='left', va='center', fontweight='bold')
        
        # 3. Claim-level distribution (if available)
        claim_dist = fact_summary.get('claim_status_distribution', {})
        if claim_dist:
            claim_data = []
            claim_colors_list = []
            for status, data in claim_dist.items():
                formatted_status = status.replace('_', ' ').title()
                claim_data.append(data['count'])
                claim_colors_list.append(langgraph_colors.get(formatted_status, '#9E9E9E'))
            
            claim_labels = [status.replace('_', ' ').title() for status in claim_dist.keys()]
            
            ax3.pie(claim_data, labels=claim_labels, autopct='%1.1f%%', colors=claim_colors_list)
            ax3.set_title('🔍 Individual Claim Results', fontweight='bold', pad=20)
        else:
            ax3.text(0.5, 0.5, 'No claim-level\ndata available', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('🔍 Individual Claim Results', fontweight='bold', pad=20)
        
        # 4. Verification accuracy assessment
        accuracy_data = []
        accuracy_labels = []
        accuracy_colors = []
        
        for status, data in sentence_dist.items():
            if status in ['TRUE', 'FALSE', 'MIXED_PARTIAL']:
                accuracy_data.append(data['count'])
                accuracy_labels.append(status.replace('_', ' ').title())
                accuracy_colors.append(langgraph_colors.get(status.replace('_', ' ').title(), '#9E9E9E'))
        
        if accuracy_data:
            bars4 = ax4.bar(accuracy_labels, accuracy_data, color=accuracy_colors)
            ax4.set_title('✅ Verification Accuracy', fontweight='bold', pad=20)
            ax4.set_ylabel('Count')
            ax4.tick_params(axis='x', rotation=45)
            for bar in bars4:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'No verifiable\nclaims found', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('✅ Verification Accuracy', fontweight='bold', pad=20)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Interactive data table
        st.markdown("#### 📋 Detailed Verification Breakdown")
        st.dataframe(
            df_sentences.style.background_gradient(subset=['Count', 'Percentage']),
            use_container_width=True
        )
    
    # Show detailed claim results if available
    if fact_results and len(fact_results) > 0:
        st.markdown("#### 🔍 Individual Sentence Analysis")
        
        # Create expandable sections for each sentence with claims
        for i, result in enumerate(fact_results):
            if result['claims_count'] > 0:
                with st.expander(f"Sentence {i+1}: {result['sentence'][:100]}... ({result['claims_count']} claims)"):
                    
                    # Show overall sentence assessment
                    col1, col2 = st.columns(2)
                    with col1:
                        status_color = "🟢" if result['overall_status'] == "TRUE" else (
                            "🔴" if result['overall_status'] == "FALSE" else (
                                "🟡" if "MIXED" in result['overall_status'] else "⚪"
                            )
                        )
                        st.write(f"**Overall Status:** {status_color} {result['overall_status']}")
                        st.write(f"**Confidence:** {result['confidence']:.3f}")
                    
                    with col2:
                        st.write(f"**Summary:** {result['summary']}")
                        st.write(f"**Analysis Time:** {result['analysis_timestamp']}")
                    
                    # Show individual claims
                    if result.get('verified_claims'):
                        st.markdown("**Individual Claims:**")
                        for j, claim in enumerate(result['verified_claims']):
                            claim_color = "🟢" if claim['verification_status'] == "TRUE" else (
                                "🔴" if claim['verification_status'] == "FALSE" else (
                                    "🟡" if claim['verification_status'] == "PARTIALLY_TRUE" else "⚪"
                                )
                            )
                            
                            st.markdown(f"**Claim {j+1}:** {claim['claim']}")
                            st.markdown(f"- **Status:** {claim_color} {claim['verification_status']} (Confidence: {claim['confidence']:.3f})")
                            st.markdown(f"- **Explanation:** {claim['explanation']}")
                            if claim.get('context'):
                                st.markdown(f"- **Context:** {claim['context']}")
                            st.markdown("---")


def display_detailed_results(results):
    """
    Display detailed sentence-by-sentence results
    """
    st.subheader("📋 Detailed Analysis Results")
    
    # Create DataFrame
    app = FactSnapV()
    df = app.create_detailed_dataframe(results)
    
    # Add color coding
    def highlight_bias(row):
        """Color code bias levels"""
        if row['bias_level'] == 'High':
            return ['background-color: #ffcccb'] * len(row)  # Light red
        elif row['bias_level'] == 'Medium':
            return ['background-color: #ffe4b5'] * len(row)  # Light orange
        else:
            return [''] * len(row)
    
    # Display DataFrame with styling
    st.dataframe(
        df.style.apply(highlight_bias, axis=1),
        use_container_width=True,
        height=400
    )
    
    # Download options
    st.subheader("📥 Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV download
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"factsnap_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # JSON download
        import json
        summary_data = {
            'file_info': results['file_info'],
            'summary': {
                'emotion_summary': results['emotion_analysis']['summary'],
                'bias_summary': results['bias_analysis']['summary'],
                'fact_summary': results['fact_verification']['summary']
            }
        }
        
        json_str = json.dumps(summary_data, indent=2)
        st.download_button(
            label="Download Summary (JSON)",
            data=json_str,
            file_name=f"factsnap_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )


def main():
    """
    Main Streamlit application
    """
    init_streamlit_config()
    
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'app_initialized' not in st.session_state:
        st.session_state.app_initialized = False
    
    # Display enhanced header and get input method
    input_method = display_file_uploader()
    
    # Handle different input methods
    if input_method == "🎙️ Real-time Streaming":
        # Import and display streaming interface
        from streamlit_audio_streaming import display_streaming_interface
        display_streaming_interface()
        return
    
    # Original file upload logic
    uploaded_file = display_file_upload_section()
    
    # Display enhanced analysis options
    extract_claims, show_detailed_results, confidence_threshold, batch_size = display_analysis_options()
    
    # Process uploaded file
    if uploaded_file is not None:
        st.success(f"File uploaded: {uploaded_file.name}")
        
        # Save uploaded file temporarily
        import tempfile
        import os
        
        # Create a temporary file with proper extension
        file_extension = os.path.splitext(uploaded_file.name)[1]
        temp_file_path = os.path.join(tempfile.gettempdir(), f"factsnap_{uploaded_file.name}")
        
        try:
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Verify file was created and has content
            if not os.path.exists(temp_file_path):
                st.error("Failed to save uploaded file")
                return
            
            file_size = os.path.getsize(temp_file_path)
            st.info(f"File saved: {temp_file_path} ({file_size:,} bytes)")
        
        except Exception as e:
            st.error(f"Error saving file: {str(e)}")
            return
        
        # Enhanced analyze button with progress
        if st.button("🚀 Start Enhanced Analysis", type="primary", help="Click to begin comprehensive analysis"):
            try:
                # Initialize progress bar
                progress_container = st.container()
                
                with progress_container:
                    st.info("Initializing FactSnap-V components...")
                    
                    # Initialize app if not already done
                    if not st.session_state.app_initialized:
                        with st.spinner("Loading AI models..."):
                            app = FactSnapV()
                            st.session_state.app = app
                            st.session_state.app_initialized = True
                    else:
                        app = st.session_state.app
                    
                    st.success("✓ All components initialized!")
                    
                    # Start analysis
                    st.info("Starting analysis...")
                    
                    with st.spinner("Analyzing file... This may take a few minutes."):
                        results = app.analyze_file(temp_file_path, extract_claims=extract_claims)
                        st.session_state.analysis_results = results
                    
                    st.success("✓ Analysis completed!")
                
                # Clean up temporary file
                try:
                    os.remove(temp_file_path)
                except:
                    pass
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                try:
                    os.remove(temp_file_path)
                except:
                    pass
    
    # Display results if available
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        st.header("📊 Analysis Results")
        
        # Display transcript info
        st.subheader("📝 Transcript Information")
        transcript_info = results['transcript']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Sentences", transcript_info['sentence_count'])
        with col2:
            st.metric("Total Words", transcript_info['word_count'])
        with col3:
            st.metric("Total Characters", transcript_info['character_count'])
        with col4:
            st.metric("Analysis Time", f"{results['file_info']['analysis_time']:.1f}s")
        
        # Display transcript preview
        with st.expander("View Transcript"):
            st.text_area("Full Transcript", transcript_info['text'], height=200)
        
        # Display analysis results
        st.markdown("---")
        
        # Emotion Analysis
        display_emotion_analysis(
            results['emotion_analysis']['results'],
            results['emotion_analysis']['summary']
        )
        
        st.markdown("---")
        
        # Bias Analysis
        display_bias_analysis(
            results['bias_analysis']['results'],
            results['bias_analysis']['summary']
        )
        
        st.markdown("---")
        
        # Fact Verification
        if results['fact_verification']['results']:
            display_fact_verification(
                results['fact_verification']['results'],
                results['fact_verification']['summary']
            )
            st.markdown("---")
        
        # Detailed Results
        if show_detailed_results:
            display_detailed_results(results)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>FactSnap-V - AI-Powered Audio Analysis Tool</p>
            <p>Built with ❤️ using Python, Streamlit, and Open Source AI</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
