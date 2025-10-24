"""
Streamlit components for real-time audio streaming
"""

import streamlit as st
import time
import threading
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from real_time_audio import StreamingFactSnapV, RealTimeAudioProcessor
from main import FactSnapV


class StreamlitAudioStreaming:
    """
    Streamlit interface for real-time audio streaming
    """
    
    def __init__(self):
        self.streaming_processor = None
        self.is_streaming = False
        
        # Initialize session state
        if 'streaming_results' not in st.session_state:
            st.session_state.streaming_results = {
                'transcripts': [],
                'emotions': [],
                'bias': [],
                'timestamps': []
            }
        
        if 'streaming_active' not in st.session_state:
            st.session_state.streaming_active = False
        
        if 'streaming_processor_instance' not in st.session_state:
            st.session_state.streaming_processor_instance = None
        
        # Full buffer analysis results holder
        if 'full_buffer_results' not in st.session_state:
            st.session_state.full_buffer_results = None
    
    def display_device_selector(self):
        """
        Display audio device selector
        """
        st.markdown("### 🎤 Audio Device Selection")
        
        try:
            # Get available devices
            processor = RealTimeAudioProcessor()
            devices = processor.get_available_devices()
            processor.cleanup()
            
            if not devices:
                st.error("No audio input devices found!")
                return None
            
            # Create device options
            device_options = {}
            for device in devices:
                label = f"{device['name']} (Channels: {device['channels']}, Rate: {int(device['sample_rate'])}Hz)"
                device_options[label] = device['index']
            
            selected_device_label = st.selectbox(
                "Select audio input device:",
                options=list(device_options.keys()),
                help="Choose the microphone or audio input device to use"
            )
            
            return device_options[selected_device_label]
            
        except Exception as e:
            st.error(f"Error getting audio devices: {str(e)}")
            return None
    
    def display_streaming_controls(self, device_index):
        """
        Display streaming control buttons
        
        Args:
            device_index (int): Selected audio device index
        """
        st.markdown("### 🎛️ Streaming Controls")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🎙️ Start Streaming", 
                        disabled=st.session_state.streaming_active,
                        type="primary"):
                self.start_streaming(device_index)
        
        with col2:
            if st.button("⏹️ Stop Streaming", 
                        disabled=not st.session_state.streaming_active,
                        type="secondary"):
                self.stop_streaming()
        
        with col3:
            if st.button("🔄 Clear Results", 
                        disabled=st.session_state.streaming_active):
                st.session_state.streaming_results = {
                    'transcripts': [],
                    'emotions': [],
                    'bias': [],
                    'timestamps': []
                }
                st.session_state.full_buffer_results = None
                st.success("Results cleared!")
        
        with col4:
            if st.button("💾 Save Session", 
                        disabled=not st.session_state.streaming_active):
                self.save_current_session()
        
        # Second row controls for full-buffer analysis
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            analyze_disabled = st.session_state.streaming_processor_instance is None
            if st.button("🎧 Analyze Full Audio", disabled=analyze_disabled, help="Analyze the entire buffered audio at once"):
                self.analyze_full_buffer()
        with col6:
            if st.button("📥 Download Buffer", disabled=st.session_state.streaming_processor_instance is None, help="Download the current buffered audio"):
                try:
                    if st.session_state.streaming_processor_instance:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_path = f"streaming_buffer_{timestamp}.wav"
                        saved_path = st.session_state.streaming_processor_instance.save_session(output_path)
                        with open(saved_path, 'rb') as f:
                            st.download_button(
                                label="Download Buffered Audio",
                                data=f.read(),
                                file_name=os.path.basename(saved_path),
                                mime="audio/wav",
                                key=f"download_buffer_{timestamp}"
                            )
                except Exception as e:
                    st.error(f"Error downloading buffer: {str(e)}")
        
        # Status indicator
        if st.session_state.streaming_active:
            st.markdown("""
            <div style="background: linear-gradient(90deg, #ff6b6b, #ffa500);
                        padding: 0.5rem; border-radius: 5px; text-align: center; margin: 1rem 0;">
                <strong style="color: white;">🔴 LIVE - Listening for audio...</strong>
            </div>
            """, unsafe_allow_html=True)
            
            # Processing tips
            st.markdown("""
            **💡 Tips for best results:**
            - Speak clearly and at normal volume
            - Pause between sentences for better processing
            - Avoid background noise when possible
            - The system processes audio every 5 seconds
            """)
        else:
            st.markdown("""
            <div style="background: #e0e0e0; padding: 0.5rem; border-radius: 5px; text-align: center; margin: 1rem 0;">
                <strong>⚪ Ready to start streaming</strong>
            </div>
            """, unsafe_allow_html=True)
    
    def start_streaming(self, device_index):
        """
        Start real-time streaming
        
        Args:
            device_index (int): Audio device index
        """
        try:
            # Stop any existing streaming session first
            if st.session_state.streaming_processor_instance:
                st.session_state.streaming_processor_instance.stop_streaming()
                st.session_state.streaming_processor_instance.cleanup()
            
            # Initialize FactSnap-V if not already done
            if 'factsnap_app' not in st.session_state:
                with st.spinner("Initializing FactSnap-V..."):
                    st.session_state.factsnap_app = FactSnapV()
            
            # Initialize streaming processor
            self.streaming_processor = StreamingFactSnapV(st.session_state.factsnap_app)
            st.session_state.streaming_processor_instance = self.streaming_processor
            
            # Clear previous results
            st.session_state.streaming_results = {
                'transcripts': [],
                'emotions': [],
                'bias': [],
                'timestamps': []
            }
            
            # Start streaming
            self.streaming_processor.start_streaming(device_index)
            st.session_state.streaming_active = True
            
            st.success("🎙️ Streaming started! Speak clearly into your microphone...")
            st.info("💡 The system processes audio every 5 seconds. Speak for at least 2 seconds to trigger processing.")
            
        except Exception as e:
            st.error(f"Error starting streaming: {str(e)}")
    
    def stop_streaming(self):
        """
        Stop real-time streaming
        """
        try:
            # Stop current instance
            if self.streaming_processor:
                self.streaming_processor.stop_streaming()
                self.streaming_processor.cleanup()
                self.streaming_processor = None
            
            # Stop session state instance
            if st.session_state.streaming_processor_instance:
                st.session_state.streaming_processor_instance.stop_streaming()
                st.session_state.streaming_processor_instance.cleanup()
                st.session_state.streaming_processor_instance = None
            
            st.session_state.streaming_active = False
            
            # Show summary
            total_segments = len(st.session_state.streaming_results['transcripts'])
            st.success(f"⏹️ Streaming stopped! Processed {total_segments} audio segments.")
            
        except Exception as e:
            st.error(f"Error stopping streaming: {str(e)}")
    
    def update_streaming_results(self):
        """
        Update streaming results from the processor queue (thread-safe)
        """
        if st.session_state.streaming_processor_instance:
            new_results = st.session_state.streaming_processor_instance.get_new_results()
            
            for result in new_results:
                st.session_state.streaming_results['transcripts'].append(result['transcript'])
                st.session_state.streaming_results['emotions'].append(result['emotions'])
                st.session_state.streaming_results['bias'].append(result['bias'])
                st.session_state.streaming_results['timestamps'].append(result['timestamp'])
                
                # Show notification for new result
                st.success(f"✅ Processed: {result['sentence_count']} sentences - {result['transcript'][:50]}...")
    
    def save_current_session(self):
        """
        Save current streaming session
        """
        try:
            if self.streaming_processor:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"streaming_session_{timestamp}.wav"
                
                saved_path = self.streaming_processor.save_session(output_path)
                st.success(f"💾 Session saved to: {saved_path}")
                
                # Offer download
                with open(saved_path, 'rb') as f:
                    st.download_button(
                        label="📥 Download Audio",
                        data=f.read(),
                        file_name=f"streaming_session_{timestamp}.wav",
                        mime="audio/wav"
                    )
            
        except Exception as e:
            st.error(f"Error saving session: {str(e)}")
    
    def display_live_results(self):
        """
        Display live streaming results
        """
        # Update results from processor queue if streaming is active
        if st.session_state.streaming_active:
            self.update_streaming_results()
        
        if not st.session_state.streaming_results['transcripts']:
            st.info("🎤 Start streaming to see live results...")
        else:
            st.markdown("### 📊 Live Analysis Results")
            
            # Show summary metrics
            total_segments = len(st.session_state.streaming_results['transcripts'])
            total_words = sum(len(t.split()) for t in st.session_state.streaming_results['transcripts'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📝 Segments Processed", total_segments)
            with col2:
                st.metric("💬 Total Words", total_words)
            with col3:
                if st.session_state.streaming_results['timestamps']:
                    latest_time = st.session_state.streaming_results['timestamps'][-1]
                    try:
                        dt = datetime.fromisoformat(latest_time.replace('Z', '+00:00'))
                        time_str = dt.strftime("%H:%M:%S")
                        st.metric("🕐 Latest Update", time_str)
                    except:
                        st.metric("🕐 Latest Update", "Now")
            
            # Create tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["📝 Transcript", "😊 Emotions", "⚖️ Bias", "📈 Analytics"])
            
            with tab1:
                self.display_live_transcript()
            
            with tab2:
                self.display_live_emotions()
            
            with tab3:
                self.display_live_bias()
            
            with tab4:
                self.display_live_analytics()
        
        # If full buffer analysis results exist, display them below
        if st.session_state.get('full_buffer_results'):
            self.display_full_buffer_results(st.session_state.full_buffer_results)
    
    def display_live_transcript(self):
        """
        Display live transcript updates
        """
        st.markdown("#### 📝 Real-time Transcript")
        
        # Show recent transcripts
        transcripts = st.session_state.streaming_results['transcripts']
        timestamps = st.session_state.streaming_results['timestamps']
        
        if transcripts:
            # Create scrollable container
            transcript_container = st.container()
            
            with transcript_container:
                for i, (transcript, timestamp) in enumerate(zip(transcripts[-10:], timestamps[-10:])):
                    # Parse timestamp
                    try:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        time_str = dt.strftime("%H:%M:%S")
                    except:
                        time_str = timestamp
                    
                    st.markdown(f"**[{time_str}]** {transcript}")
                    st.markdown("---")
            
            # Show statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Segments", len(transcripts))
            with col2:
                total_words = sum(len(t.split()) for t in transcripts)
                st.metric("Total Words", total_words)
            with col3:
                if transcripts:
                    avg_length = sum(len(t) for t in transcripts) / len(transcripts)
                    st.metric("Avg Segment Length", f"{avg_length:.0f} chars")
    
    def display_live_emotions(self):
        """
        Display live emotion analysis
        """
        st.markdown("#### 😊 Real-time Emotion Analysis")
        
        emotions_data = st.session_state.streaming_results['emotions']
        
        if not emotions_data:
            st.info("No emotion data available yet...")
            return
        
        # Aggregate emotion data
        emotion_counts = {}
        emotion_confidences = {}
        
        for emotion_batch in emotions_data:
            for emotion_result in emotion_batch:
                emotion = emotion_result['emotion']
                confidence = emotion_result['confidence']
                
                if emotion not in emotion_counts:
                    emotion_counts[emotion] = 0
                    emotion_confidences[emotion] = []
                
                emotion_counts[emotion] += 1
                emotion_confidences[emotion].append(confidence)
        
        if emotion_counts:
            # Create emotion distribution chart
            emotions = list(emotion_counts.keys())
            counts = list(emotion_counts.values())
            avg_confidences = [sum(emotion_confidences[e])/len(emotion_confidences[e]) for e in emotions]
            
            # Pie chart
            fig_pie = px.pie(
                values=counts,
                names=emotions,
                title="Emotion Distribution"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Bar chart with confidence
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=emotions,
                y=counts,
                text=[f"{c}<br>Conf: {conf:.2f}" for c, conf in zip(counts, avg_confidences)],
                textposition='auto'
            ))
            fig_bar.update_layout(title="Emotion Counts with Average Confidence")
            st.plotly_chart(fig_bar, use_container_width=True)
    
    def display_live_bias(self):
        """
        Display live bias analysis
        """
        st.markdown("#### ⚖️ Real-time Bias Analysis")
        
        bias_data = st.session_state.streaming_results['bias']
        
        if not bias_data:
            st.info("No bias data available yet...")
            return
        
        # Aggregate bias data
        bias_levels = {'Low': 0, 'Medium': 0, 'High': 0}
        bias_scores = []
        
        for bias_batch in bias_data:
            for bias_result in bias_batch:
                level = bias_result['bias_level']
                score = bias_result['bias_score']
                
                bias_levels[level] += 1
                bias_scores.append(score)
        
        if bias_scores:
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Low Bias", bias_levels['Low'])
            with col2:
                st.metric("Medium Bias", bias_levels['Medium'])
            with col3:
                st.metric("High Bias", bias_levels['High'])
            with col4:
                avg_score = sum(bias_scores) / len(bias_scores)
                st.metric("Avg Bias Score", f"{avg_score:.3f}")
            
            # Bias distribution
            fig_bias = px.bar(
                x=list(bias_levels.keys()),
                y=list(bias_levels.values()),
                title="Bias Level Distribution",
                color=list(bias_levels.values()),
                color_continuous_scale=['green', 'orange', 'red']
            )
            st.plotly_chart(fig_bias, use_container_width=True)
            
            # Bias score timeline
            if len(bias_scores) > 1:
                fig_timeline = px.line(
                    x=list(range(len(bias_scores))),
                    y=bias_scores,
                    title="Bias Score Over Time",
                    labels={'x': 'Segment', 'y': 'Bias Score'}
                )
                fig_timeline.add_hline(y=0.3, line_dash="dash", line_color="orange", 
                                     annotation_text="Medium Threshold")
                fig_timeline.add_hline(y=0.7, line_dash="dash", line_color="red", 
                                     annotation_text="High Threshold")
                st.plotly_chart(fig_timeline, use_container_width=True)
    
    def display_live_analytics(self):
        """
        Display live analytics dashboard
        """
        st.markdown("#### 📈 Live Analytics Dashboard")
        
        results = st.session_state.streaming_results
        
        if not results['transcripts']:
            st.info("No analytics data available yet...")
            return
        
        # Overall statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Segments", len(results['transcripts']))
        
        with col2:
            total_words = sum(len(t.split()) for t in results['transcripts'])
            st.metric("Total Words", total_words)
        
        with col3:
            if results['timestamps']:
                start_time = datetime.fromisoformat(results['timestamps'][0].replace('Z', '+00:00'))
                end_time = datetime.fromisoformat(results['timestamps'][-1].replace('Z', '+00:00'))
                duration = (end_time - start_time).total_seconds()
                st.metric("Duration", f"{duration:.0f}s")
        
        with col4:
            if total_words > 0 and duration > 0:
                wpm = (total_words / duration) * 60
                st.metric("Words/Min", f"{wpm:.0f}")
        
        # Activity timeline
        if len(results['timestamps']) > 1:
            # Create activity chart
            timestamps = [datetime.fromisoformat(ts.replace('Z', '+00:00')) for ts in results['timestamps']]
            word_counts = [len(t.split()) for t in results['transcripts']]
            
            fig_activity = px.line(
                x=timestamps,
                y=word_counts,
                title="Speaking Activity Over Time",
                labels={'x': 'Time', 'y': 'Words per Segment'}
            )
            st.plotly_chart(fig_activity, use_container_width=True)
    def analyze_full_buffer(self, extract_claims=True):
        """
        Save the current buffered audio and run a full analysis on it.
        """
        try:
            if not st.session_state.streaming_processor_instance:
                st.error("No streaming session initialized. Start streaming first to capture audio.")
                return
            
            with st.spinner("Saving buffered audio and running full analysis... this may take a moment"):
                # Save current buffer
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                temp_audio_path = f"full_buffer_{timestamp}.wav"
                saved_path = st.session_state.streaming_processor_instance.save_session(temp_audio_path)
                
                # Ensure FactSnap-V app exists
                if 'factsnap_app' not in st.session_state:
                    st.session_state.factsnap_app = FactSnapV()
                
                # Run full analysis on saved audio
                results = st.session_state.factsnap_app.analyze_file(saved_path, extract_claims=extract_claims)
                st.session_state.full_buffer_results = results
            
            st.success("🎧 Full buffer analysis complete! See results below.")
        except ValueError as ve:
            st.error(str(ve))
        except Exception as e:
            st.error(f"Error analyzing full buffer: {str(e)}")
    
    def display_full_buffer_results(self, results):
        """
        Display the results of the full-buffer analysis.
        """
        st.markdown("---")
        st.markdown("### 🎧 Full Buffer Analysis Results")
        
        try:
            # Summary metrics
            transcript_info = results['transcript']
            file_info = results['file_info']
            facts = results['fact_verification']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Sentences", transcript_info['sentence_count'])
            with col2:
                st.metric("Words", transcript_info['word_count'])
            with col3:
                st.metric("Chars", transcript_info['character_count'])
            with col4:
                st.metric("Analysis Time", f"{file_info['analysis_time']:.1f}s")
            
            # Detailed table
            app = st.session_state.factsnap_app
            df = app.create_detailed_dataframe(results)
            st.dataframe(df, use_container_width=True, height=300)
            
            # Download buttons
            colA, colB = st.columns(2)
            with colA:
                csv = df.to_csv(index=False)
                st.download_button("Download Detailed CSV", data=csv, file_name=f"full_buffer_{file_info['timestamp'].replace(':','-')}.csv", mime="text/csv")
            with colB:
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
                st.download_button("Download Summary JSON", data=json_str, file_name=f"full_buffer_{file_info['timestamp'].replace(':','-')}.json", mime="application/json")
        except Exception as e:
            st.error(f"Error displaying results: {str(e)}")


def display_streaming_interface():
    """
    Main function to display streaming interface
    """
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 1.5rem; border-radius: 15px; margin-bottom: 2rem;">
        <h2 style="color: white; margin: 0; text-align: center;">
            🎙️ Real-time Audio Streaming
        </h2>
        <p style="color: white; text-align: center; margin: 0.5rem 0 0 0;">
            Live speech analysis with emotion, bias, and fact-checking
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize streaming interface
    streaming_ui = StreamlitAudioStreaming()
    
    # Device selection
    device_index = streaming_ui.display_device_selector()
    
    if device_index is not None:
        # Streaming controls
        streaming_ui.display_streaming_controls(device_index)
        
        # Live results container
        results_container = st.container()
        
        with results_container:
            streaming_ui.display_live_results()
        
        # Auto-refresh if streaming is active
        if st.session_state.streaming_active:
            time.sleep(2)
            st.rerun()
    
    # Instructions
    with st.expander("📖 How to use Real-time Streaming"):
        st.markdown("""
        ### Getting Started:
        1. **Select Audio Device**: Choose your microphone from the dropdown
        2. **Start Streaming**: Click "Start Streaming" to begin live analysis
        3. **Speak Naturally**: The system will process your speech in real-time
        4. **Monitor Results**: Watch live updates in the tabs below
        5. **Save Session**: Click "Save Session" to download the recorded audio
        
        ### Features:
        - **Real-time Transcription**: Live speech-to-text conversion
        - **Emotion Analysis**: Detect emotions as you speak
        - **Bias Detection**: Monitor for biased language in real-time
        - **Live Analytics**: Track speaking patterns and statistics
        
        ### Tips:
        - Speak clearly and at a normal pace for best results
        - Ensure your microphone is working and not muted
        - The system processes audio in 5-second chunks
        - Results may have a slight delay due to processing time
        - **Results will appear automatically** as you speak
        
        ### Troubleshooting:
        - If no results appear, check your microphone volume
        - Make sure you're speaking for at least 2-3 seconds
        - Try stopping and restarting the stream if issues persist
        """)


if __name__ == "__main__":
    display_streaming_interface()