import streamlit as st
import joblib
import traceback
import pandas as pd
from io import StringIO
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load the model pipeline
@st.cache_resource
def load_model():
    try:
        model = joblib.load('NEWS_PREDICTOR/news_detection.pkl')
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.error(traceback.format_exc())
        return None

model = load_model()

st.set_page_config(page_title="News Detection", page_icon="📰", layout="centered")

st.sidebar.title("About")
st.sidebar.info(
    """
    This app uses a machine learning pipeline to classify news articles as **Real** or **Fake**.
    Enter the news text and click **Predict** to see the result.
    """
)

# Theme toggle
theme = st.sidebar.radio("Select Theme", options=["Light", "Dark"], index=1)
if theme == "Light":
    st.markdown(
        """
        <style>
        .main {
            background-color: #ffffff;
            color: #000000;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
        .main {
            background-color: #0e1117;
            color: #fafafa;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

st.title("📰 News Detection Model Deployment")

# Example news texts
examples = [
    "The government has announced a new policy to improve education.",
    "Scientists discovered a cure for the common cold.",
    "Celebrity involved in a scandalous event last night.",
    "Fake news spreads misinformation about health risks.",
    "New technology promises to revolutionize the industry."
]

def plot_wordcloud(text):
    wordcloud = WordCloud(width=400, height=200, background_color='white').generate(text)
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

def plot_prediction_distribution(history):
    if not history:
        st.info("No prediction history to display.")
        return
    df = pd.DataFrame(history)
    counts = df['label'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=['#2ecc71', '#e74c3c'])
    ax.axis('equal')
    st.pyplot(fig)

if model is not None:
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'confidence_threshold' not in st.session_state:
        st.session_state.confidence_threshold = 0.0
    if 'search_text' not in st.session_state:
        st.session_state.search_text = ""

    col1, col2 = st.columns([3, 2])

    with col1:
        user_input = st.text_area("Enter the news text to classify:", height=200)
        if st.button("Clear"):
            user_input = ""

        st.markdown("### Example News Texts")
        for example in examples:
            if st.button(example[:50] + "..."):
                user_input = example

        uploaded_file = st.file_uploader("Upload a text file for batch prediction", type=["txt"])
        if uploaded_file is not None:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            lines = stringio.readlines()
            batch_results = []
            with st.spinner("Processing batch predictions..."):
                for line in lines:
                    line = line.strip()
                    if line:
                        try:
                            pred = model.predict([line])[0]
                            proba = model.predict_proba([line])[0]
                            label = "🟢 Real News" if pred == 1 else "🔴 Fake News"
                            confidence = max(proba) * 100
                            batch_results.append({"text": line, "label": label, "confidence": confidence})
                        except Exception as e:
                            batch_results.append({"text": line, "label": f"Error: {e}", "confidence": 0})
            df_results = pd.DataFrame(batch_results)
            with st.expander("Batch Prediction Results"):
                st.dataframe(df_results)
                csv = df_results.to_csv(index=False).encode('utf-8')
                st.download_button("Download Results as CSV", data=csv, file_name="batch_predictions.csv", mime="text/csv")

    with col2:
        st.markdown("### Prediction Result")
        if st.button("Predict"):
            if user_input:
                with st.spinner("Predicting..."):
                    try:
                        prediction = model.predict([user_input])[0]
                        proba = model.predict_proba([user_input])[0]
                        label = "🟢 Real News" if prediction == 1 else "🔴 Fake News"
                        confidence = max(proba) * 100
                        if confidence >= st.session_state.confidence_threshold:
                            st.success(f"Prediction: {label} (Confidence: {confidence:.2f}%)")
                            st.session_state.history.append({"text": user_input, "label": label, "confidence": confidence})

                            # Word cloud visualization
                            st.markdown("### Word Cloud of Input Text")
                            plot_wordcloud(user_input)
                        else:
                            st.warning(f"Prediction confidence {confidence:.2f}% is below the threshold {st.session_state.confidence_threshold:.2f}%")
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")
                        st.error(traceback.format_exc())
            else:
                st.warning("Please enter some text to classify.")

        st.markdown("### Confidence Threshold")
        st.session_state.confidence_threshold = st.slider("Minimum confidence to show prediction (%)", 0.0, 100.0, st.session_state.confidence_threshold)

        st.markdown("### Search Prediction History")
        st.session_state.search_text = st.text_input("Filter history by text", st.session_state.search_text)

        if st.session_state.history:
            with st.expander("Prediction History"):
                if st.button("Clear History"):
                    st.session_state.history = []
                filtered_history = [record for record in st.session_state.history if st.session_state.search_text.lower() in record['text'].lower()]
                for i, record in enumerate(reversed(filtered_history[-10:])):
                    st.write(f"{i+1}. {record['text'][:100]}... - {record['label']} ({record['confidence']:.2f}%)")

            st.markdown("### Prediction Distribution")
            plot_prediction_distribution(st.session_state.history)
else:
    st.stop()
