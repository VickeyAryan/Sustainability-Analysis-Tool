import streamlit as st
import pandas as pd
from transformers import pipeline  # for sentiment analysis
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import io
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize

# NEW IMPORTS FOR EXTRACTIVE SUMMARIZATION & KEY PHRASE EXTRACTION
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from rake_nltk import Rake

# Page Configuration
st.set_page_config(layout="wide")

# Custom Sidebar Background Color
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: #E8F5E9; /* Soft green background */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Sustainability Analysis Tool")

###############################################################################
# Sidebar Navigation
###############################################################################
menu_options = ["About the Tool", "Data Analysis", "Insights & Recommendations"]
selection = st.sidebar.radio("Navigate", menu_options)

# Initialize session state to store data
if 'kpi_data' not in st.session_state:
    st.session_state['kpi_data'] = None
if 'feedback_data' not in st.session_state:
    st.session_state['feedback_data'] = None
if 'analysis_done' not in st.session_state:
    st.session_state['analysis_done'] = False

###############################################################################
# Helper Functions
###############################################################################
def load_data(file, paste_data, file_type):
    if file is not None:
        if file_type == "csv":
            return pd.read_csv(file)
        elif file_type == "xlsx":
            return pd.read_excel(file)
    elif paste_data:
        return pd.read_csv(io.StringIO(paste_data))
    else:
        return None

def preprocess_kpi_data(kpi_data):
    # Define which KPIs are better when higher
    higher_better_kpis = ['training hours', 'fuel efficiency', 'on-time delivery rate']
    
    # Ensure 'KPI Name' is in lowercase for consistency
    kpi_data['KPI Name'] = kpi_data['KPI Name'].str.lower()
    
    # Calculate Percentage Deviation based on KPI direction
    def calculate_percentage_deviation(row):
        if row['KPI Name'] in higher_better_kpis:
            if row['Target'] != 0:
                return ((row['Value'] - row['Target']) / row['Target']) * 100
            else:
                return 0
        else:
            if row['Target'] != 0:
                return ((row['Target'] - row['Value']) / row['Target']) * 100
            else:
                return 0
    
    kpi_data['Deviation (%)'] = kpi_data.apply(calculate_percentage_deviation, axis=1)
    return kpi_data

def preprocess_feedback_data(feedback_data):
    feedback_data['Feedback'] = feedback_data['Feedback'].fillna('')
    feedback_data['Cleaned_Feedback'] = feedback_data['Feedback'].str.lower().str.strip()
    return feedback_data

def transformer_sentiment_analysis(feedback_data):
    sentiment_model = pipeline("sentiment-analysis")
    
    feedback_data['Sentiment'] = feedback_data['Cleaned_Feedback'].apply(
        lambda x: sentiment_model(x)[0]['label'] if isinstance(x, str) and x.strip() != '' else 'NEUTRAL'
    )
    feedback_data['Sentiment_Score'] = feedback_data['Cleaned_Feedback'].apply(
        lambda x: sentiment_model(x)[0]['score'] if isinstance(x, str) and x.strip() != '' else 0.0
    )
    
    def classify_sentiment(row, threshold=0.6):
        if row['Sentiment'].upper() == 'POSITIVE' and row['Sentiment_Score'] >= threshold:
            return 'Positive'
        elif row['Sentiment'].upper() == 'NEGATIVE' and row['Sentiment_Score'] >= threshold:
            return 'Negative'
        else:
            return 'Neutral'
    
    feedback_data['Sentiment_Label'] = feedback_data.apply(classify_sentiment, axis=1)
    return feedback_data

def topic_modeling(feedback_data):
    vectorizer = CountVectorizer(stop_words='english', max_features=5000)
    feedback_matrix = vectorizer.fit_transform(feedback_data['Cleaned_Feedback'])

    lda = LatentDirichletAllocation(n_components=3, random_state=42)
    lda.fit(feedback_matrix)

    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-6:-1]]
        topics.append(", ".join(top_words))

    feedback_data['Topic'] = lda.transform(feedback_matrix).argmax(axis=1)
    return feedback_data, topics


###############################################################################
# UPDATED: sentiment_heatmap
###############################################################################
def sentiment_heatmap(feedback_data):
    """
    Display a smaller, more aesthetic heatmap with reduced fonts and no thick borders.
    """
    heatmap_data = feedback_data.groupby(['Department', 'Sentiment_Label']).size().unstack().fillna(0)
    heatmap_data = heatmap_data.div(heatmap_data.sum(axis=1), axis=0) * 100

    # Use a smaller context and remove black borders (spines).
    with sns.plotting_context("paper", font_scale=0.7):
        sns.set_style("white")
        fig, ax = plt.subplots(figsize=(4, 2.5), dpi=100)
        heat = sns.heatmap(
            heatmap_data,
            annot=True,
            annot_kws={"size": 7},
            fmt='.2f',
            cmap='coolwarm',
            ax=ax,
            cbar_kws={"shrink": 0.7},
            linewidths=0.2,
            linecolor='white'
        )
        # Remove the colorbar outline if desired
        cbar = heat.collections[0].colorbar
        cbar.outline.set_visible(False)

        sns.despine(left=True, bottom=True, ax=ax)
        ax.set_title("Sentiment Heatmap (in %)", fontsize=9)
        ax.tick_params(axis='both', which='major', labelsize=7)

        plt.tight_layout()
        st.pyplot(fig)


def predict_kpi_target(kpi_data):
    predictions = []
    for kpi_name in kpi_data['KPI Name'].unique():
        kpi_subset = kpi_data[kpi_data['KPI Name'] == kpi_name]

        X = kpi_subset[['Value', 'Deviation (%)']]
        y = kpi_subset['Target']

        if len(X) < 2:  # If not enough data, skip
            continue

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)

        predicted_target = model.predict([[X['Value'].mean(), X['Deviation (%)'].mean()]])[0]
        unit = kpi_subset['Unit'].iloc[0] if 'Unit' in kpi_subset.columns else 'Unknown Unit'

        predictions.append({
            'KPI Name': kpi_name.title(),
            'Unit': unit,
            'Current Average': round(X['Value'].mean(), 3),
            'Predicted Target': round(predicted_target, 3)
        })
    return predictions


###############################################################################
# UPDATED: display_kpi_predictions
###############################################################################
def display_kpi_predictions(predictions):
    st.subheader("Predicted KPI Targets")
    st.markdown(
        """
        The table below displays the current average KPI values and predicted future targets based on trends. 
        These predictions help identify focus areas for improvement.
        """
    )

    prediction_df = pd.DataFrame(predictions)
    
    higher_better_kpis = ['training hours', 'fuel efficiency', 'on-time delivery rate']
    prediction_df['kpi_name_lower'] = prediction_df['KPI Name'].str.lower()
    
    def calculate_predicted_percentage_deviation(row):
        if row['kpi_name_lower'] in higher_better_kpis:
            if row['Predicted Target'] != 0:
                return ((row['Current Average'] - row['Predicted Target']) / row['Predicted Target']) * 100
            else:
                return 0
        else:
            if row['Predicted Target'] != 0:
                return ((row['Predicted Target'] - row['Current Average']) / row['Predicted Target']) * 100
            else:
                return 0
    
    prediction_df['Predicted Deviation (%)'] = prediction_df.apply(calculate_predicted_percentage_deviation, axis=1)
    prediction_df = prediction_df.drop(columns=['kpi_name_lower'])
    
    st.table(prediction_df)
    
    if not prediction_df.empty:
        top_focus = prediction_df.reindex(
            prediction_df['Predicted Deviation (%)'].abs().sort_values(ascending=False).index
        ).head(5)

        with sns.plotting_context("paper", font_scale=0.7):
            sns.set_style("white")
            fig, ax = plt.subplots(figsize=(4, 2.5), dpi=100)
            bar = sns.barplot(
                data=top_focus,
                y='KPI Name',
                x='Predicted Deviation (%)',
                ax=ax,
                palette='Blues_d',
                edgecolor=None
            )
            sns.despine(left=True, bottom=True, ax=ax)

            ax.set_title("Top 5 Focus Areas for KPI Improvement Based on Predicted Deviation (%)", fontsize=9)
            ax.set_xlabel("Predicted Deviation (%)", fontsize=8)
            ax.set_ylabel("KPI Name", fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=7)

            # Remove bar edges
            for patch in bar.patches:
                patch.set_linewidth(0)

            # Annotate each bar with the deviation value
            for i, patch in enumerate(ax.patches):
                x_val = patch.get_width()
                y_val = patch.get_y() + patch.get_height() / 2
                ax.text(x_val, y_val, f"{x_val:.2f}%", va='center', ha='left', fontsize=7)

            plt.tight_layout()
            st.pyplot(fig)


def generate_recommendation_and_plan(key_pain_points, avg_deviation, negative_sentiment, key_phrases):
    recommendation_parts = []
    future_planning_parts = []

    if any(kw in key_pain_points for kw in ["equipment", "maintenance", "machine", "breakdowns"]):
        recommendation_parts.append("Address equipment maintenance issues and reduce downtime.")
        future_planning_parts.append("Implement structured maintenance schedules and ensure spare parts availability.")

    if any(kw in key_pain_points for kw in ["energy", "efficiency", "emissions"]):
        recommendation_parts.append("Improve energy efficiency and reduce emissions.")
        future_planning_parts.append("Upgrade to energy-efficient equipment and conduct regular energy audits.")

    if any(kw in key_pain_points for kw in ["waste", "recycling", "packaging"]):
        recommendation_parts.append("Optimize waste management and improve recycling practices.")
        future_planning_parts.append("Introduce waste segregation protocols and partner with recycling vendors.")

    if not recommendation_parts:
        recommendation_parts.append("Focus on the key issues highlighted by feedback to reduce deviation and negativity.")
    if not future_planning_parts:
        future_planning_parts.append("Develop a targeted action plan, allocate resources effectively, and track progress with regular reviews.")

    rec_paragraph = (
        f"Given a deviation of {avg_deviation:.2f}% and negative sentiment of {negative_sentiment:.2f}%, "
        f"it is essential to take targeted measures. "
        + " ".join(recommendation_parts)
    )

    if key_phrases:
        rec_paragraph += f" Key Phrases: {', '.join(key_phrases)}."

    future_planning_text = (
        "To move forward and align with sustainability targets, consider the following steps: "
        + " ".join(future_planning_parts)
    )
    return rec_paragraph, future_planning_text

def hybrid_extractive_analysis(text, num_summary_sentences=3, num_key_phrases=5):
    if not text.strip():
        return ("No substantial feedback to summarize.", [])

    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    lexrank = LexRankSummarizer()
    summary_sentences = lexrank(parser.document, num_summary_sentences)

    processed_sents = []
    for sent in summary_sentences:
        s_str = str(sent).strip()
        if not s_str.endswith('.'):
            s_str += '.'
        s_str = s_str[0].upper() + s_str[1:]
        processed_sents.append(s_str)

    final_summary = f"**Summary:** {' '.join(processed_sents)}"

    rake_extractor = Rake()
    rake_extractor.extract_keywords_from_text(text)
    top_phrases_raw = rake_extractor.get_ranked_phrases()[:num_summary_sentences]
    top_phrases = [phrase.strip().title() for phrase in top_phrases_raw]

    return final_summary, top_phrases

def insights_by_department(kpi_data, feedback_data):
    feedback_data, topics = topic_modeling(feedback_data)
    
    insights = []

    for department in kpi_data['Department'].unique():
        dept_kpi = kpi_data[kpi_data['Department'] == department]
        avg_deviation = dept_kpi['Deviation (%)'].mean()
        avg_target = dept_kpi['Target'].mean()
        avg_value = dept_kpi['Value'].mean()

        dept_feedback = feedback_data[feedback_data['Department'] == department]
        sentiment_summary = dept_feedback['Sentiment_Label'].value_counts(normalize=True).mul(100).to_dict()

        positive_sentiment = sentiment_summary.get('Positive', 0)
        negative_sentiment = sentiment_summary.get('Negative', 0)
        neutral_sentiment = sentiment_summary.get('Neutral', 0)

        topic_counts = dept_feedback['Topic'].value_counts()
        if not topic_counts.empty:
            top_topic_id = topic_counts.idxmax()
            top_topic_description = topics[top_topic_id]
        else:
            top_topic_description = "No significant recurring themes identified."

        all_feedback_text = " ".join(dept_feedback['Cleaned_Feedback'].tolist())

        summary_text, key_phrases = hybrid_extractive_analysis(
            text=all_feedback_text,
            num_summary_sentences=3,
            num_key_phrases=5
        )

        feedback_analysis = (
            f"**Feedback Analysis:** In the **{department}** department, employee feedback can be summarized as:\n\n"
            f"{summary_text}\n\n"
            f"Overall sentiment distribution in {department}: {positive_sentiment:.2f}% Positive, "
            f"{negative_sentiment:.2f}% Negative, {neutral_sentiment:.2f}% Neutral."
        )

        recommendation_text, future_planning_text = generate_recommendation_and_plan(
            [kw.strip() for kw in top_topic_description.split(", ")],
            avg_deviation,
            negative_sentiment,
            key_phrases
        )

        insights.append({
            'Department': department,
            'Average Value': round(avg_value, 3),
            'Average Target': round(avg_target, 3),
            'Deviation (%)': round(avg_deviation, 3),
            'Feedback Analysis': feedback_analysis,
            'Recommendation': f"**Recommendation:** {recommendation_text}",
            'Future Planning': f"**Future Planning:** {future_planning_text}"
        })

    st.subheader("Key Insights and Recommendations")
    insights_df = pd.DataFrame(insights)
    st.table(insights_df[['Department', 'Average Value', 'Average Target']])

    st.markdown("### Detailed Insights:")
    for insight in insights:
        st.markdown(f"#### Department: {insight['Department']}")
        st.markdown(insight['Feedback Analysis'])
        st.markdown(insight['Recommendation'])
        st.markdown(insight['Future Planning'])
        st.markdown("---")


###############################################################################
# Page: About the Tool
###############################################################################
if selection == "About the Tool":
    st.subheader("About the Tool")
    st.markdown(
        """
        **Welcome to the Sustainability Analysis Tool!**

        This comprehensive application is designed to help organizations better understand,
        monitor, and enhance their sustainability efforts. By consolidating Key Performance
        Indicators (KPIs) and employee feedback, the tool enables you to:

        - **Identify KPI Deviations**: Quickly spot where performance is lagging or leading
          against sustainability targets, and see how far off you are from your goals.
        - **Assess Employee Sentiments**: Gain insights into how your workforce perceives
          the organization’s sustainability programs and initiatives. Negative or positive
          trends can reveal key areas needing attention.
        - **Predict Future KPI Targets**: Harness historical data to anticipate future
          performance levels, thereby helping you set realistic goals and allocate resources
          effectively.
        - **Plan Actionable Recommendations**: Use the summarized feedback and KPI data
          to derive meaningful recommendations and future planning steps that align with
          your sustainability goals.

        **How This Tool Helps Organizations**:
        - Provides data-driven insights for strategic decision-making around sustainability.
        - Highlights potential operational inefficiencies, waste management issues, or
          emission challenges.
        - Encourages a culture of continuous improvement by monitoring sentiment and
          encouraging employee participation.

        **Outcome**:
        - A concise overview of your organization's sustainability performance.
        - Prioritized focus areas and clear next steps to enhance sustainability efforts.
        - A plan to integrate ongoing feedback and adapt to evolving sustainability targets.

        By using this tool, organizations can independently delve into their data, understand
        where they stand, and chart a clear course toward a more sustainable future.
        """
    )
    st.info("**Note**: To proceed, click on 'Data Analysis' in the sidebar to upload your data.")


###############################################################################
# Page: Data Analysis
###############################################################################
elif selection == "Data Analysis":
    st.subheader("Upload Data for Analysis")
    kpi_file = st.file_uploader("Upload KPI Data (CSV or Excel)", type=["csv", "xlsx"])
    kpi_paste = st.text_area("Or Paste KPI Data (CSV Format)")
    feedback_file = st.file_uploader("Upload Feedback Data (CSV or Excel)", type=["csv", "xlsx"])
    feedback_paste = st.text_area("Or Paste Feedback Data (CSV Format)")

    if st.button("Analyze"):
        # Load KPI Data
        if kpi_file:
            file_type = "csv" if kpi_file.name.endswith("csv") else "xlsx"
            st.session_state['kpi_data'] = load_data(kpi_file, None, file_type)
        else:
            st.session_state['kpi_data'] = load_data(None, kpi_paste, "csv")

        # Load Feedback Data
        if feedback_file:
            file_type = "csv" if feedback_file.name.endswith("csv") else "xlsx"
            st.session_state['feedback_data'] = load_data(feedback_file, None, file_type)
        else:
            st.session_state['feedback_data'] = load_data(None, feedback_paste, "csv")

        if st.session_state['kpi_data'] is not None and st.session_state['feedback_data'] is not None:
            # Process data
            st.session_state['kpi_data'] = preprocess_kpi_data(st.session_state['kpi_data'])
            st.session_state['feedback_data'] = preprocess_feedback_data(st.session_state['feedback_data'])
            st.session_state['feedback_data'] = transformer_sentiment_analysis(st.session_state['feedback_data'])
            st.session_state['analysis_done'] = True

            st.success("Analysis Completed!")
        else:
            st.warning("Please upload or paste both KPI and Feedback data to proceed.")

    if st.session_state['analysis_done']:
        st.info("**Note**: Now that analysis is complete, proceed to 'Insights & Recommendations' to view results.")


###############################################################################
# Page: Insights & Recommendations
###############################################################################
elif selection == "Insights & Recommendations":
    if st.session_state['analysis_done']:
        st.subheader("Analysis Results & Recommendations")

        # Display KPI Data
        st.write("### KPI Data")
        st.dataframe(st.session_state['kpi_data'].style.highlight_max(subset=['Deviation (%)'], color='green', axis=0))

        # Display Feedback Data
        st.write("### Feedback Data")
        st.dataframe(st.session_state['feedback_data'].style.highlight_max(subset=['Sentiment_Label'], color='yellow', axis=0))

        # Display Sentiment Heatmap
        st.subheader("Sentiment Heatmap")
        sentiment_heatmap(st.session_state['feedback_data'])

        # Display KPI Predictions
        predictions = predict_kpi_target(st.session_state['kpi_data'])
        display_kpi_predictions(predictions)

        # Display Department Insights
        insights_by_department(st.session_state['kpi_data'], st.session_state['feedback_data'])

        st.info("**Note**: Review the above insights, predictions, and recommendations for your sustainability planning.")
    else:
        st.warning("Data has not been analyzed yet. Please go to 'Data Analysis' and click 'Analyze' first.")
