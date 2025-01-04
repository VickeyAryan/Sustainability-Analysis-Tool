# Sustainability Analysis Tool

The **Sustainability Analysis Tool** is an interactive web application designed to help organizations monitor and enhance their sustainability efforts. It integrates **Key Performance Indicators (KPIs)** and **employee feedback analysis** to provide actionable insights, recommendations, and predictive analytics. Built with Python and Streamlit, this tool empowers businesses to make data-driven decisions toward sustainability goals.

---

## Features

### 1. **Data Upload and Processing**
- Supports CSV and Excel file uploads for KPI and feedback data.
- Allows direct pasting of data in CSV format.
- Automatically preprocesses data for analysis.

### 2. **KPI Analysis**
- Calculates percentage deviation from targets for various KPIs.
- Highlights KPIs where performance lags or leads against expectations.

### 3. **Sentiment Analysis**
- Uses advanced NLP models to analyze employee feedback.
- Categorizes feedback into Positive, Neutral, or Negative sentiments.
- Visualizes sentiment distribution across departments.

### 4. **Topic Modeling**
- Identifies recurring themes in employee feedback using LDA.
- Extracts actionable insights and key phrases.

### 5. **Predictive Analytics**
- Predicts future KPI targets using Random Forest Regression.
- Provides a focused view of areas needing attention.

### 6. **Visualization and Recommendations**
- Heatmaps to visualize sentiment trends across departments.
- Detailed insights with actionable recommendations for improvement.
- Bar charts highlighting top focus areas for KPI improvement.

---

## Installation

### Prerequisites
- Python 3.8 or above
- pip (Python package manager)

### Steps
1. Clone the repository:
   git clone https://github.com/VickeyAryan/Sustainability-Analysis-Tool.git
   cd Sustainability-Analysis-Tool
2. Install required dependencies:
   pip install -r requirements.txt
3. Download NLTK dependencies:
   import nltk
   nltk.download('punkt')
4. Run the application:
   streamlit run main.py


### How to Use the Tool
1. Launch the application using:
   streamlit run main.py

2. Navigate through the sidebar menu:
-  About the Tool: Learn about the features and functionality.
-  Data Analysis: Upload or paste your KPI and feedback data.
-  Insights & Recommendations: View insights, predictions, and recommendations.

3. Upload your data:
-  KPI Data: Provide data on sustainability KPIs (CSV or Excel format).
-  Feedback Data: Provide employee feedback data (CSV or Excel format).

4. Analyze the results:
-  Visualize sentiment heatmaps and KPI predictions.
-  Review insights and recommendations for improving sustainability performance.

### File Structure
.
├── main.py                    # Main application file
├── requirements.txt           # Dependencies for the project
├── README.md                  # Project documentation
└── data/                      # (Optional) Folder for sample datasets

### Technologies Used
### 1. **Streamlit:** 
- For creating the interactive web app.
### 2. **Python Libraries:**
- pandas: For data manipulation.
- transformers: For sentiment analysis.
- scikit-learn: For machine learning and predictions.
- nltk: For natural language processing.
- sumy and rake_nltk: For text summarization and key phrase extraction.
- matplotlib and seaborn: For data visualization.

### Contributing
We welcome contributions to enhance the Sustainability Analysis Tool. To contribute:

1. Fork the repository.
2. Create a feature branch:
- git checkout -b feature-name
3. Commit your changes:
- git commit -m "Add a new feature"
4. Push your changes:
- git push origin feature-name
5. Open a pull request.

### License
This project is licensed under the MIT License. See the LICENSE file for more details.

### Contact
For any questions or suggestions:

**Author:** Vickey Swami
**Email:** aryanisation1@gmail.com
**GitHub:** VickeyAryan

### Future Enhancements
1. Add advanced visualization dashboards.
2. Support for multilingual feedback analysis.
3. Integrate real-time data processing via APIs.
4.  predictive models for KPI and sentiment trends.
