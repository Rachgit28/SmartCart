# 🛒 SmartCart Customer Clustering System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![ML](https://img.shields.io/badge/ML-Unsupervised%20Learning-green.svg)](https://scikit-learn.org/)

> An intelligent customer segmentation system using unsupervised machine learning to discover hidden patterns in customer behavior and enable data-driven marketing strategies.

## 📋 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Key Features](#-key-features)
- [Dataset Description](#-dataset-description)
- [Getting Started](#-getting-started)
- [Project Workflow](#-project-workflow)
- [Technologies Used](#-technologies-used)
- [Results & Insights](#-results--insights)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

## 🎯 Overview

**SmartCart** is a growing e-commerce platform serving customers across multiple countries. This project implements an **intelligent customer segmentation system** using **unsupervised machine learning** to analyze customer behavior patterns and enable personalized marketing strategies.

The system analyzes **2,240 customer records** with **22 attributes** covering demographics, purchase behavior, website activity, and customer feedback to group customers into meaningful clusters based on purchasing behavior, engagement levels, and loyalty indicators.

## 🚨 Problem Statement

SmartCart currently faces several challenges:

- **Generic Marketing**: One-size-fits-all marketing and engagement strategies for all customers
- **Inefficient Resource Allocation**: Marketing efforts not targeted to specific customer segments
- **Missed Opportunities**: Inability to identify and retain high-value customers
- **Delayed Churn Detection**: Late identification of customers at risk of leaving the platform

### Solution

Build an **intelligent customer segmentation system** using clustering algorithms to:
- Discover hidden patterns in customer behavior
- Group customers into meaningful clusters
- Enable data-driven decision-making for personalized marketing
- Support proactive customer retention strategies

## ✨ Key Features

- 🎯 **Customer Segmentation**: Automated clustering based on purchasing behavior and engagement
- 📊 **Behavioral Analysis**: Deep insights into customer demographics and purchase patterns
- 🤖 **Unsupervised Learning**: K-Means, Hierarchical, and DBSCAN clustering algorithms
- 📈 **Data Visualization**: Interactive visualizations of customer segments
- 💡 **Actionable Insights**: Marketing recommendations for each customer cluster
- 🔍 **Pattern Discovery**: Identification of high-value, at-risk, and loyal customer groups

## 📊 Dataset Description

The dataset contains **2,240 customer records** with **22 attributes** organized into four categories:

### 1. Customer Demographics

| Feature | Description |
|---------|-------------|
| `ID` | Unique customer identifier |
| `Year_Birth` | Year of birth of the customer |
| `Education` | Highest education level achieved |
| `Marital_Status` | Marital status of the customer |
| `Income` | Yearly household income |
| `Kidhome` | Number of small children in household |
| `Teenhome` | Number of teenagers in household |
| `Dt_Customer` | Date when customer enrolled with SmartCart |

### 2. Purchase Behavior (Amount Spent)

| Feature | Description |
|---------|-------------|
| `MntWines` | Amount spent on wine products |
| `MntFruits` | Amount spent on fruits |
| `MntMeatProducts` | Amount spent on meat products |
| `MntFishProducts` | Amount spent on fish products |
| `MntSweetProducts` | Amount spent on sweet products |
| `MntGoldProds` | Amount spent on gold products |

### 3. Purchase Behavior (Frequency)

| Feature | Description |
|---------|-------------|
| `NumDealsPurchases` | Purchases made using discounts |
| `NumWebPurchases` | Purchases made through website |
| `NumCatalogPurchases` | Purchases made through catalog |
| `NumStorePurchases` | Purchases made in physical stores |
| `NumWebVisitsMonth` | Number of visits to website per month |

### 4. Customer Feedback & Engagement

| Feature | Description |
|---------|-------------|
| `Recency` | Number of days since last purchase |
| `Complain` | Customer complained in last 2 years (1 = Yes, 0 = No) |

## 🚀 Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Rachgit28/SmartCart.git
   cd SmartCart
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```
   
   **Core dependencies:**
   ```bash
   pip install jupyter pandas numpy matplotlib seaborn scikit-learn scipy
   ```

### Running the Project

1. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

2. **Open and run the notebook**
   - Open `smartcart.ipynb`
   - Execute cells sequentially or select `Cell > Run All`

## 🔬 Project Workflow

### 1. Data Loading & Exploration
- Import and inspect the customer dataset
- Understand feature distributions and statistics
- Identify data quality issues

### 2. Data Preprocessing
- Handle missing values
- Remove outliers
- Encode categorical variables
- Feature engineering (create derived features like total spending, customer tenure, etc.)

### 3. Exploratory Data Analysis (EDA)
- Visualize customer demographics
- Analyze purchase patterns across product categories
- Investigate correlations between features
- Identify key behavioral indicators

### 4. Feature Scaling & Selection
- Normalize/standardize numerical features
- Select relevant features for clustering
- Apply dimensionality reduction if needed (PCA)

### 5. Clustering Analysis
- **K-Means Clustering**: Partition customers into k clusters
- **Hierarchical Clustering**: Build dendrograms to understand cluster relationships
- **DBSCAN**: Identify clusters of varying density
- Determine optimal number of clusters using Elbow Method and Silhouette Score

### 6. Cluster Evaluation
- Calculate silhouette scores
- Analyze cluster characteristics
- Validate cluster quality

### 7. Insights & Recommendations
- Profile each customer segment
- Generate marketing strategies for each cluster
- Identify high-value and at-risk customers

## 🛠️ Technologies Used

### Programming & Environment
- **Python 3.8+** - Primary programming language
- **Jupyter Notebook** - Interactive development environment

### Data Science Libraries
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **matplotlib** - Data visualization
- **seaborn** - Statistical data visualization
- **scikit-learn** - Machine learning algorithms
- **scipy** - Scientific computing

### Machine Learning Algorithms
- K-Means Clustering
- Hierarchical Clustering (Agglomerative)
- DBSCAN (Density-Based Spatial Clustering)
- Principal Component Analysis (PCA)

## 📈 Results & Insights

*[This section will be populated based on your analysis results]*

### Customer Segments Identified

**Segment 1: [Name]**
- Characteristics: [Description]
- Size: [Percentage of total customers]
- Marketing Strategy: [Recommendations]

**Segment 2: [Name]**
- Characteristics: [Description]
- Size: [Percentage of total customers]
- Marketing Strategy: [Recommendations]

**Segment 3: [Name]**
- Characteristics: [Description]
- Size: [Percentage of total customers]
- Marketing Strategy: [Recommendations]

### Key Findings
- 📊 Finding 1: [Description]
- 💰 Finding 2: [Description]
- 🎯 Finding 3: [Description]

### Business Impact
- **Improved Marketing ROI**: [Expected improvement]
- **Customer Retention**: [Expected retention rate increase]
- **Personalization**: [Number of targeted campaigns possible]

## 📁 Project Structure

```
SmartCart/
│
├── smartcart.ipynb              # Main Jupyter notebook with clustering analysis
├── .gitignore                   # Git ignore file
├── README.md                    # Project documentation (this file)
├── requirements.txt             # Python dependencies
│
├── data/                        # Data directory
│   ├── raw/                     # Raw customer data
│   └── processed/               # Cleaned and preprocessed data
│
├── models/                      # Saved clustering models
│
├── visualizations/              # Generated plots and charts
│   ├── eda/                     # Exploratory data analysis plots
│   └── clusters/                # Cluster visualization plots
│
└── reports/                     # Analysis reports and insights
    └── cluster_profiles.md      # Detailed cluster characteristics
```

## 🎓 Learning Outcomes

This project demonstrates:
- Application of unsupervised machine learning techniques
- Customer segmentation strategies for e-commerce
- Data preprocessing and feature engineering
- Clustering algorithm comparison and evaluation
- Data-driven marketing insights generation
- Business problem solving using AI/ML

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

### Ideas for Contributions
- Add more clustering algorithms (Gaussian Mixture Models, Mean Shift, etc.)
- Implement real-time customer segmentation API
- Create interactive dashboards using Plotly or Streamlit
- Add predictive models for customer lifetime value
- Implement A/B testing framework for marketing strategies

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Rachit Rajput**
- GitHub: [@Rachgit28](https://github.com/Rachgit28)
- Email: rachitrajput@gmail.com

## 🙏 Acknowledgments

- Dataset provided as part of the SmartCart customer segmentation challenge
- Inspired by real-world e-commerce customer analytics needs
- Built as an AI/ML engineering project for demonstrating unsupervised learning applications

## 📚 References

- [Scikit-learn Clustering Documentation](https://scikit-learn.org/stable/modules/clustering.html)
- [Customer Segmentation Best Practices](https://www.sciencedirect.com/topics/computer-science/customer-segmentation)
- [E-commerce Analytics Strategies](https://research.google/pubs/)

---

**⭐ If you find this project helpful, please consider giving it a star!**

**💡 Have suggestions or found a bug? Feel free to open an issue!**
