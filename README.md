# Introduction to Machine Learning

This folder provides a foundational introduction to machine learning concepts, covering the fundamental differences between rule-based systems and ML, as well as supervised and unsupervised learning approaches.

## 📚 Contents

### `IntroductionToML.ipynb`
**Comprehensive Introduction to Machine Learning Fundamentals**

This notebook serves as an entry point to machine learning, covering:

- **What is Machine Learning?**
  - Definition and formal mathematical representation
  - Model as a function: $\hat{y} = f_{\theta}(x)$
  - Contrast with traditional rule-based programming

- **Rule-Based Systems vs Machine Learning**
  - Explicit if-then rules vs learned patterns
  - When to use each approach
  - Advantages and limitations

- **Supervised Learning**
  - Concept: Learning from labeled data (inputs → outputs)
  - **Practical Example**: Predicting student pass/fail based on study hours and attendance
  - **Logistic Regression Implementation**:
    - Mathematical foundation (sigmoid function)
    - Model training and evaluation
    - Accuracy and confusion matrix interpretation
    - Making predictions on new data

- **Unsupervised Learning**
  - Concept: Finding patterns in unlabeled data
  - **Practical Example**: K-Means clustering of student cost-of-living data
  - Grouping students by spending patterns (rent, groceries, transport)
  - Cluster interpretation and real-world applications

**Key Learning Outcomes:**
- Understand the fundamental concepts of machine learning
- Differentiate between supervised and unsupervised learning
- Implement basic logistic regression for binary classification
- Apply K-Means clustering to discover patterns in data
- Interpret model predictions and cluster results

## 🛠️ Technologies Used

- **pandas**: Data manipulation
- **scikit-learn**: Machine learning algorithms
  - `LogisticRegression`: Binary classification
  - `KMeans`: Clustering
  - `train_test_split`: Data splitting
  - `accuracy_score`, `confusion_matrix`: Model evaluation
- **numpy**: Numerical operations
- **matplotlib**: Data visualization

## 📋 Prerequisites

- Basic Python programming
- Understanding of basic statistics
- Familiarity with pandas DataFrames
- No prior machine learning experience required

## 🚀 Getting Started

1. **Install Required Packages**:
   ```bash
   pip install pandas numpy scikit-learn matplotlib
   ```

2. **Run the Notebook**:
   - Follow the cells sequentially
   - Execute code cells to see results
   - Read markdown cells for explanations

## 📊 Datasets Used

1. **Student Pass/Fail Dataset** (Supervised Learning):
   - Features: `hours_studied`, `attendance`
   - Target: `passed` (binary: 0 or 1)
   - 30 samples for training and testing

2. **Student Cost-of-Living Dataset** (Unsupervised Learning):
   - Features: `Rent ($)`, `Groceries ($)`, `Transport ($)`
   - 12 students grouped into 2 clusters
   - No labels required

## 💡 Key Concepts Explained

### Logistic Regression
- **Sigmoid Function**: Maps any real number to probability [0, 1]
- **Decision Boundary**: Threshold at 0.5 for binary classification
- **Model Training**: Learning parameters from data
- **Evaluation Metrics**: Accuracy, confusion matrix

### K-Means Clustering
- **Clusters**: Groups of similar data points
- **Cluster Centers**: Representative points for each group
- **Unsupervised**: No labels needed, algorithm discovers patterns

## 🎯 Learning Path

This notebook is designed as the **first step** in a machine learning journey:

1. **Start Here**: Understand ML basics
2. **Next Steps**: 
   - Deep dive into specific algorithms (see `logistic_regression/`)
   - Explore classification techniques (see `Classification/`)
   - Learn model evaluation (see `model_evaluation/`)

## 📝 Notes

- All datasets are synthetic and generated within the notebook
- Results are reproducible with fixed random seeds
- Code is well-commented for educational purposes
- Mathematical formulas are included for deeper understanding
