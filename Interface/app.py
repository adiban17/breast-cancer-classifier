import streamlit as st
import numpy as np
import pickle

def project_summary():
    st.title("🔬 Detecting Breast Cancer using Machine Learning")

    st.markdown("""
This breast cancer classification project uses a dataset sourced from the **UCI Machine Learning Repository**, containing patient tumor characteristics and diagnosis labels. We trained three machine learning models — Logistic Regression, K-Nearest Neighbors (KNN), and Support Vector Classifier (SVC) — all of which were **hyper-parameter tuned** to optimize their performance.

The training results demonstrate a significant improvement from basic models to tuned estimators:

- **Logistic Regression**: basic accuracy 95%, tuned accuracy 95%
- **KNN**: basic accuracy ~68%, tuned accuracy 85%
- **SVC**: basic accuracy ~67%, tuned accuracy 96.7%

To reduce false negatives and enhance early detection of malignant tumors, the model with the **best recall score** was selected for final deployment.

---

**Malignant Tumors:**  
Malignant tumors are cancerous growths that have the potential to invade nearby tissues and spread to other parts of the body through the bloodstream or lymphatic system. They tend to grow rapidly and can be life-threatening if not detected and treated early.

**Benign Tumors:**  
Benign tumors are non-cancerous growths that typically grow slowly and do not spread to other parts of the body. While generally less serious, they can still cause health problems depending on their size and location and might require medical attention.
""")

def load_model_and_scaler():
    with open('Exported Models/Best Model/scaler.pkl', 'rb') as f_scaler:
        scaler = pickle.load(f_scaler)
    with open('Exported Models/Best Model/cancer_classification_model.pkl', 'rb') as f_model:
        model = pickle.load(f_model)
    return scaler, model

def user_inputs():
    st.sidebar.header("Adjust Tumor Features")

    with st.sidebar.expander("Mean Features", expanded=True):
        radius_mean = st.number_input("Radius Mean", 0.0, 30.0, 14.0, 0.01,
                                      help="Mean of distances from center to points on the perimeter")
        texture_mean = st.number_input("Texture Mean", 0.0, 40.0, 20.0, 0.01)
        smoothness_mean = st.number_input("Smoothness Mean", 0.0, 1.0, 0.1, 0.0001)
        compactness_mean = st.number_input("Compactness Mean", 0.0, 1.0, 0.1, 0.0001)
        concavity_mean = st.number_input("Concavity Mean", 0.0, 1.0, 0.1, 0.0001)
        concave_points_mean = st.number_input("Concave Points Mean", 0.0, 1.0, 0.05, 0.0001)
        symmetry_mean = st.number_input("Symmetry Mean", 0.0, 1.0, 0.2, 0.0001)
        fractal_dimension_mean = st.number_input("Fractal Dimension Mean", 0.0, 1.0, 0.06, 0.0001)

    with st.sidebar.expander("Standard Error (SE) Features", expanded=False):
        radius_se = st.number_input("Radius SE", 0.0, 5.0, 0.5, 0.01)
        texture_se = st.number_input("Texture SE", 0.0, 10.0, 1.2, 0.01)
        smoothness_se = st.number_input("Smoothness SE", 0.0, 0.5, 0.01, 0.0001)
        compactness_se = st.number_input("Compactness SE", 0.0, 0.5, 0.02, 0.0001)
        concavity_se = st.number_input("Concavity SE", 0.0, 0.5, 0.02, 0.0001)
        concave_points_se = st.number_input("Concave Points SE", 0.0, 0.5, 0.01, 0.0001)
        symmetry_se = st.number_input("Symmetry SE", 0.0, 0.5, 0.02, 0.0001)
        fractal_dimension_se = st.number_input("Fractal Dimension SE", 0.0, 0.1, 0.003, 0.0001)

    with st.sidebar.expander("Worst Features", expanded=False):
        radius_worst = st.number_input("Radius Worst", 0.0, 40.0, 16.0, 0.01)
        smoothness_worst = st.number_input("Smoothness Worst", 0.0, 1.0, 0.15, 0.0001)
        symmetry_worst = st.number_input("Symmetry Worst", 0.0, 1.0, 0.3, 0.0001)
        fractal_dimension_worst = st.number_input("Fractal Dimension Worst", 0.0, 1.0, 0.08, 0.0001)

    features = {
        "radius_mean": radius_mean,
        "texture_mean": texture_mean,
        "smoothness_mean": smoothness_mean,
        "compactness_mean": compactness_mean,
        "concavity_mean": concavity_mean,
        "concave points_mean": concave_points_mean,
        "symmetry_mean": symmetry_mean,
        "fractal_dimension_mean": fractal_dimension_mean,
        "radius_se": radius_se,
        "texture_se": texture_se,
        "smoothness_se": smoothness_se,
        "compactness_se": compactness_se,
        "concavity_se": concavity_se,
        "concave points_se": concave_points_se,
        "symmetry_se": symmetry_se,
        "fractal_dimension_se": fractal_dimension_se,
        "radius_worst": radius_worst,
        "smoothness_worst": smoothness_worst,
        "symmetry_worst": symmetry_worst,
        "fractal_dimension_worst": fractal_dimension_worst
    }
    return features

def main():
    project_summary()

    scaler, model = load_model_and_scaler()
    features = user_inputs()

    features_no_index = [
        "radius_mean", "texture_mean", "smoothness_mean", "compactness_mean",
        "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
        "radius_se", "texture_se", "smoothness_se", "compactness_se",
        "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se",
        "radius_worst", "smoothness_worst", "symmetry_worst", "fractal_dimension_worst"
    ]

    input_values = [features[feat] for feat in features_no_index]
    input_array = np.array(input_values).reshape(1, -1)
    scaled_features = scaler.transform(input_array)
    expanded_input = np.hstack([np.array([[0]]), scaled_features])  # Add dummy zero column

    prediction = model.predict(expanded_input)[0]
    prediction_proba = model.predict_proba(expanded_input)[0]

    diagnosis = "🛑 Malignant " if prediction == 1 else "✅ Benign"
    confidence = prediction_proba[prediction] * 100

    st.subheader("Prediction Result")
    st.write(f"**Diagnosis:** {diagnosis}")
    st.write(f"**Confidence:** {confidence:.2f}%")
    st.progress(int(confidence))

    if prediction == 1:
        st.warning("⚠️ Please consult your healthcare provider for further advice.")
    else:
        st.info("This indicates a benign tumor, but regular check-ups are recommended.")

if __name__ == "__main__":
    main()