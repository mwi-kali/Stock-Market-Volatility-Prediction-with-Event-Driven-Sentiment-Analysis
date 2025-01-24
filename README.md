# Stock Market Volatility Prediction Using Event-Driven Sentiment Analysis

This repository offers a Jupyter Notebook meant to forecast stock market volatility using a hybrid strategy that blends historical stock data with event-driven sentiment analysis. The research uses advanced Natural Language Processing (NLP) techniques, technical indicators, and deep learning models to improve forecast accuracy.

## Technologies and Libraries Used
- Python libraries (`dash`, `plotly`, `scikit-learn`, `statsmodels`, `ta`, `tensorflow`, `yfinance`)
- NLP and Sentiment Analysis ('Transformers' library with 'Finbert-tone' for financial sentiment analysis.)
- Deep learning frameworks ('Keras' and TensorFlow backend)
- Visualisation ('matplotlib','seaborn', 'plotly')

## How to Run the Project 

1. Install the necessary packages with:

```bash 
!pip Install dash, plotly, scikit-learn, statsmodels, tensorflow, and yfinance.
```

2. Clone the repository and launch the Jupyter notebook.
3. Ensure access to API keys (for example, NewsAPI for article retrieval).
4. Run the cells sequentially, following the notebook's sections.

## Future Enhancements 

**Incorporating Additional Data Sources**

Adding macroeconomic factors, social media sentiment, and options market data can enhance the model's capacity to reflect market intricacies.

**Advanced Model Architectures**

Using Transformer-based models and GRU networks can help the model understand complex temporal patterns and improve pattern identification.

**Regularization Techniques**

Using greater dropout rates, L2 regularization, and early halting can assist prevent overfitting and increase model generalization.

**Feature Importance**

Using interpretability methods such as SHAP (Shapley Additive Explanations) or LIME (Local Interpretable Model-agnostic Explanations) can reveal the features that have the greatest influence on the model's predictions, increasing transparency, which is especially important in financial environments.
