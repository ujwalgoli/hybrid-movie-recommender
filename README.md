# 🎬 Hybrid Movie Recommender

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A personalized movie recommendation engine combining **User-based CF**, **Item-based CF**, **Genre similarity**, and **Year similarity**. Built with Python, Streamlit, and the MovieLens dataset, featuring dynamic taste profiling, recommendation explanations, and poster fetching via the OMDb API.

---

## 📌 Features

| Feature | Description |
|---------|-------------|
| Hybrid Recommendation | Combines User CF + Item CF + Genre similarity + Year similarity for accurate suggestions. |
| Poster Fetching | Automatically fetches movie posters using OMDb API with caching. |
| Filters | Filter recommendations by Genre, Release Year, and IMDb Rating. |
| Taste Profile | Generates a short summary of user preferences based on recommendations. |
| Explanation per Recommendation | Displays why each movie is recommended. |
| Export Recommendations | Download recommendations and profile as a `.txt` file for demo or interview use. |
| Clean UI | Minimal, interactive, and responsive interface using Streamlit. |

---

## 📊 Tech Stack

- **Frontend**: Streamlit  
- **Backend / Logic**: Python (`pandas`, `numpy`, `scikit-learn`)  
- **Dataset**: MovieLens 100K  
- **API**: OMDb API for poster, year, genre, and IMDb ratings  

---

## 🖥️ Screenshots

> Replace these placeholders with actual screenshots of your app.

| Home / Input | Recommendations |
|--------------|----------------|
| ![Home](https://via.placeholder.com/350x200?text=Home+Screen) | ![Recommendations](https://via.placeholder.com/350x200?text=Recommendations) |

---


