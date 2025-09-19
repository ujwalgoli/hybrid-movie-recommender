# Hybrid Movie Recommender

A personalized movie recommendation engine combining **User-based CF**, **Item-based CF**, **Genre similarity**, and **Year similarity**. Built with Python, Streamlit, and the MovieLens dataset, featuring dynamic taste profiling, recommendation explanations, and poster fetching via the OMDb API.

---

## Features

- **Hybrid Recommendation**: Combines User CF + Item CF + Genre similarity + Year similarity for accurate recommendations.  
- **Poster Fetching**: Automatically fetches movie posters using OMDb API with caching.  
- **Filters**: Filter recommendations by Genre, Release Year, and IMDb Rating.  
- **Taste Profile**: Generates a short summary of user preferences based on recommendations.  
- **Explanation per Recommendation**: Shows why a movie was recommended.  
- **Export Recommendations**: Download recommendations and profile as a `.txt` file.  
- **Minimal, Clean UI** for easy interaction.  

---

## Tech Stack

- **Frontend**: Streamlit  
- **Backend / Logic**: Python (`pandas`, `numpy`, `scikit-learn`)  
- **Dataset**: MovieLens 100K  
- **API**: OMDb API for poster, year, genre, and IMDb ratings  

---

## Installation & Usage

```bash
# Clone the repository
git clone https://github.com/<your-username>/hybrid-movie-recommender.git
cd hybrid-movie-recommender

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
