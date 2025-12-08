# 🎬 Netflix Movie Recommender

A graph-based movie recommendation system using bipartite graphs and collaborative filtering algorithms.

**Academic Project** • Algorithms and Data Structures 2 • Data Science and AI • ETF Sarajevo

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![NetworkX](https://img.shields.io/badge/NetworkX-2.8-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Technologies](#technologies)
- [Architecture](#architecture)
- [Algorithms](#algorithms)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Screenshots](#screenshots)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project implements a **movie recommendation system** using **bipartite graph theory** and **collaborative filtering**. Built as part of the **Algorithms and Data Structures 2** course at the Faculty of Electrical Engineering (ETF), University of Sarajevo, it demonstrates practical applications of graph algorithms in real-world recommendation systems.

The system analyzes user-movie relationships in a bipartite graph structure to find similar users and recommend movies based on shared preferences using Jaccard similarity and common neighbors algorithms.

---

## ✨ Features

### Core Functionality
- 🎯 **Smart Recommendations** - Jaccard similarity & Common Neighbors algorithms
- 👥 **Similar Users Detection** - Find users with matching movie preferences
- 🎨 **Interactive Visualization** - Plotly-based bipartite graph exploration
- 🔍 **Real-time Search** - Filter through 9,000+ movies instantly
- ⭐ **Rating Filter** - Set minimum rating thresholds (0-5 stars)
- 🎭 **Genre Filtering** - Filter by specific movie genres
- ❤️ **Like System** - Select multiple movies to build preference profile

### User Experience
- 🌙 **Netflix Dark Theme** - Modern, familiar interface
- 📱 **Responsive Design** - Works on desktop, tablet, and mobile
- 🚀 **Fast Performance** - Optimized graph algorithms
- 💾 **Session Persistence** - Maintain state across page navigation

---

## 🎥 Demo

**Live Demo:** [https://ln.run/NetflixRecommendation](https://netflix-movie-recommender.onrender.com) *(may take 30s to wake up)*

### Quick Tour:
1. 🔍 Search and select movies you like
2. ⚙️ Configure recommendation settings
3. 🎯 Get personalized recommendations
4. 📊 Visualize the recommendation graph
5. 👥 See which users have similar taste

---

## 🛠️ Technologies

### Backend
- **Python 3.11** - Core programming language
- **Flask 3.0** - Web framework
- **NetworkX 2.8** - Graph algorithms and data structures
- **Pandas 2.0** - Data manipulation and analysis
- **Plotly 5.0** - Interactive graph visualizations

### Frontend
- **HTML5 & CSS3** - Structure and styling
- **JavaScript (ES6+)** - Client-side interactivity
- **Font Awesome 6.5** - Icon library

### Data
- **MovieLens Dataset** - 100,000+ ratings from 600+ users on 9,000+ movies
- **CSV Format** - ratings.csv, movies.csv, tags.csv, links.csv

---

## 🏗️ Architecture

### Bipartite Graph Structure

```
       Users                Movies
    ┌─────────┐          ┌─────────┐
    │ User 1  │──────────│ Movie A │
    └─────────┘          └─────────┘
         │                    │
         │               ┌─────────┐
         └───────────────│ Movie B │
                         └─────────┘
    ┌─────────┐              │
    │ User 2  │──────────────┘
    └─────────┘          ┌─────────┐
         │               │ Movie C │
         └───────────────└─────────┘
```

- **Nodes**: Users and Movies (two distinct sets)
- **Edges**: User-Movie ratings (threshold: ≥3.5 stars)
- **Properties**: Movie metadata (title, genres, avg rating)

### Recommendation Flow

1. **User Selection** → Select liked movies via UI
2. **Graph Query** → Find users who liked the same movies
3. **Similarity Calculation** → Compute Jaccard or CN scores
4. **Candidate Filtering** → Apply genre/rating filters
5. **Ranking** → Sort by score and present top-N results
6. **Visualization** → Generate interactive bipartite graph

---

## 🧮 Algorithms

### 1. Jaccard Similarity (2-Hop)

Measures overlap between user movie sets:

```
J(A,B) = |A ∩ B| / |A ∪ B|
```

**Use case:** Balanced similarity metric, good for diverse recommendations

### 2. Common Neighbors (CN)

Counts shared movie preferences:

```
CN(u,v) = |N(u) ∩ N(v)|
```

**Use case:** Emphasizes strong overlaps, popular movie bias

### 3. Rating Prioritization

Optional weighted scoring:

```
Score_weighted = Score × (avg_rating / 5.0)
```

**Use case:** Boost highly-rated movies in recommendations

---

## 📦 Installation

### Prerequisites
- Python 3.11 or higher
- pip (Python package manager)
- Git

### Step 1: Clone Repository
```bash
git clone https://github.com/ehadziabdic/NetflixRecommendation.git
cd NetflixRecommendation
```

### Step 2: Create Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Set Environment Variables
```bash
copy .env.example .env
```

Edit `.env` and add your secret key:
```env
SECRET_KEY=your_generated_secret_key_here
FLASK_DEBUG=True
PORT=5000
```

Generate secret key:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

---

## 🚀 Usage

### Run Locally
```bash
cd web
python app.py
```

Open browser: `http://localhost:5000`

### Using the Application

1. **Select Movies**
   - Search for movies using the search bar
   - Click ❤️ heart icon to like movies
   - Selected movies appear in "Your Movies" section

2. **Configure Settings**
   - **Top # Recommendations**: Number of results (1-100)
   - **Genre**: Filter by specific genre or "All"
   - **Rating Limit**: Minimum average rating (0-5)
   - **Algorithm**: Toggle between Jaccard/Common Neighbors
   - **Prioritize Rating**: Weight scores by movie ratings

3. **Get Recommendations**
   - Click "Get recommendations" button
   - View results in sortable table
   - See rank, title, score, genres, and ratings

4. **Visualize Graph**
   - Click "Visualize Recommendation Graph"
   - Interactive Plotly visualization shows:
     - 🟡 You (virtual user node)
     - 🟢 Similar users (circles)
     - 🟦 Liked movies (squares)
     - 🟩 Recommended movies (squares)
   - Info panel displays similar users with shared movie counts

---

## 📁 Project Structure

```
NetflixRecommendation/
├── web/                      # Web application
│   ├── app.py               # Flask application & routes
│   ├── templates/           # HTML templates
│   │   ├── index.html       # Main page (movie selection)
│   │   ├── recommendations.html  # Results page
│   │   └── graph.html       # Graph visualization
│   └── static/              # Static assets
│       ├── css/
│       │   └── style.css    # Netflix dark theme styling
│       ├── js/
│       │   └── script.js    # Client-side interactivity
│       └── icons/
│           └── favcon.png   # Favicon
├── src/                     # Core algorithms
│   ├── graph.py            # Graph construction & data loading
│   ├── scoring.py          # Recommendation algorithms
│   └── graphvis.py         # Plotly visualization generator
├── res/                     # MovieLens dataset
│   ├── ratings.csv         # User ratings (100K entries)
│   ├── movies.csv          # Movie metadata (9K movies)
│   ├── tags.csv            # User-generated tags
│   └── links.csv           # External IDs (IMDB, TMDB)
├── test/                    # Testing & analysis
│   ├── test.py             # Basic recommendation tests
│   ├── recommend.py        # CLI recommendation tool
│   └── graphshow.py        # Matplotlib graph visualization
├── docs/                    # Documentation
│   └── EDA.ipynb           # Exploratory Data Analysis notebook
├── requirements.txt         # Python dependencies
├── .env.example            # Environment template
├── .gitignore              # Git ignore rules
├── DEPLOYMENT.md           # Deployment guide
└── README.md               # This file
```

---

## 📸 Screenshots

### Main Page - Movie Selection
![Main Page](https://i.ibb.co/v4ZZFGx9/image.png)

### Recommendations Table
![Results](https://i.ibb.co/Xr120xzK/image.png)

### Interactive Graph Visualization
![Graph](https://i.ibb.co/kgGqL1NQ/image.png)

---

## 🌐 Deployment

### Deploy on Render (Free)

1. **Push to GitHub:**
   ```bash
   git push origin main
   ```

2. **Create Render Account:**
   - Go to [render.com](https://render.com)
   - Sign up with GitHub

3. **Deploy:**
   - New → Web Service
   - Connect repository
   - Build: `pip install -r requirements.txt`
   - Start: `cd web && python app.py`
   - Add environment variable: `SECRET_KEY`

4. **Access:**
   - Your app: `https://your-app.onrender.com`

## 👥 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **MovieLens Dataset** - GroupLens Research @ University of Minnesota
- **ETF Sarajevo** - Course instructors and teaching assistants
- **NetworkX Team** - Excellent graph algorithms library
- **Flask Community** - Web framework and documentation

---

## 📧 Contact

**Emin Hadžiabdić**  
Data Science and AI Student  
ETF Sarajevo

- GitHub: [@ehadziabdic](https://github.com/ehadziabdic)
- Project Link: [https://github.com/ehadziabdic/NetflixRecommendation](https://github.com/ehadziabdic/NetflixRecommendation)

---

<div align="center">

⭐ Star this repo if you found it helpful!

</div>
