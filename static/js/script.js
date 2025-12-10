// Netflix Movie Recommender - Interactive Features

document.addEventListener('DOMContentLoaded', function() {
    // State management
    const state = {
        likedMovies: new Set(),
        allMovies: []
    };

    // DOM Elements
    const searchInput = document.getElementById('search-input');
    const movieList = document.getElementById('movie-list');
    const yourMoviesList = document.getElementById('your-movies-list');
    const sliderValue = document.getElementById('slider-value');
    const ratingSlider = document.getElementById('rating-slider');
    const algorithmBtns = document.querySelectorAll('.algorithm-btn');
    const ratingBtns = document.querySelectorAll('.rating-btn');
    const recommendForm = document.getElementById('recommend-form');

    // Initialize
    init();

    function init() {
        // Store all movies from the DOM
        const movieItems = document.querySelectorAll('.movie-item');
        movieItems.forEach(item => {
            const movieId = item.dataset.movieId;
            const movieTitle = item.dataset.movieTitle;
            state.allMovies.push({ id: movieId, title: movieTitle, element: item });
        });

        // Restore liked movies from server (if returning from results/graph page)
        restoreLikedMovies();

        // Setup event listeners
        setupSearchFilter();
        setupHeartButtons();
        setupSlider();
        setupToggleButtons();
        setupFormSubmit();
    }

    // Restore liked movies from hidden input (populated by server)
    function restoreLikedMovies() {
        const likedMoviesData = document.getElementById('liked-movies-data');
        if (likedMoviesData && likedMoviesData.value) {
            const likedIds = likedMoviesData.value.split(',').filter(x => x);
            likedIds.forEach(movieId => {
                const movie = state.allMovies.find(m => m.id == movieId);
                if (movie) {
                    state.likedMovies.add(movieId);
                    // Update heart button
                    const heartBtn = movie.element.querySelector('.heart-btn');
                    if (heartBtn) {
                        heartBtn.classList.remove('unliked');
                        heartBtn.classList.add('liked');
                        heartBtn.innerHTML = '<i class="fa-solid fa-heart"></i>';
                    }
                    // Add to Your Movies
                    addToYourMovies(movieId, movie.title);
                }
            });
        }
    }

    // Real-time search filter
    function setupSearchFilter() {
        searchInput.addEventListener('input', function() {
            const searchTerm = this.value.toLowerCase().trim();
            
            state.allMovies.forEach(movie => {
                const titleMatch = movie.title.toLowerCase().includes(searchTerm);
                
                if (titleMatch || searchTerm === '') {
                    movie.element.classList.remove('hidden');
                } else {
                    movie.element.classList.add('hidden');
                }
            });
        });
    }

    // Heart button functionality
    function setupHeartButtons() {
        movieList.addEventListener('click', function(e) {
            if (e.target.classList.contains('heart-btn')) {
                const movieItem = e.target.closest('.movie-item');
                const movieId = movieItem.dataset.movieId;
                const movieTitle = movieItem.dataset.movieTitle;
                
                if (state.likedMovies.has(movieId)) {
                    // Unlike
                    state.likedMovies.delete(movieId);
                    e.target.classList.remove('liked');
                    e.target.classList.add('unliked');
                    e.target.innerHTML = '<i class="fa-regular fa-heart"></i>';
                    removeFromYourMovies(movieId);
                } else {
                    // Like
                    state.likedMovies.add(movieId);
                    e.target.classList.remove('unliked');
                    e.target.classList.add('liked');
                    e.target.innerHTML = '<i class="fa-solid fa-heart"></i>';
                    addToYourMovies(movieId, movieTitle);
                }
            }
        });
    }

    // Add movie to "Your Movies" list
    function addToYourMovies(movieId, movieTitle) {
        // Remove empty message if exists
        const emptyMsg = yourMoviesList.querySelector('.empty-message');
        if (emptyMsg) {
            emptyMsg.remove();
        }

        // Create movie item
        const movieItem = document.createElement('div');
        movieItem.className = 'your-movie-item';
        movieItem.dataset.movieId = movieId;
        movieItem.innerHTML = `
            <span class="movie-title">${movieTitle}</span>
            <button type="button" class="remove-btn" data-movie-id="${movieId}">✕</button>
        `;

        yourMoviesList.appendChild(movieItem);

        // Setup remove button
        movieItem.querySelector('.remove-btn').addEventListener('click', function() {
            removeFromYourMovies(movieId);
            updateHeartButton(movieId, false);
        });
    }

    // Remove movie from "Your Movies" list
    function removeFromYourMovies(movieId) {
        const movieItem = yourMoviesList.querySelector(`[data-movie-id="${movieId}"]`);
        if (movieItem) {
            movieItem.remove();
        }

        state.likedMovies.delete(movieId);

        // Show empty message if no movies
        if (state.likedMovies.size === 0) {
            yourMoviesList.innerHTML = '<div class="empty-message">List of movies liked by user currently</div>';
        }
    }

    // Update heart button state
    function updateHeartButton(movieId, liked) {
        const movieItem = movieList.querySelector(`[data-movie-id="${movieId}"]`);
        if (movieItem) {
            const heartBtn = movieItem.querySelector('.heart-btn');
            if (liked) {
                heartBtn.classList.remove('unliked');
                heartBtn.classList.add('liked');
                heartBtn.innerHTML = '<i class="fa-solid fa-heart"></i>';
            } else {
                heartBtn.classList.remove('liked');
                heartBtn.classList.add('unliked');
                heartBtn.innerHTML = '<i class="fa-regular fa-heart"></i>';
            }
        }
    }

    // Rating slider
    function setupSlider() {
        ratingSlider.addEventListener('input', function() {
            sliderValue.textContent = this.value;
        });
    }

    // Toggle buttons (Algorithm and Rating Priority)
    function setupToggleButtons() {
        // Algorithm toggle
        algorithmBtns.forEach(btn => {
            btn.addEventListener('click', function() {
                algorithmBtns.forEach(b => b.classList.remove('active'));
                this.classList.add('active');
            });
        });

        // Rating priority toggle
        ratingBtns.forEach(btn => {
            btn.addEventListener('click', function() {
                ratingBtns.forEach(b => b.classList.remove('active'));
                this.classList.add('active');
            });
        });
    }

    // Form submission
    function setupFormSubmit() {
        recommendForm.addEventListener('submit', function(e) {
            // Validate that at least one movie is liked
            if (state.likedMovies.size === 0) {
                e.preventDefault();
                alert('Please like at least one movie before getting recommendations!');
                return;
            }

            // Add liked movies to form data
            const likedMoviesInput = document.getElementById('liked-movies-input');
            likedMoviesInput.value = Array.from(state.likedMovies).join(',');
        });
    }
});
