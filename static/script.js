document.addEventListener('DOMContentLoaded', function() {
    //Кнопка "Выбрать фильм"
    const startBtn = document.getElementById('start-btn');
    if (startBtn) {
        startBtn.addEventListener('click', function() {
            window.location.href = '/questionnaire';
        });
    }

    //Опросника
    const form = document.getElementById('questionnaire-form');
    if (form) {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(form);
            const data = {
                criteria: {
                    humor: formData.get('humor') || "0",
                    plot: formData.get('plot') || "0",
                    visuals: formData.get('visuals') || "0",
                    acting: formData.get('acting') || "0",
                    sound: formData.get('sound') || "0"
                },
                filters: {
                    age: [],
                    type: [],
                    genres: []
                }
            };

            //Сбор значений фильтров
            document.querySelectorAll('input[name="age"]:checked').forEach(el => {
                data.filters.age.push(el.value);
            });
            
            document.querySelectorAll('input[name="type"]:checked').forEach(el => {
                data.filters.type.push(el.value);
            });
            
            document.querySelectorAll('input[name="genres"]:checked').forEach(el => {
                data.filters.genres.push(el.value);
            });

            //Отправка данных на сервер
            fetch('/process_answers', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    window.location.href = '/results';
                }
            })
            .catch(error => {
                console.error('Error:', error);
            });
        });
    }

    //Звездного рейтинга
    const starRatings = document.querySelectorAll('.star-rating');
    starRatings.forEach(rating => {
        const stars = rating.querySelectorAll('input[type="radio"]');
        stars.forEach(star => {
            star.addEventListener('change', function() {
                console.log(`Selected value: ${this.value}`);
            });
        });
    });
});
