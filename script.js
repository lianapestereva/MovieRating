document.addEventListener('DOMContentLoaded', function() {
    // Кнопка "Выбрать фильм" на главной странице
    const startBtn = document.getElementById('start-btn');
    if (startBtn) {
        startBtn.addEventListener('click', function() {
            window.location.href = '/questionnaire';
        });
    }

    // Обработка формы опросника
    const form = document.getElementById('questionnaire-form');
    if (form) {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(form);
            const data = {
                humor: formData.get('humor'),
                plot: formData.get('plot'),
                visuals: formData.get('visuals'),
                acting: formData.get('acting'),
                sound: formData.get('sound'),
                age: [],
                type: [],
                genres: []
            };

            // Собираем выбранные checkbox значения
            document.querySelectorAll('input[name="age"]:checked').forEach(el => {
                data.age.push(el.value);
            });
            
            document.querySelectorAll('input[name="type"]:checked').forEach(el => {
                data.type.push(el.value);
            });
            
            document.querySelectorAll('input[name="genres"]:checked').forEach(el => {
                data.genres.push(el.value);
            });

            // Отправка данных на сервер
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
});