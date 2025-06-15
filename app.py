from app import app
import os
import json

if __name__ == '__main__':
    #os.makedirs('data', exist_ok=True)
    
    for file in ['user_answers.json', 'recommendations.json']:
        if not os.path.exists(f'data/{file}'):
            with open(f'data/{file}', 'w') as f:
                json.dump([], f)
    
    app.run(debug=True)
