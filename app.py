from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    try:
        # Get data from the request
        data = request.json
        input_data = data['inputData']

        # Process the data (processing logic here)

        # Send a response back to the front-end
        response_message = f"Data received and processed: {input_data}"
        return jsonify({'message': response_message})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
