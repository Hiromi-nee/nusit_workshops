from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/classify', methods = ['POST'])
def classify():
    json_data = request.get_json()
    
    # # print data received
    # print(json_data) 
    
    # # data type
    # print(type(json_data))
    
    # convert to json before sending response
    return jsonify(json_data)

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 8881)