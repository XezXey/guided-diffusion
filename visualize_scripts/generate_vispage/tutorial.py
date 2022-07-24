from flask import Flask
  
app = Flask(__name__)
  
# Pass the required route to the decorator.
@app.route("/hello")
def hello():
    return "Hello, Welcome to GeeksForGeeks"
    
@app.route("/")
def index():
    return "Homepage of GeeksForGeeks"
  
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4747, debug=True)