from flask import Flask, render_template, request, redirect, url_for, flash

app = Flask(__name__)

@app.route('/')
def index():
    #Show some text on the index page without html and show it on the browser
    return "Hello World"

if __name__ == '__main__':
    app.run(debug=True)