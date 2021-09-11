# from flask import Flask
# from flask import request, jsonify
#
# app = Flask(__name__)
#
# @app.route('/test', methods=['GET', 'POST'])
# def get():
#     name = request.args.get('name', '')
#     if name == 'xuefeilong':
#         age = 21
#     else:
#         age = 'valid name'
#     return jsonify(
#         data={name: age},
#         extra={
#             'total': '120'
#         }
#     )





# _*_ coding=utf-8 _*_
from flask import Flask
from flask import request, jsonify
import json
app = Flask(__name__)
@app.route('/post', methods=['GET', 'POST'])
def post():
    data = request.get_json()
    print(data)
    return jsonify(
        data=json.dumps(data),
        extra={
            'message': 'success'
        }
    )

if __name__ == '__main__':
    app.debug = True
    app.run()