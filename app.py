from prediction_service import prediction
from flask import Flask, render_template, request,jsonify
import os
from application_logging.logger import App_Logger

webapp_root = "webapp"

static_dir = os.path.join(webapp_root, "static")
template_dir = os.path.join(webapp_root, "templates")
logger = App_Logger()
file_object=open('Prediction_log.txt','+a')


app = Flask(__name__, static_folder=static_dir,template_folder=template_dir)

@app.route("/",methods=['GET','POST'])
def index():
      if request.method == 'POST' :

            try:

                        #print('file selected')
                        #if 'file' in request.files :
                            path = request.files['file']
                            path = path.filename
                            print(path)
                            logger.log(file_object,"File name read successfully")
                            response = prediction.form_response(path)
                            return render_template("index1.html", response=response)
                        #else:
                            #print('File not found')


            except Exception as e:
                    error = {"error":e}
                    return render_template("404.html",error=error)
 
      else:
          return render_template('index1.html')
         

if __name__ =="__main__":
    app.run(host="0.0.0.0",port=5000,debug=True)