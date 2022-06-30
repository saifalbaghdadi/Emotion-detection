import os
from threading import local
import cv2
import random
import numpy as np
from werkzeug.utils import secure_filename
from urllib.request import Request, urlopen
from flask import Flask, render_template, Response, request, redirect, flash
from visual.camera import VideoCamera
from visual.Graphical_Visualisation import Emotion_Analysis

app = Flask(__name__)

# Define some global parameters.
# When rendering files, I set the maximum cache control lifetime to zero number of seconds to refresh the cache.
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def gen(camera):
    "" "Helps in Passing frames from Web Camera to server"""
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def allowed_file(filename):
    """ Checks the file format when file is uploaded"""
    return ('.' in filename and
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)


###################################################################################
# List of texts to be shown to the user based on their feelings
# I used website(https://www.brainyquote.com), in order to get quotes.
QHappy = {'There is only one happiness in this life, to love and be loved.': 'Happiness lies in the joy of achievement and the thrill of creative effort.',
     'Spread love everywhere you go. Let no one ever come to you without leaving happier.': 'Happiness is when what you think, what you say, and what you do are in harmony.',
     'Be happy for this moment. This moment is your life.': 'There is nothing like deep breaths after laughing that hard, Nothing in the world like a sore stomach for the right reasons.' }
QSad = {'An arrow can only be shot by pulling it backward. So when life is dragging you back with difficulties, it means that it is going to launch you into something great.':
     'Grief never ends... but it changes. It is a passage, not a place to stay. Grief is not a sign of weakness, nor a lack of faith... it is the price of love.' ,
    'Sometimes when things are falling apart they may actually be falling into place.': 'Sometimes life does not give you something you want, not because you do not deserve it, but because you deserve more.',
    'Whoever is trying to bring you down, is already below you.':'My entire life can be described in one sentence: it didn not go as planned, and that is okay..'}
QDisgust = {'The vine bears three kinds of grapes: the first of pleasure, the second of intoxication, the third of disgust.': 'Love turns, with a little indulgence, to indifference or disgust; hatred alone is immortal.',
     'Disgust is often more deeply buried than envy and anger, but it compounds and intensifies the other negative emotions.': 'If there is a God, the phrase that must disgust him is - holy war.',
     'Any newspaper, from the first line to the last, is nothing but a web of horrors, I cannot understand how an innocent hand can touch a newspaper without convulsing in disgust.': 'The time will come when it will disgust you to look in the mirror.' }
QNeutral = {'If you are neutral in situations of injustice, you have chosen the side of the oppressor. If an elephant has its foot on the tail of a mouse and you say that you are neutral, the mouse will not appreciate your neutrality.': 'Neutral men are the devil is allies.',
     'Time is neutral and does not change things. With courage and initiative, leaders change things.': 'Washing one is hands of the conflict between the powerful and the powerless means to side with the powerful, not to be neutral.',
     'People who demand neutrality in any situation are usually not neutral but in favor of the status quo.': 'Journalists typically don not carry weapons, even in war zones, for fear of compromising their status as neutral observers, If you are armed, the theory goes, other armed people will consider you a target.' }
QFear = {'It is better to be feared than loved, if you cannot be both.': 'If we can acknowledge our fear, we can realize that right now we are okay.',
     'We can easily forgive a child who is afraid of the dark; the real tragedy of life is when men are afraid of the light. ': 'The only thing we have to fear is fear itself. ',
     'Always do what you are afraid to do. ': 'The oldest and strongest emotion of mankind is fear, and the oldest and strongest kind of fear is fear of the unknown.' }
QAngry = {'You can choose to not let little things upset you. ': 'For every minute you remain angry, you give up sixty seconds of peace of mind. ',
     'When angry count to ten before you speak. If very angry, count to one hundred. ': 'Speak when you are angry - and you will make the best speech you will ever regret. ',
     'It is so important to realize that every time you get upset, it drains your emotional energy, Losing your cool makes you tired, Getting angry a lot messes with your health. ': 'People will not have time for you if you are always angry or complaining.' }
QSurprise = {'The backbone of surprise is fusing speed with secrecy.': 'Do not tell people how to do things, tell them what to do and let them surprise you with their results.',
     'The husband who decides to surprise his wife is often very much surprised himself.': 'The secret to humor is surprise. ',
     'A good, sympathetic review is always a wonderful surprise. ': 'Machines take me by surprise with great frequency Alan Turing Art must take reality by surprise.' }
# Function to show random text quote from a list of quotes.
def mood(result):
    if result=="Happy":
        return random.choice(list(QHappy.values()))
    elif result=="Sad":
        return random.choice(list(QSad.values()))
    elif result=="Disgust":
        return random.choice(list(QDisgust.values()))
    elif result=="Neutral":
        return random.choice(list(QNeutral.values()))
    elif result=="Fear":
        return random.choice(list(QFear.values()))
    elif result=="Angry":
        return random.choice(list(QAngry.values()))
    elif result=="Surprise":
        return random.choice(list(QSurprise.values()))

###################################################################################

@app.route('/')
def Start():
    """ Renders the Home Page """

    return render_template('Start.html')


@app.route('/video_feed')
def video_feed():
    """ A route that returns a streamed response needs to return a Response object
    that is initialized with the generator function."""

    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/RealTime', methods=['POST'])
def RealTime():
    """ Video streaming (Real Time Image from WebCam Video) home page."""

    return render_template('RealTime.html')

@app.route('/takeimage', methods=['POST'])
def takeimage():
    """ Captures Images from WebCam, saves them, does Emotion Analysis & renders. """

    v = VideoCamera()
    _, frame = v.video.read()
    save_to = "static/"
    cv2.imwrite(save_to + "capture" + ".jpg", frame)

    result = Emotion_Analysis("capture.jpg")

    # When Classifier could not detect any Face.
    if len(result) == 1:
        return render_template('NoDetection.html', orig=result[0])

    sentence=mood(result[3])
    return render_template('Visual.html', orig=result[0], pred=result[1], bar=result[2],suggest=result[3],sentence=sentence)

@app.route('/ManualUpload', methods=['POST'])
# Manual Uploading of Images via URL or Upload .
def ManualUpload():
    return render_template('ManualUpload.html')

@app.route('/uploadimage', methods=['POST'])
# Load Image from System, does Emotion Analysis & renders.
def uploadimage():
    if request.method == 'POST':

        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        # If user uploads the correct Image File
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            result = Emotion_Analysis(filename)

            # When Classifier could not detect any Face.
            if len(result) == 1:

                return render_template('NoDetection.html', orig=result[0])

            sentence=mood(result[3])
            return render_template('Visual.html', orig=result[0], pred=result[1], bar=result[2],suggest=result[3],sentence=sentence)


@app.route('/imageurl', methods=['POST'])
# Fetch Image from URL Provided, does Emotion Analysis & renders.
def imageurl():
    url = request.form['url']
    req = Request(url,
                  headers={'User-Agent': 'chrome/5.0'})
    # Reading and Saving it to the static Folder
    webpage = urlopen(req).read()
    arr = np.asarray(bytearray(webpage), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    save_to = "static/"
    cv2.imwrite(save_to + "url.jpg", img)

    result = Emotion_Analysis("url.jpg")

    # When Classifier could not detect any Face.
    if len(result) == 1:
        return render_template('NoDetection.html', orig=result[0])

    sentence=mood(result[3])
    return render_template('Visual.html', orig=result[0], pred=result[1], bar=result[2],suggest=result[3],sentence=sentence)


if __name__ == '__main__':
    app.run(host='', port='9900' ,debug=False)
