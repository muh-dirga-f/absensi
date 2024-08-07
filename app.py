# app.py
import base64
import os
import uuid
import cv2
from flask import Flask, render_template, request, redirect, url_for, session, Response
import sqlite3
from datetime import datetime
from werkzeug.utils import secure_filename
import numpy as np

app = Flask(__name__)
app.secret_key = 'absensi-perpus'

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def init_db():
    conn = sqlite3.connect('mahasiswa.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS mahasiswa (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nama TEXT,
            nim TEXT,
            jurusan TEXT,
            angkatan TEXT,
            foto TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            password TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS absensi (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            id_mahasiswa INTEGER,
            tanggal TEXT,
            FOREIGN KEY (id_mahasiswa) REFERENCES mahasiswa(id)
        )
    ''')
    conn.commit()
    conn.close()


init_db()


@app.route('/')
def index():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    return render_template('login.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect('mahasiswa.db')
        cursor = conn.cursor()
        cursor.execute(
            'SELECT * FROM user WHERE username = ? AND password = ?', (username, password))
        user = cursor.fetchone()
        conn.close()

        if user:
            session['logged_in'] = True
            return redirect(url_for('absensi'))
        else:
            return "Username atau Password salah"

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))


@app.route('/absensi', methods=['GET', 'POST'])
def absensi():
    if 'logged_in' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        id_mahasiswa = request.form['id_mahasiswa']
        tanggal = datetime.now().strftime('%Y-%m-%d')

        conn = sqlite3.connect('mahasiswa.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO absensi (id_mahasiswa, tanggal)
            VALUES (?, ?)
        ''', (id_mahasiswa, tanggal))
        conn.commit()
        conn.close()

        return redirect(url_for('absensi'))

    conn = sqlite3.connect('mahasiswa.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT absensi.id, mahasiswa.nim, mahasiswa.nama, mahasiswa.jurusan, mahasiswa.angkatan, absensi.tanggal
        FROM absensi
        JOIN mahasiswa ON absensi.id_mahasiswa = mahasiswa.id
    ''')
    data = cursor.fetchall()

    cursor.execute('SELECT id, nama FROM mahasiswa')
    mahasiswa_list = cursor.fetchall()
    conn.close()

    current_year = datetime.now().year
    years = [current_year - i for i in range(6)]

    return render_template('absensi.html', data=data, mahasiswa_list=mahasiswa_list, years=years)


@app.route('/mahasiswa', methods=['GET', 'POST'])
def mahasiswa():
    if 'logged_in' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        nama = request.form['nama']
        nim = request.form['nim']
        jurusan = request.form['jurusan']
        angkatan = request.form['angkatan']

        foto_data_url = request.form['foto_data_url']
        if foto_data_url:
            header, encoded = foto_data_url.split(",", 1)
            data = base64.b64decode(encoded)
            extension = header.split('/')[1].split(';')[0]
            random_filename = f"{uuid.uuid4().hex}.{extension}"
            foto_path = os.path.join(
                app.config['UPLOAD_FOLDER'], random_filename)
            with open(foto_path, "wb") as f:
                f.write(data)

        conn = sqlite3.connect('mahasiswa.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO mahasiswa (nama, nim, jurusan, angkatan, foto)
            VALUES (?, ?, ?, ?, ?)
        ''', (nama, nim, jurusan, angkatan, foto_path))
        conn.commit()
        conn.close()

        return redirect(url_for('mahasiswa'))

    conn = sqlite3.connect('mahasiswa.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM mahasiswa')
    data = cursor.fetchall()
    conn.close()
    return render_template('mahasiswa.html', data=data)


@app.route('/delete_mahasiswa/<int:id>', methods=['POST'])
def delete_mahasiswa(id):
    if 'logged_in' not in session:
        return redirect(url_for('login'))

    conn = sqlite3.connect('mahasiswa.db')
    cursor = conn.cursor()
    cursor.execute('SELECT foto FROM mahasiswa WHERE id = ?', (id,))
    foto_path = cursor.fetchone()[0]

    if os.path.exists(foto_path):
        os.remove(foto_path)

    cursor.execute('DELETE FROM mahasiswa WHERE id = ?', (id,))
    conn.commit()
    conn.close()

    return redirect(url_for('mahasiswa'))


@app.route('/recognize')
def recognize():
    return render_template('recognize.html')


def generate_frames():
    video_capture = cv2.VideoCapture(0)

    conn = sqlite3.connect('mahasiswa.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id, nama, foto FROM mahasiswa')
    mahasiswa = cursor.fetchall()
    # conn.close()

    known_faces = []
    known_names = []

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    for idx, row in enumerate(mahasiswa):
        nama = row[1]
        foto_path = row[2]
        known_image = cv2.imread(foto_path, cv2.IMREAD_GRAYSCALE)
        faces = face_cascade.detectMultiScale(
            known_image, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = known_image[y:y+h, x:x+w]
            known_faces.append(face)
            known_names.append(idx)

    print(known_faces)
    recognizer.train(known_faces, np.array(known_names))

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = gray_frame[y:y+h, x:x+w]
            label, confidence = recognizer.predict(face)
            name = "Unknown"
            if confidence < 70:
                id_mahasiswa = mahasiswa[label][0]
                name = mahasiswa[label][1]
                tanggal = datetime.now().strftime('%Y-%m-%d')

                # Periksa apakah sudah absen hari ini
                cursor.execute('''
                    SELECT COUNT(*)
                    FROM absensi
                    WHERE id_mahasiswa = ? AND tanggal = ?
                ''', (id_mahasiswa, tanggal))
                result = cursor.fetchone()

                if result[0] == 0:
                    cursor.execute('''
                        INSERT INTO absensi (id_mahasiswa, tanggal)
                        VALUES (?, ?)
                    ''', (id_mahasiswa, tanggal))
                    conn.commit()

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, name, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    video_capture.release()


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
