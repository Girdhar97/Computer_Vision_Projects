# Behavioral Cloning – Self-Driving Car 🚗

An end-to-end deep learning project to **clone human driving behavior** in the Udacity Self-Driving Car Simulator using a Convolutional Neural Network (CNN) that predicts steering angles from camera images. [web:32]

---

## 🌟 Features

- End-to-end **behavioral cloning**: learn steering directly from images and human driving logs. [web:32]
- Uses **Udacity Self-Driving Car Simulator** for data collection and evaluation. [web:32]
- **NVIDIA-inspired CNN architecture** for steering prediction. [web:21]
- Multi-camera (left, center, right) with **steering angle correction**. [web:21]
- **Data augmentation**: flipping, correction, cropping, normalization. [web:18]
- Real-time **autonomous driving** via `drive.py` and websocket connection. [file:25][web:35]

---

## 🏗️ Project Structure

```bash
Behaviour-cloning-master/
├── Behavioral_Cloning_main.ipynb   # Main Jupyter notebook (data, model, training)
├── drive.py                        # Script to run the trained model in the simulator
├── model.h5                        # Trained Keras model (not in repo by default)
├── video.py                        # Script to convert recorded frames to a video
├── output_images/                  # Sample outputs (plots, example images)
│   ├── left_*.jpg
│   ├── center_*.jpg
│   ├── right_*.jpg
│   ├── carnd-using-multiple-cameras.png
│   └── *.png (histograms, examples, etc.)
├── Data/                           # Driving data folder (NOT INCLUDED – user must add)
│   ├── driving_log.csv
│   └── IMG/
│       ├── center_*.jpg
│       ├── left_*.jpg
│       └── right_*.jpg
└── README.md

📦 Requirements
Prerequisites
Python 3.7+

pip

Udacity Self-Driving Car Simulator (desktop app) [web:32]

(Optional but recommended) Virtual environment (venv/conda)

Python Dependencies
Install the main dependencies:

pip install tensorflow keras numpy pillow flask python-socketio eventlet matplotlib opencv-python

💾 Dataset Setup
The project expects a Udacity behavioral cloning dataset structure: images from three cameras and a driving_log.csv describing each frame. [web:32][web:22]

1️⃣ Create Data Folder

In the project root:

mkdir Data
mkdir Data/IMG

2️⃣ Options to Get Data

Option A – Use Udacity’s Sample Data
Download the sample behavioral cloning dataset from the Udacity project page or other mirror. [web:32]

Extract contents so that you have:

Data/
├── driving_log.csv
└── IMG/
    ├── center_2016_*.jpg
    ├── left_2016_*.jpg
    └── right_2016_*.jpg

Option B – Record Your Own Data in Simulator

Launch Udacity Self-Driving Car Simulator.

Select Training Mode. [web:22]

Drive manually around the track:

Try to stay centered.

Include turns, recovery from edges, etc.

Copy the generated driving_log.csv and IMG/ folder into this project’s Data/ directory.

⚙️ Configuration

In Behavioral_Cloning_main.ipynb, update the data directory:

data_dir = '/absolute/or/relative/path/to/Data'
# Example (Linux/WSL):
# data_dir = '/mnt/g/Projects/Computer_Vision_Projects/Behaviour-cloning-master/Data'

This path should point to the folder that contains driving_log.csv and IMG/. [web:34]

If you change folder names, update accordingly.

🧠 Model & Approach (High Level)
Input: RGB image from center camera (after cropping & normalization). [web:21]

Output: Single continuous steering angle (in radians). [web:21]

Architecture: NVIDIA-style CNN

5 convolutional layers (5×5 and 3×3 kernels, strided) + ReLU. [web:21]

3 fully connected layers.

Optional normalization layer at input. [web:36]

Multi-camera: Left and right cameras included with steering correction:

Left: angle + 0.15

Right: angle - 0.15 [web:21]

Preprocessing:

Cropping sky/hood regions (focus on road). [web:18]

Normalization to range around 
−
0.5
−0.5 to 
0.5
0.5. [web:18]

Horizontal flip + sign inversion of steering angle. [web:21]

Training:

Train/validation split (e.g., 80/20). [web:18]

Loss: Mean Squared Error (MSE). [web:21]

Optimizer: Adam (lr ≈ 1e-3). [web:21]

Data generator with fit / fit_generator for efficient loading. [web:18]

The full implementation and experiment outputs are available inside Behavioral_Cloning_main.ipynb.

🚀 How to Run – Step by Step

1. Start Jupyter:

cd Behaviour-cloning-master
jupyter notebook

2. Open Behavioral_Cloning_main.ipynb.

3. Ensure data_dir is correctly set to your Data folder.

4. Run all cells in order:

- Data loading and visualization
- Preprocessing and augmentation
- Model definition (NVIDIA architecture)
- Training loop

5. At the end, the notebook should save the model:

model.save('model.h5')

This model.h5 file will be used for autonomous driving.

2️⃣ Running Autonomous Mode (drive.py)
drive.py opens a websocket server that communicates with the Udacity simulator, receives images, runs the model, and sends steering + throttle commands back. [file:25][web:35]

Usage:

python drive.py model.h5
# or, to record frames:
python drive.py model.h5 run1

- model.h5 – path to your trained model file. [web:33][web:35]
- run1 – optional folder name where each frame image will be saved.

Steps:

In terminal:

bash
cd Behaviour-cloning-master
python drive.py model.h5
Launch Udacity Self-Driving Car Simulator.

Select AUTONOMOUS MODE and connect. [web:33]

If everything is correct:

Simulator will stream camera images to drive.py. [file:25]

drive.py will:

Decode image from base64. [file:25]

Convert to NumPy array.

Pass it to model.predict(...). [file:25]

Set a throttle (e.g., 0.2 constant). [file:25]

Send steering + throttle back to simulator.

You will see steering angle and throttle printed in terminal. [file:25]

If you passed an image folder (e.g., run1), images will be saved with a timestamp. [file:25]

3️⃣ Creating a Video from a Recording
If you recorded a run using:

bash
python drive.py model.h5 run1
You can create a video using video.py:

bash
python video.py run1
# Optionally specify FPS:
# python video.py run1 --fps 30
This will combine the images in run1/ into a video file (e.g., run1.mp4).

🔍 Common Issues & Fixes
Problem	Possible Cause	Fix
FileNotFoundError: driving_log.csv	Wrong data_dir path	Update data_dir to point to Data correctly.
Car doesn’t move in simulator	Version mismatch or simulator not in autonomous mode	Ensure simulator is in Autonomous Mode and drive.py is running. [web:31][web:33]
socketio errors	Incompatible python-socketio version	Try python-socketio==4.2.1 as some setups require older versions. [web:31]
Car goes off track quickly	Model undertrained or preprocessing mismatch	Check that the same cropping/normalization used during training is applied in drive.py.
High bias to straight driving	Dataset dominated by zero steering	Consider downsampling 0-angle samples or augment turning data. [web:18]
🧪 Commands Summary
bash
# 1. Install dependencies
pip install tensorflow keras numpy pillow flask python-socketio eventlet matplotlib opencv-python

# 2. Create data folder
mkdir Data
mkdir Data/IMG

# 3. Place driving_log.csv and images in Data/

# 4. (Optional) Launch Jupyter notebook and train
jupyter notebook Behavioral_Cloning_main.ipynb

# 5. Run autonomous driving (no recording)
python drive.py model.h5

# 6. Run autonomous driving (record frames)
python drive.py model.h5 run1

# 7. Convert recorded run to video
python video.py run1
# or with FPS
python video.py run1 --fps 30
📄 Notes
Large data and model.h5 are not stored in git; keep them locally or provide external download links.

This project is primarily educational and based on the Udacity Behavioral Cloning project and NVIDIA’s end-to-end self-driving research. [web:32][web:21]

🙏 Acknowledgments
Udacity – Self-Driving Car Nanodegree Behavioral Cloning Project. [web:32]

NVIDIA – End-to-End Learning for Self-Driving Cars. [web:21]

Original Udacity drive.py and simulator tooling. [web:35]