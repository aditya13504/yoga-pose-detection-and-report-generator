import cv2
import mediapipe as mp
import numpy as np
import os
import joblib
import time
# Removed datetime, re
# Removed tkinter
from sklearn.model_selection import train_test_split # Still needed for train_or_load
from sklearn.neural_network import MLPClassifier # Still needed for train_or_load
from sklearn.preprocessing import StandardScaler # Still needed for train_or_load
from sklearn.metrics import accuracy_score, classification_report # Still needed for train_or_load
from PIL import Image # Needed for imagehash for loading (though not used in main loop now)
import imagehash      # Needed for loading (though not used in main loop now)
import math # Added for angle calculation
from datetime import datetime # Added back for session tracking
# Added yoga_report_generator
try:
    from yoga_report_generator import YogaReportGenerator
except ImportError:
    print("Warning: yoga_report_generator module not found. Report generation will be disabled.")
    YogaReportGenerator = None
# Added pyttsx3
try:
    import pyttsx3
except ImportError:
    print("Error: pyttsx3 library not found.")
    print("Please install it using: pip install pyttsx3")
    # Depending on the OS, additional engines might be needed (like espeak on Linux)
    print("Note: You might also need to install text-to-speech engines like NSSpeechSynthesizer (macOS), SAPI5 (Windows), or espeak (Linux).")
    exit()


# --- Configuration ---
DATASET_PATH = r"./yoga_data" # Keep for training/loading logic
TREE_POSE_FOLDER = os.path.join(DATASET_PATH, "tree") # Keep for training/loading logic
OTHER_POSES_FOLDER = os.path.join(DATASET_PATH, "other") # Keep for training/loading logic
MODEL_FILENAME = "yoga_pose_mlp.joblib"
SCALER_FILENAME = "pose_scaler.joblib"
POSE_CLASSES = ["other", "tree"] # 0: other, 1: tree
REFERENCE_POSE_IMAGE = "reference_tree_pose.jpg" # <<<--- IMPORTANT: User needs to provide this image
CORRECT_POSES_SAVE_DIR = "correct_poses" # Folder to save the final correct frame

MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
CORRECT_POSE_THRESHOLD = 0.80 # Confidence threshold for overall pose classification

# --- Feedback Configuration ---
REFERENCE_DISPLAY_TIME = 10 # Seconds to show the reference pose initially
CORRECTION_TIME_LIMIT = 10 # Seconds before giving specific feedback
FEEDBACK_DEBOUNCE_TIME = 2.0 # Seconds between identical spoken messages
STRAIGHT_LEG_ANGLE_THRESHOLD = 165 # Angle threshold for considering the standing leg straight (degrees)

# --- Perceptual Hashing Configuration (Only needed if training/loading involves it) ---
HASH_SIZE = 8
PERCEPTUAL_HASH_THRESHOLD = 5

# --- Session Data Configuration ---
MAX_SESSION_DURATION = 300 # Maximum session duration in seconds (5 minutes)
MAX_POSE_ATTEMPTS = 20 # Maximum number of pose attempts per session
GENERATE_REPORT_AFTER_SESSION = True # Whether to automatically generate a report after a session

# --- MediaPipe Initialization ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# --- Drawing Styles ---
# Green for reference pose
ref_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3)
ref_connection_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
# Red for user pose (Not drawn anymore, but keep spec in case needed later)
user_drawing_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
user_connection_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)


# --- Text-to-Speech Initialization ---
tts_engine = None
last_spoken_message = ""
last_spoken_time = 0

def initialize_tts():
    """Initializes the pyttsx3 engine."""
    global tts_engine
    try:
        tts_engine = pyttsx3.init()
        print("TTS Engine Initialized.")
        tts_engine.say("Ready to analyze pose.")
        tts_engine.runAndWait() # Wait for speech to finish
        return True
    except Exception as e:
        print(f"Error initializing TTS engine: {e}")
        print("Auditory feedback will be disabled.")
        tts_engine = None
        return False

def speak(text):
    """Speaks the given text using TTS, with debouncing."""
    global tts_engine, last_spoken_message, last_spoken_time
    if not tts_engine:
        return # TTS disabled or failed to initialize

    current_time = time.time()
    if text != last_spoken_message or (current_time - last_spoken_time > FEEDBACK_DEBOUNCE_TIME):
        print(f"Speaking: {text}") # Log what's being said
        try:
            tts_engine.say(text)
            tts_engine.runAndWait() # Blocks until speech is done
            last_spoken_message = text
            last_spoken_time = current_time
        except Exception as e:
            print(f"Error during TTS processing: {e}")

# --- Helper Functions ---

# calculate_perceptual_hash (Only needed if training uses it, keep for now)
def calculate_perceptual_hash(image_np):
    """Calculates the perceptual hash (phash) of a NumPy image array."""
    try:
        image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
        return imagehash.phash(image_pil, hash_size=HASH_SIZE)
    except Exception as e:
        print(f"Error calculating perceptual hash: {e}")
        return None

# extract_landmarks remains the same
def extract_landmarks(image, pose_detector):
    """Processes an image with MediaPipe Pose and extracts landmarks."""
    image.flags.writeable = False # Performance optimization
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(image_rgb)
    image.flags.writeable = True # Make image writeable again for drawing

    if results.pose_landmarks:
        # Flatten landmarks (x, y, z, visibility) into a feature vector
        # Ensure we only use the first 33 landmarks if more are somehow detected
        landmarks = results.pose_landmarks.landmark[:33]
        if len(landmarks) == 33:
             feature_vector = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks]).flatten()
             return feature_vector, results.pose_landmarks # Return raw landmarks too
        else:
             print(f"Warning: Detected {len(landmarks)} landmarks, expected 33. Skipping frame.")
             return None, None # Indicate error due to unexpected landmark count
    else:
        return None, None

# load_data remains the same (needed for training part of train_or_load_model)
def load_data(tree_folder, other_folder):
    """Loads images, extracts landmarks, and creates features (X) and labels (y)."""
    features = []
    labels = []
    pose_detector_load = None
    try:
         pose_detector_load = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

         print(f"Loading data from {tree_folder} (Label: 1)")
         if not os.path.isdir(tree_folder):
             print(f"Warning: Directory not found {tree_folder}. Skipping.")
         else:
             for img_name in os.listdir(tree_folder):
                 img_path = os.path.join(tree_folder, img_name)
                 image = cv2.imread(img_path)
                 if image is None:
                     print(f"Warning: Could not read image {img_path}. Skipping.")
                     continue
                 landmarks_vec, _ = extract_landmarks(image, pose_detector_load)
                 if landmarks_vec is not None:
                     features.append(landmarks_vec)
                     labels.append(1) # 1 for 'tree' pose
                 else:
                     print(f"Warning: No pose detected in {img_path}. Skipping.")


         print(f"Loading data from {other_folder} (Label: 0)")
         if not os.path.isdir(other_folder):
             print(f"Warning: Directory not found {other_folder}. Skipping.")
         else:
             for img_name in os.listdir(other_folder):
                 img_path = os.path.join(other_folder, img_name)
                 image = cv2.imread(img_path)
                 if image is None:
                     print(f"Warning: Could not read image {img_path}. Skipping.")
                     continue
                 landmarks_vec, _ = extract_landmarks(image, pose_detector_load)
                 if landmarks_vec is not None:
                     features.append(landmarks_vec)
                     labels.append(0) # 0 for 'other' poses
                 else:
                     print(f"Warning: No pose detected in {img_path}. Skipping.")

    finally:
        if pose_detector_load:
            pose_detector_load.close() # Ensure closure

    if not features:
         raise ValueError("No features extracted! Check dataset paths and image content.")

    # Ensure all feature vectors have the correct shape (33*4=132)
    expected_shape = 33 * 4
    # Convert to numpy array first to check shape easily
    try:
        features_np = np.array(features, dtype=float) # Ensure float type
        if features_np.ndim > 1 and features_np.shape[1] != expected_shape:
             print(f"Warning during data loading: Feature vectors have shape {features_np.shape[1]}, expected {expected_shape}. Filtering.")
             # Filter based on the original list before it was potentially made ragged by errors
             original_indices = [i for i, vec in enumerate(features) if isinstance(vec, np.ndarray) and vec.shape[0] == expected_shape]
             if not original_indices:
                 raise ValueError("No valid feature vectors found after filtering during data loading.")
             features = [features[i] for i in original_indices]
             labels = [labels[i] for i in original_indices]
             features_np = np.array(features, dtype=float) # Recreate array
             print(f"Proceeding with {len(features)} valid samples after loading.")
        elif features_np.ndim == 1 and len(features) > 0: # Handle case where only one sample might have loaded incorrectly
            print("Warning: Feature data seems one-dimensional. Rechecking individual samples.")
            original_indices = [i for i, vec in enumerate(features) if isinstance(vec, np.ndarray) and vec.shape[0] == expected_shape]
            if not original_indices:
                 raise ValueError("No valid feature vectors found after filtering single samples.")
            features = [features[i] for i in original_indices]
            labels = [labels[i] for i in original_indices]
            features_np = np.array(features, dtype=float)
            print(f"Proceeding with {len(features)} valid samples after single sample check.")


    except ValueError as ve:
        # Handle cases where array creation fails (e.g., ragged array)
        print(f"ValueError during feature array creation: {ve}. Filtering based on shape.")
        original_indices = [i for i, vec in enumerate(features) if isinstance(vec, np.ndarray) and vec.shape[0] == expected_shape]
        if not original_indices:
             raise ValueError("No valid feature vectors found after filtering ragged array attempt.")
        features = [features[i] for i in original_indices]
        labels = [labels[i] for i in original_indices]
        features_np = np.array(features, dtype=float) # Try creating array again
        print(f"Proceeding with {len(features)} valid samples after filtering.")


    return features_np, np.array(labels)


# train_or_load_model remains mostly the same
def train_or_load_model():
    """Trains a new model if one doesn't exist, otherwise loads it."""
    if not os.path.exists(MODEL_FILENAME) or not os.path.exists(SCALER_FILENAME):
        print("Model/Scaler not found. Training a new model...")
        if not os.path.exists(TREE_POSE_FOLDER) or not os.path.exists(OTHER_POSES_FOLDER):
             raise FileNotFoundError(f"Dataset folders '{TREE_POSE_FOLDER}' or '{OTHER_POSES_FOLDER}' not found. Required for training.")

        X, y = load_data(TREE_POSE_FOLDER, OTHER_POSES_FOLDER)

        if len(X) == 0:
             raise ValueError("No data loaded. Cannot train model.")
        if len(np.unique(y)) < 2:
             raise ValueError(f"Only one class found in labels: {np.unique(y)}. Need at least two classes ('tree' and 'other') to train.")

        print(f"Total samples loaded: {len(X)}")
        # Shape check already done in load_data

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print("Training MLP Classifier...")
        mlp = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=1000,
                            random_state=42, learning_rate_init=0.001, alpha=0.0001, verbose=False)
        mlp.fit(X_train_scaled, y_train)

        print("\nEvaluating Model...")
        y_pred = mlp.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")
        print("Classification Report:\n", classification_report(y_test, y_pred, target_names=POSE_CLASSES))

        print(f"Saving model to {MODEL_FILENAME}")
        joblib.dump(mlp, MODEL_FILENAME)
        print(f"Saving scaler to {SCALER_FILENAME}")
        joblib.dump(scaler, SCALER_FILENAME)

        return mlp, scaler
    else:
        print("Loading existing model and scaler...")
        try:
            mlp = joblib.load(MODEL_FILENAME)
            scaler = joblib.load(SCALER_FILENAME)
            print("Model and scaler loaded successfully.")
            # Basic check for scaler compatibility
            expected_features = 33 * 4
            if hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ != expected_features:
                 raise ValueError(f"Scaler expects {scaler.n_features_in_} features, but {expected_features} are needed.")
            elif hasattr(scaler, 'scale_') and len(scaler.scale_) != expected_features:
                 raise ValueError(f"Scaler expects {len(scaler.scale_)} features, but {expected_features} are needed.")

            return mlp, scaler
        except FileNotFoundError:
             print(f"Error: Could not find model file '{MODEL_FILENAME}' or scaler file '{SCALER_FILENAME}'.")
             print("Please ensure the files are in the correct directory or retrain the model.")
             exit()
        except Exception as e:
            print(f"Error loading model/scaler: {e}")
            print("Consider deleting the .joblib files and retraining.")
            exit()

def load_reference_landmarks(image_path):
    """Loads an image, detects pose landmarks, and returns them."""
    print(f"Loading reference pose from: {image_path}")
    ref_image = cv2.imread(image_path)
    if ref_image is None:
        print(f"Error: Could not load reference image at {image_path}")
        speak("Error loading reference pose image.")
        return None, None

    ref_pose_detector = None
    try:
        ref_pose_detector = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
        _, ref_landmarks = extract_landmarks(ref_image, ref_pose_detector)
        if ref_landmarks:
            print("Reference landmarks extracted successfully.")
            return ref_landmarks, ref_image.shape # Return landmarks and shape
        else:
            print(f"Error: No pose detected in reference image {image_path}")
            speak("Could not detect pose in reference image.")
            return None, None
    finally:
        if ref_pose_detector:
            ref_pose_detector.close()


def calculate_angle(a, b, c):
    """Calculates the angle between three landmarks (angle at b)."""
    # Ensure landmarks have visibility > threshold? Maybe not needed for angle itself.
    try:
        # Calculate vectors using only x and y for 2D angle
        ba = np.array([a.x - b.x, a.y - b.y])
        bc = np.array([c.x - b.x, c.y - b.y])

        # Dot product
        dot_product = np.dot(ba, bc)

        # Magnitudes
        magnitude_ba = np.linalg.norm(ba)
        magnitude_bc = np.linalg.norm(bc)

        # Cosine of the angle
        # Add epsilon to avoid division by zero if magnitudes are zero
        epsilon = 1e-6
        if magnitude_ba * magnitude_bc < epsilon:
             return None # Avoid division by zero if points are coincident

        cosine_angle = dot_product / (magnitude_ba * magnitude_bc)

        # Clip value to prevent errors with acos due to floating point inaccuracies
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

        # Angle in radians
        angle_radians = np.arccos(cosine_angle)

        # Convert to degrees
        angle_degrees = np.degrees(angle_radians)
        return angle_degrees
    except Exception as e:
        print(f"Error calculating angle: {e}")
        return None # Return None on error

# --- Pose Specific Feedback Logic ---

def check_tree_pose_hands(landmarks_list, frame_width, frame_height):
    """
    Checks if hands are in a valid Tree Pose position (Anjali Mudra or raised).
    Returns: True if hands are correct, False otherwise.
    """
    if landmarks_list is None or len(landmarks_list) != 33:
        return False

    l_wrist = landmarks_list[15]
    r_wrist = landmarks_list[16]
    l_shoulder = landmarks_list[11]
    r_shoulder = landmarks_list[12]
    nose = landmarks_list[0]
    l_hip = landmarks_list[23]
    r_hip = landmarks_list[24]

    # Use relative distances (normalized coordinates) to be less dependent on frame size/zoom
    wrist_dist_x_norm = abs(l_wrist.x - r_wrist.x)
    wrist_dist_y_norm = abs(l_wrist.y - r_wrist.y)
    body_center_x = (l_shoulder.x + r_shoulder.x) / 2
    wrists_center_x = (l_wrist.x + r_wrist.x) / 2
    center_offset_x_norm = abs(wrists_center_x - body_center_x)
    shoulders_y = (l_shoulder.y + r_shoulder.y) / 2
    hips_y = (l_hip.y + r_hip.y) / 2
    wrists_y = (l_wrist.y + r_wrist.y) / 2

    # Adjust thresholds for normalized coordinates (these definitely need tuning)
    max_wrist_sep_norm = 0.1 # e.g., 10% of normalized width/height space
    max_center_offset_norm = 0.15
    visibility_threshold = 0.6

    hands_at_chest = False
    if (l_wrist.visibility > visibility_threshold and
        r_wrist.visibility > visibility_threshold and
        l_shoulder.visibility > visibility_threshold and
        r_shoulder.visibility > visibility_threshold and
        l_hip.visibility > visibility_threshold and
        r_hip.visibility > visibility_threshold):

        hands_at_chest = (wrist_dist_x_norm < max_wrist_sep_norm and
                          wrist_dist_y_norm < max_wrist_sep_norm and
                          center_offset_x_norm < max_center_offset_norm and
                          shoulders_y < wrists_y < hips_y) # Y decreases upwards

    hands_above_head = False
    if (l_wrist.visibility > visibility_threshold and
        r_wrist.visibility > visibility_threshold and
        nose.visibility > visibility_threshold):

        hands_above_head = (wrist_dist_x_norm < max_wrist_sep_norm and
                            wrist_dist_y_norm < max_wrist_sep_norm and
                            wrists_y < nose.y) # Y decreases upwards

    is_correct = hands_at_chest or hands_above_head
    return is_correct

def check_tree_pose_leg(landmarks_list):
    """
    Checks if one leg is straight (standing leg) and the other foot is placed correctly.
    Returns: True if leg position is correct, False otherwise, and side (L/R) standing.
    """
    if landmarks_list is None or len(landmarks_list) != 33:
        return False, None

    # Define landmarks
    l_hip = landmarks_list[23]
    l_knee = landmarks_list[25]
    l_ankle = landmarks_list[27]
    l_foot_index = landmarks_list[31] # Tip of the foot

    r_hip = landmarks_list[24]
    r_knee = landmarks_list[26]
    r_ankle = landmarks_list[28]
    r_foot_index = landmarks_list[32]

    visibility_threshold = 0.6

    # Check visibility of key leg landmarks
    l_leg_visible = (l_hip.visibility > visibility_threshold and
                     l_knee.visibility > visibility_threshold and
                     l_ankle.visibility > visibility_threshold)
    r_leg_visible = (r_hip.visibility > visibility_threshold and
                     r_knee.visibility > visibility_threshold and
                     r_ankle.visibility > visibility_threshold)

    if not (l_leg_visible and r_leg_visible):
        #print("Legs not fully visible") # Debug
        return False, None # Cannot determine if both legs aren't sufficiently visible

    # Calculate leg angles (use 2D version)
    left_leg_angle = calculate_angle(l_hip, l_knee, l_ankle)
    right_leg_angle = calculate_angle(r_hip, r_knee, r_ankle)

    if left_leg_angle is None or right_leg_angle is None:
        #print("Could not calculate leg angles") # Debug
        return False, None # Error calculating angles

    left_leg_straight = left_leg_angle > STRAIGHT_LEG_ANGLE_THRESHOLD
    right_leg_straight = right_leg_angle > STRAIGHT_LEG_ANGLE_THRESHOLD

    # Determine standing leg (the straighter one, assuming only one is straight in Tree Pose)
    standing_leg = None
    if left_leg_straight and not right_leg_straight:
        standing_leg = 'L'
        #print("Standing on Left") # Debug
    elif right_leg_straight and not left_leg_straight:
        standing_leg = 'R'
        #print("Standing on Right") # Debug
    elif left_leg_straight and right_leg_straight:
        # Both legs straight? Not Tree Pose. Or standing straight up.
        #print("Both legs straight") # Debug
        return False, None
    else:
        # Neither leg straight enough
        #print("Neither leg straight") # Debug
        return False, None

    # --- Check position of the non-standing foot ---
    foot_placement_ok = False
    # Use ankle and knee landmarks for placement check
    if standing_leg == 'L':
        # Right foot (r_ankle) should be near left knee or left thigh (l_knee or l_hip)
        # Check Y position: r_ankle.y should be between l_hip.y and l_knee.y
        # Check X position: r_ankle.x should be close to l_knee.x (horizontally)
        if r_ankle.visibility > visibility_threshold:
            # Adjust Y check: knee is higher index -> smaller y value
            foot_placement_ok = (l_knee.y < r_ankle.y < l_hip.y and # Y check corrected
                                 abs(r_ankle.x - l_knee.x) < 0.15) # Threshold for horizontal distance (normalized)
            #print(f"Check L Stand: R Ankle Y={r_ankle.y:.2f}, L Knee Y={l_knee.y:.2f}, L Hip Y={l_hip.y:.2f}, X Diff={abs(r_ankle.x - l_knee.x):.2f}, OK={foot_placement_ok}") # Debug
    elif standing_leg == 'R':
        # Left foot (l_ankle) should be near right knee or thigh
         if l_ankle.visibility > visibility_threshold:
            foot_placement_ok = (r_knee.y < l_ankle.y < r_hip.y and # Y check corrected
                                 abs(l_ankle.x - r_knee.x) < 0.15)
            #print(f"Check R Stand: L Ankle Y={l_ankle.y:.2f}, R Knee Y={r_knee.y:.2f}, R Hip Y={r_hip.y:.2f}, X Diff={abs(l_ankle.x - r_knee.x):.2f}, OK={foot_placement_ok}") # Debug

    is_correct = foot_placement_ok # Correctness is primarily based on foot placement for this check
    return is_correct, standing_leg


def get_specific_correction_feedback(landmarks_list, frame_shape, tree_confidence_ok, hands_ok, leg_ok, standing_leg_side):
    """Analyzes incorrect pose and provides specific feedback."""
    if not tree_confidence_ok:
        return "Try to form the Tree Pose shape." # General if classification fails

    # Prioritize feedback based on failed checks
    if not leg_ok:
        # Give feedback about legs first
        if standing_leg_side == 'L':
            # Problem might be right leg placement or left leg not straight
            l_hip, l_knee, l_ankle = landmarks_list[23], landmarks_list[25], landmarks_list[27]
            left_leg_angle = calculate_angle(l_hip, l_knee, l_ankle) or 0
            if left_leg_angle < STRAIGHT_LEG_ANGLE_THRESHOLD:
                return "Straighten your left standing leg."
            else:
                return "Place your right foot higher on your left inner thigh or calf, avoid the knee."
        elif standing_leg_side == 'R':
            r_hip, r_knee, r_ankle = landmarks_list[24], landmarks_list[26], landmarks_list[28]
            right_leg_angle = calculate_angle(r_hip, r_knee, r_ankle) or 0
            if right_leg_angle < STRAIGHT_LEG_ANGLE_THRESHOLD:
                return "Straighten your right standing leg."
            else:
                 return "Place your left foot higher on your right inner thigh or calf, avoid the knee."
        else: # Neither leg is identified as straight and standing
             # Check if maybe both are bent
             l_hip, l_knee, l_ankle = landmarks_list[23], landmarks_list[25], landmarks_list[27]
             r_hip, r_knee, r_ankle = landmarks_list[24], landmarks_list[26], landmarks_list[28]
             left_leg_angle = calculate_angle(l_hip, l_knee, l_ankle) or 180
             right_leg_angle = calculate_angle(r_hip, r_knee, r_ankle) or 180
             if left_leg_angle < STRAIGHT_LEG_ANGLE_THRESHOLD and right_leg_angle < STRAIGHT_LEG_ANGLE_THRESHOLD:
                  return "Straighten one leg to stand on."
             else: # Legs might be visible but angles not calculable or other issue
                  return "Check your leg position for Tree Pose."


    if not hands_ok:
        # Give feedback about hands
        l_wrist, r_wrist = landmarks_list[15], landmarks_list[16]
        nose = landmarks_list[0]
        wrists_y = (l_wrist.y + r_wrist.y) / 2
        wrist_dist_x_norm = abs(l_wrist.x - r_wrist.x)
        visibility_threshold = 0.6

        if l_wrist.visibility < visibility_threshold or r_wrist.visibility < visibility_threshold:
             return "Make sure both hands are visible."

        if wrists_y < nose.y: # Hands seem to be attempting the 'raised' position
            if wrist_dist_x_norm > 0.1: # Check if hands are too far apart
                 return "Bring your hands closer together above your head."
            else:
                 return "Keep hands steady above your head." # General if close but maybe moving
        else: # Hands seem to be attempting the 'chest' position
             if wrist_dist_x_norm > 0.1:
                 return "Bring your hands closer together at your chest."
             else:
                 # Check vertical position relative to shoulders/hips?
                 l_shoulder, r_shoulder = landmarks_list[11], landmarks_list[12]
                 l_hip, r_hip = landmarks_list[23], landmarks_list[24]
                 if l_shoulder.visibility > visibility_threshold and r_shoulder.visibility > visibility_threshold and l_hip.visibility > visibility_threshold and r_hip.visibility > visibility_threshold:
                     shoulders_y = (l_shoulder.y + r_shoulder.y) / 2
                     hips_y = (l_hip.y + r_hip.y) / 2
                     if wrists_y < shoulders_y:
                          return "Lower your hands slightly towards your chest."
                     elif wrists_y > hips_y:
                          return "Raise your hands slightly towards your chest."
                     else:
                          return "Keep hands steady at your chest."
                 else:
                      return "Keep hands steady at your chest, ensure shoulders and hips are visible."

    # If confidence is OK, hands OK, leg OK, but somehow still here? Should not happen.
    return "Adjust your pose slightly." # Generic fallback


# --- Real-Time Feedback Loop ---

def run_realtime_feedback(model, scaler, ref_landmarks, ref_shape):
    """Runs the real-time pose detection loop with auditory feedback."""
    global last_spoken_message, last_spoken_time # Allow modification

    # Initialize session data
    session_data = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "pose_type": "tree",  # Currently only supporting tree pose
        "duration": 0,
        "attempts": 0,
        "correct_poses": [],
        "feedback_history": []
    }
    
    session_start_time = time.time()
    attempt_count = 0
    feedback_messages = set()  # Track unique feedback messages

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        speak("Error: Could not open webcam.")
        return

    pose_detector = None
    try:
        pose_detector = mp_pose.Pose(
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )

        print("\n--- Starting Real-Time Pose Detection with Feedback ---")
        print("Press 'q' to quit.")

        # State variables
        current_state = "SHOWING_REFERENCE" # Initial state
        state_start_time = time.time()
        pose_correct_in_frame = False
        incorrect_pose_start_time = None
        expected_features = 33 * 4

        # Create save directory if it doesn't exist
        os.makedirs(CORRECT_POSES_SAVE_DIR, exist_ok=True)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                speak("Webcam error.")
                break

            frame_height, frame_width, _ = frame.shape
            # Flip frame horizontally for natural movement mirroring
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy() # Create a copy for drawing

            # --- Check session limits ---
            current_time = time.time()
            session_duration = current_time - session_start_time
            session_data["duration"] = int(session_duration)
            
            # Check if session duration or max attempts exceeded
            if session_duration > MAX_SESSION_DURATION or attempt_count >= MAX_POSE_ATTEMPTS:
                print(f"Session limits reached: Duration={session_duration:.1f}s, Attempts={attempt_count}")
                speak("Session complete.")
                break

            # --- State Machine ---
            if current_state == "SHOWING_REFERENCE":
                # Draw reference pose in Green
                if ref_landmarks:
                     mp_drawing.draw_landmarks(
                         display_frame, ref_landmarks, mp_pose.POSE_CONNECTIONS,
                         landmark_drawing_spec=ref_drawing_spec,
                         connection_drawing_spec=ref_connection_spec
                         )
                cv2.putText(display_frame, "Memorize Tree Pose", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                # Speak only once when entering state
                if last_spoken_message != "Memorize the reference pose.":
                    speak("Memorize the reference pose.")

                # Check timer
                if time.time() - state_start_time > REFERENCE_DISPLAY_TIME:
                    current_state = "USER_POSING"
                    state_start_time = time.time()
                    incorrect_pose_start_time = time.time() # Start timer for correction
                    speak("Now try to match the pose.")
                    last_spoken_message = "" # Clear last message to allow immediate feedback

            elif current_state == "USER_POSING":
                pose_correct_in_frame = False # Reset for current frame
                tree_pose_confidence = 0.0 # Default confidence
                user_feedback_text = "Detecting..."
                text_color = (255, 150, 0) # Default orange

                # Detect User Pose
                feature_vector, user_landmarks = extract_landmarks(frame, pose_detector)

                # DO NOT Draw Reference Pose in this state <<< CHANGE
                # if ref_landmarks:
                #     mp_drawing.draw_landmarks(...)

                if feature_vector is not None and user_landmarks:
                    # DO NOT Draw User Pose (Red) on top <<< CHANGE
                    # mp_drawing.draw_landmarks(
                    #     display_frame, user_landmarks, mp_pose.POSE_CONNECTIONS,
                    #     landmark_drawing_spec=user_drawing_spec,
                    #     connection_drawing_spec=user_connection_spec
                    #     )

                    # Check feature vector length before prediction
                    if feature_vector.shape[0] != expected_features:
                         user_feedback_text = "Landmark Count Error"
                         text_color = (0, 0, 255)
                         print(f"Warning: Inconsistent landmark count ({feature_vector.shape[0]}).")
                         incorrect_pose_start_time = None # Reset timer if landmarks are weird
                    else:
                         try:
                             # --- Prediction & Detailed Checks ---
                             feature_vector_scaled = scaler.transform([feature_vector])
                             probabilities = model.predict_proba(feature_vector_scaled)[0]
                             predicted_class_index = np.argmax(probabilities)
                             tree_pose_confidence = probabilities[1] # Confidence for 'tree' <<< Stays here

                             tree_confidence_ok = (predicted_class_index == 1 and tree_pose_confidence >= CORRECT_POSE_THRESHOLD)
                             hands_ok = check_tree_pose_hands(user_landmarks.landmark, frame_width, frame_height)
                             leg_ok, standing_leg_side = check_tree_pose_leg(user_landmarks.landmark)

                             pose_correct_in_frame = tree_confidence_ok and hands_ok and leg_ok

                             # --- Feedback Logic ---
                             if pose_correct_in_frame:
                                 user_feedback_text = f"Tree Pose Correct! (Conf: {tree_pose_confidence:.2f})"
                                 text_color = (0, 255, 0) # Green
                                 speak("Posture done properly.")
                                 
                                 # Record successful pose
                                 attempt_count += 1
                                 session_data["attempts"] = attempt_count
                                 
                                 # Save frame and pose data
                                 timestamp = time.strftime("%Y%m%d_%H%M%S")
                                 save_path = os.path.join(CORRECT_POSES_SAVE_DIR, f"correct_tree_{timestamp}.png")
                                 # Save the original frame (before flipping) or the flipped one? Saving flipped.
                                 cv2.imwrite(save_path, frame) # Save the clean flipped frame
                                 print(f"Correct pose detected and frame saved to {save_path}")
                                 speak("Frame saved.")
                                 
                                 # Add to session data
                                 session_data["correct_poses"].append({
                                     "timestamp": timestamp,
                                     "confidence": float(tree_pose_confidence),
                                     "image_path": save_path,
                                     "standing_leg": standing_leg_side
                                 })
                                 
                                 # Give user a moment to see success, then continue
                                 time.sleep(1)
                                 
                                 # If we've collected enough correct poses or reached attempt limit, exit
                                 if len(session_data["correct_poses"]) >= 3 or attempt_count >= MAX_POSE_ATTEMPTS:
                                     speak("Session complete. Generating report.")
                                     break

                             else: # Pose is incorrect
                                 # Count as an attempt if user is trying (detected landmarks)
                                 if not incorrect_pose_start_time:  # Only count new attempts
                                     attempt_count += 1
                                     session_data["attempts"] = attempt_count
                                 
                                 # Display confidence even when incorrect
                                 user_feedback_text = f"Adjust Pose (Conf: {tree_pose_confidence:.2f})"
                                 text_color = (0, 0, 255) # Red

                                 if incorrect_pose_start_time is None: # Timer not running, start it
                                     incorrect_pose_start_time = time.time()
                                     # Give initial general feedback only if pose detected
                                     if user_landmarks: 
                                         speak("Adjust pose.")
                                         # Add to feedback history
                                         if "Adjust pose." not in feedback_messages:
                                             feedback_messages.add("Adjust pose.")
                                             session_data["feedback_history"].append("Adjust pose.")
                                 else:
                                     # Timer is running, check if limit exceeded
                                     elapsed_incorrect_time = time.time() - incorrect_pose_start_time
                                     if elapsed_incorrect_time > CORRECTION_TIME_LIMIT:
                                         # Get specific feedback
                                         specific_feedback = get_specific_correction_feedback(
                                             user_landmarks.landmark, frame.shape,
                                             tree_confidence_ok, hands_ok, leg_ok, standing_leg_side
                                         )
                                         speak(specific_feedback)
                                         
                                         # Record feedback in session data
                                         if specific_feedback not in feedback_messages:
                                             feedback_messages.add(specific_feedback)
                                             session_data["feedback_history"].append(specific_feedback)
                                             
                                         incorrect_pose_start_time = time.time() # Reset timer after giving feedback

                                     # Display timer countdown only if timer is active
                                     if incorrect_pose_start_time:
                                         remaining_time = max(0, CORRECTION_TIME_LIMIT - elapsed_incorrect_time)
                                         timer_text = f"Correct in: {remaining_time:.1f}s"
                                         cv2.putText(display_frame, timer_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)


                         except Exception as e:
                             print(f"Error during prediction or feedback: {e}")
                             user_feedback_text = "Processing Error"
                             text_color = (0, 0, 255)
                             incorrect_pose_start_time = None # Reset timer on error

                else: # No user pose detected
                    user_feedback_text = "No Pose Detected"
                    text_color = (0, 165, 255) # Orange
                    speak("No pose detected.")
                    
                    # Add to feedback history
                    if "No pose detected." not in feedback_messages:
                        feedback_messages.add("No pose detected.")
                        session_data["feedback_history"].append("No pose detected.")
                    
                    incorrect_pose_start_time = None # Reset timer if pose lost

                # Update display text - Always show the latest confidence if pose detected
                if user_landmarks:
                     # Update text to include current confidence regardless of correctness state
                     if pose_correct_in_frame:
                          user_feedback_text = f"Correct! (Conf: {tree_pose_confidence:.2f})"
                     else:
                          user_feedback_text = f"Adjust (Conf: {tree_pose_confidence:.2f})"
                else: # No pose detected text
                     user_feedback_text = "No Pose Detected"

                # Display attempts and session time
                cv2.putText(display_frame, user_feedback_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2, cv2.LINE_AA)
                cv2.putText(display_frame, f"Attempts: {attempt_count}/{MAX_POSE_ATTEMPTS}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(display_frame, f"Time: {int(session_duration)}s/{MAX_SESSION_DURATION}s", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


            # --- Display Frame ---
            try:
                # Create window beforehand if it doesn't exist to prevent potential issues
                # cv2.namedWindow('Yoga Pose Feedback', cv2.WINDOW_AUTOSIZE) # Removed potential issue source
                cv2.imshow('Yoga Pose Feedback', display_frame)
            except cv2.error as e:
                 if "cvShowImage" in str(e) or "not implemented" in str(e) or "NULL window" in str(e):
                      print("\n--- Cannot Display Video Window (Continuing without it) ---")
                      cv2.imshow = lambda name, frame: None # Replace with dummy
                      cv2.waitKey = lambda delay: ord(' ') # Replace waitKey
                      # Try to destroy the potentially problematic named window if created
                      try: cv2.destroyAllWindows() # Try destroying all windows
                      except: pass
                 else: raise

            # --- Exit Condition ---
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                print("Exit key 'q' pressed.")
                speak("Exiting.")
                break
            elif key != 255: # If any other key is pressed, also exit? Optional.
                 pass

        # --- Session Complete ---
        # Update final session data
        final_duration = time.time() - session_start_time
        session_data["duration"] = int(final_duration)
        print(f"Session completed: {session_data['attempts']} attempts, {len(session_data['correct_poses'])} successful poses, {int(final_duration)}s duration")
        
        # Return session data for reporting
        return session_data

    finally:
        # --- Cleanup ---
        if cap and cap.isOpened(): cap.release()
        cv2.destroyAllWindows()
        if pose_detector: pose_detector.close()
        if tts_engine:
            try: tts_engine.stop()
            except Exception as e: print(f"Minor error stopping TTS engine: {e}")
        print("Resources released.")


# --- Main Execution ---
if __name__ == "__main__":
    try:
        # --- Initialize TTS ---
        print("\n--- Initializing TTS Engine ---")
        tts_enabled = initialize_tts()
        if not tts_enabled: print("Continuing without auditory feedback.")
        
        # --- Initialize Report Generator ---
        report_generator = None
        if YogaReportGenerator is not None:
            try:
                report_generator = YogaReportGenerator()
                print("\n--- Report Generation Enabled ---")
                # Ask if user wants to update profile
                update_profile = input("Would you like to update your user profile before starting? (y/n): ").strip().lower()
                if update_profile == 'y':
                    report_generator.collect_user_data()
            except Exception as e:
                print(f"Error initializing report generator: {e}")
                print("Continuing without report generation.")
                report_generator = None
        else:
            print("\n--- Report Generation Disabled ---")
            print("Install required packages to enable report generation.")

        # --- Load Reference Pose ---
        reference_landmarks, reference_shape = load_reference_landmarks(REFERENCE_POSE_IMAGE)
        if reference_landmarks is None:
            print("Exiting due to reference pose loading error.")
            exit()

        # --- Load Model ---
        print("\n--- Loading/Checking Model ---")
        # (Model loading/training logic remains the same)
        if not os.path.exists(MODEL_FILENAME) or not os.path.exists(SCALER_FILENAME):
             if not os.path.isdir(DATASET_PATH) or not os.path.isdir(TREE_POSE_FOLDER) or not os.path.isdir(OTHER_POSES_FOLDER):
                  print(f"Error: Model/Scaler not found and dataset directories missing/incomplete.")
                  speak("Error: Model or dataset not found.")
                  exit()
        model, scaler = train_or_load_model()


        # --- Run Real-time Feedback ---
        print("\n--- Starting Yoga Session ---")
        session_data = run_realtime_feedback(model, scaler, reference_landmarks, reference_shape)
        
        # --- Generate Report ---
        if report_generator is not None and session_data and GENERATE_REPORT_AFTER_SESSION:
            print("\n--- Generating Reports ---")
            
            # Save session data
            session_file = report_generator.save_session_data(session_data)
            if session_file:
                print(f"Session data saved to: {session_file}")
                
                # Analyze session
                session_analysis = report_generator.analyze_session(session_data)
                print(f"Session analysis complete. Accuracy: {session_analysis['accuracy']:.1f}%")
                
                # Generate comprehensive report across all sessions
                print("Analyzing all sessions for comprehensive report...")
                all_sessions_analysis = report_generator.analyze_all_sessions("tree")
                
                # Generate PDF report
                report_path = report_generator.generate_report_pdf(all_sessions_analysis)
                if report_path:
                    print(f"Complete report generated at: {report_path}")
                    # Try opening the report if on Windows
                    try:
                        if os.name == 'nt':  # Windows
                            os.startfile(report_path)
                        else:  # macOS or Linux
                            import subprocess
                            subprocess.call(["xdg-open", report_path])
                    except Exception as e:
                        print(f"Could not open report automatically: {e}")
                        print(f"Please open the report manually at: {report_path}")
            else:
                print("Failed to save session data. Report generation skipped.")
                
        elif session_data:
            print("Report generation skipped (not enabled).")
        else:
            print("No session data collected. Report generation skipped.")

    except FileNotFoundError as e:
        print(f"\nError: A required file or directory was not found.")
        print(f"Details: {e}")
        if tts_enabled: speak("File not found error.")
    except ValueError as e:
        print(f"\nData Error: {e}")
        if tts_enabled: speak("Data error.")
    except ImportError as e:
         print(f"\nImport Error: {e}. Please ensure all required libraries are installed.")
         print("Try: pip install opencv-python mediapipe numpy scikit-learn joblib Pillow imagehash pyttsx3 pandas matplotlib fpdf")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        if tts_enabled: speak("An unexpected error occurred.")
        import traceback
        print("\n--- Traceback ---")
        traceback.print_exc()

    print("\n--- Program Finished ---")