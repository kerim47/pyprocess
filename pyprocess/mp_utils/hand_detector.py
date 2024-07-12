import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple, Optional, Dict

class HandDetector:
	def __init__(self, mode: bool = False, max_hands: int = 2, model_complexity: int = 1, detection_conf: float = 0.5, track_conf: float = 0.5):
		self.mode = mode
		self.max_hands = max_hands
		self.detection_conf = detection_conf
		self.track_conf = track_conf
		self.model_complexity = model_complexity

		self.mp_hands = mp.solutions.hands
		self.hands = self.mp_hands.Hands(
			static_image_mode=self.mode,
			max_num_hands=self.max_hands,
			model_complexity=self.model_complexity,
			min_detection_confidence=self.detection_conf,
			min_tracking_confidence=self.track_conf
		)
		self.mp_draw = mp.solutions.drawing_utils
		self.results = None

	def retrieve(self, img: np.ndarray) -> Tuple[Optional[mp.solutions.hands.HandLandmark], np.ndarray]:
		curr_img = self.process(img)
		return self.results, curr_img

	def process(self, img: np.ndarray) -> mp.solutions.hands.HandLandmark:
		img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		self.results = self.hands.process(img_rgb)
		img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
		return img_bgr

	def find_hands(self, img: np.ndarray, draw: bool = True) -> Tuple[np.ndarray, List[Dict]]:
		img = self.process(img)
		
		all_hands = []
		if self.results.multi_hand_landmarks:
			for hand_landmarks, hand_info in zip(self.results.multi_hand_landmarks, self.results.multi_handedness):
				if draw:
					self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
				
				hand_info_dict = {
					'landmarks': hand_landmarks,
					'label': hand_info.classification[0].label,
					'score': hand_info.classification[0].score
				}
				all_hands.append(hand_info_dict)
		
		return img, all_hands
	
	def find_positions(self, img: np.ndarray, hand_no: int = 0, draw: bool = True) -> List[List[int]]:
		lm_list = []
		if self.results.multi_hand_landmarks and hand_no < len(self.results.multi_hand_landmarks):
			my_hand = self.results.multi_hand_landmarks[hand_no]
			for id, lm in enumerate(my_hand.landmark):
				h, w, _ = img.shape
				cx, cy = int(lm.x * w), int(lm.y * h)
				lm_list.append([id, cx, cy])
				if draw:
					cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
		return lm_list

	def get_bounding_box(self, img: np.ndarray, hand_no: int = 0, margin=10) -> Optional[Tuple[int, int, int, int]]:
		if self.results.multi_hand_landmarks and hand_no < len(self.results.multi_hand_landmarks):
			my_hand = self.results.multi_hand_landmarks[hand_no]
			x_coordinates = [lm.x for lm in my_hand.landmark]
			y_coordinates = [lm.y for lm in my_hand.landmark]
			
			x_min, x_max = min(x_coordinates), max(x_coordinates)
			y_min, y_max = min(y_coordinates), max(y_coordinates)
			
			h, w, _ = img.shape
			x_min, y_min = int(x_min * w) - margin, int(y_min * h) - margin
			x_max, y_max = int(x_max * w) + margin, int(y_max * h) + margin
			
			return (x_min, y_min, x_max - x_min, y_max - y_min)
		return None

	def draw_bounding_box(self, 
                      img: np.ndarray, 
                      bbox: Tuple[int, int, int, int], 
                      color: Tuple[int, int, int] = (0, 0, 0), 
                      thickness: int = 2,
                      text: str = None,
                      text_color: Tuple[int, int, int] = (255, 255, 255),
                      font_scale: float = 1,
                      font_thickness: int = 1,
                      text_position: str = 'left',
                      padding: int = 0) -> np.ndarray:
		x, y, w, h = bbox
		img = cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness)
		
		if text:
			text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
			
			
			
			if text_position.lower() == 'left':
				text_x = x
			elif text_position.lower() == 'right':
				text_x = x + w - text_size[0] - 2 * padding
			elif text_position.lower() == 'center':
				text_x = x + (w - text_size[0] - 2 * padding) // 2
			else:  # Default to left if invalid position is given
				text_x = x
			
			text_y = y
			
			# Arka plan dikdörtgeni çiz (padding ile)
			bg_rect = (text_x, text_y - text_size[1] - padding, 
					   text_size[0] + 2*padding, text_size[1] + 2*padding)
			cv2.rectangle(img, (bg_rect[0], bg_rect[1]), 
						  (bg_rect[0] + bg_rect[2], bg_rect[1] + bg_rect[3]), 
						  color, cv2.FILLED)
			
			# Metni yaz (padding ile)
			cv2.putText(img, text, (text_x + padding, text_y + padding), cv2.FONT_HERSHEY_SIMPLEX, 
						font_scale, text_color, font_thickness, cv2.LINE_AA)
		
		return img

	@staticmethod
	def calculate_distance(point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
		return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)