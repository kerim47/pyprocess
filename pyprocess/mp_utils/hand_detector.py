import cv2
import mediapipe as mp

class HandDetector:
	def __init__(self, mode=False, maxHands=2, modelComplexity=1,detectionConf=0.5, trackConf=0.5):
		self.mode = mode
		self.maxHands = maxHands
		self.detectionConf = detectionConf
		self.trackConf = trackConf
		self.modelComplexity = modelComplexity

		self.mpHands = mp.solutions.hands
		self.hands   = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity ,self.detectionConf, self.trackConf)
		self.mpDraw  = mp.solutions.drawing_utils

	def find_hands(self, img, draw=True):
		img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		self.results = self.hands.process(img_rgb)
		
		if self.results.multi_hand_landmarks:
			for handLms in self.results.multi_hand_landmarks:
				if draw:
					self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
		return img
	
	def find_position(self, img, hand_no=0, draw=True):
		lm_list = []

		if self.results.multi_hand_landmarks:
			my_hand = self.results.multi_hand_landmarks[hand_no]
			for id, lm in enumerate(my_hand.landmark):
				height, width, _ = img.shape
				
				cx, cy = int(lm.x * width), int(lm.y * height)
				
				lm_list.append([id, cx, cy])

				if draw:
					cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
					
		return lm_list