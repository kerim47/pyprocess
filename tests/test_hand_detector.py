# pyprocess/tests/test_hand_detector.py

import unittest
import cv2
import numpy as np
from pyprocess.mp_utils import HandDetector

class TestHandDetector(unittest.TestCase):
    def setUp(self):
        self.detector = HandDetector()

    def test_init(self):
        self.assertEqual(self.detector.mode, False)
        self.assertEqual(self.detector.maxHands, 2)
        self.assertEqual(self.detector.detectionConf, 0.5)
        self.assertEqual(self.detector.trackConf, 0.5)
        self.assertEqual(self.detector.modelComplexity, 1)

    def test_find_hands(self):
        # Create a dummy image
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Assuming the function works correctly
        result_image = self.detector.find_hands(dummy_image, draw=False)
        
        self.assertIsNotNone(result_image)
        self.assertEqual(result_image.shape, dummy_image.shape)

    def test_find_position(self):
        # Create a dummy image
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Assuming find_hands needs to be called to update self.results
        self.detector.find_hands(dummy_image, draw=False)
        
        lm_list = self.detector.find_position(dummy_image, draw=False)
        
        self.assertIsInstance(lm_list, list)
        # Further checks can be added depending on the expected output

if __name__ == '__main__':
    unittest.main()
