import unittest
import numpy as np

from pyprocess.shapes.draw import Rect, Circle, Triangle

class TestShapeDrawing(unittest.TestCase):
    def setUp(self):
        # Test için boş bir görüntü oluştur
        self.img = np.zeros((500, 500, 3), dtype=np.uint8)
        self.point = (100, 100)
        self.size = (200, 200)
        self.color = (255, 0, 255)

    def test_rectangle_draw(self):
        rect = Rect(self.point, self.size, self.color)
        img_with_rect = rect.draw(self.img.copy(), draw_corners=True)
        self.assertIsNotNone(img_with_rect)
        self.assertNotEqual(np.sum(self.img), np.sum(img_with_rect))
    
    def test_circle_draw(self):
        circle = Circle(self.point, self.size, self.color)
        img_with_circle = circle.draw(self.img.copy(), draw_corners=True)
        self.assertIsNotNone(img_with_circle)
        self.assertNotEqual(np.sum(self.img), np.sum(img_with_circle))

    def test_triangle_draw(self):
        triangle = Triangle(self.point, self.size, self.color)
        img_with_triangle = triangle.draw(self.img.copy(), draw_corners=True)
        self.assertIsNotNone(img_with_triangle)
        self.assertNotEqual(np.sum(self.img), np.sum(img_with_triangle))
    
    def test_shape_update(self):
        rect = Rect(self.point, self.size, self.color)
        cursor_point_inside = (150, 150)
        cursor_point_outside = (400, 400)
        
        rect.update(cursor_point_inside)
        self.assertEqual(rect.color, (0, 255, 0))
        self.assertNotEqual(rect.point, self.point)

        rect.update(cursor_point_outside)
        self.assertEqual(rect.color, (255, 0, 255))

if __name__ == '__main__':
    unittest.main()
