from typing import Tuple

from cv2 import (
    FILLED as CV_FILLED,
    rectangle as cv_rect,
    circle as cv_circle,
    fillPoly as cv_fillPoly,
    line as cv_line
)

from numpy import (
    array as np_array,
    int32 as np_int32,
    ndarray
)

Point = Tuple[int, int]
Size = Tuple[int, int]
Color = Tuple[int, int, int]

class Shape:
    def __init__(self, point: Point, size: Size, color: Color = (255, 0, 255)):
        """
        Shape sınıfı, temel şekil sınıfıdır. Nokta, boyut ve renk bilgilerini alır.

        :param point: Şeklin sol üst köşe noktası
        :param size: Şeklin genişlik ve yüksekliği
        :param color: Şeklin rengi (varsayılan (255, 0, 255))
        """
        self.point: Point = point
        self.size: Size = size
        self.color: Color = color

    def update(self, cursor_point: Point) -> None:
        """
        Şeklin pozisyonunu ve rengini günceller.

        :param cursor_point: Güncellenen kursör noktası
        """
        cx, cy = cursor_point
        xmin, ymin = self.point
        xmax, ymax = xmin + self.size[0], ymin + self.size[1]

        if xmin < cx < xmax and ymin < cy < ymax:
            self.color = (0, 255, 0)
            self.point = (cx - self.size[0] // 2, cy - self.size[1] // 2)
        else:
            self.color = (255, 0, 255)

class Rect(Shape):
    def draw(self, img: ndarray, draw_corners: bool = False, corner_color: Color = (0, 0, 0), corner_thickness: int = 2) -> ndarray:
        """
        Dikdörtgen çizer.

        :param img: Üzerine çizim yapılacak görüntü
        :param draw_corners: Köşelerin çizilip çizilmeyeceği
        :param corner_color: Köşe renkleri
        :param corner_thickness: Köşe çizgi kalınlığı
        :return: Güncellenmiş görüntü
        """
        xmin, ymin = self.point
        xmax, ymax = xmin + self.size[0], ymin + self.size[1]
        cv_rect(img, (xmin, ymin), (xmax, ymax), self.color, CV_FILLED)
        
        if draw_corners:
            corner_length = min(self.size) // 8

            # Sol üst köşe
            cv_line(img, (xmin, ymin), (xmin + corner_length, ymin), corner_color, corner_thickness)
            cv_line(img, (xmin, ymin), (xmin, ymin + corner_length), corner_color, corner_thickness)

            # Sağ üst köşe
            cv_line(img, (xmax, ymin), (xmax - corner_length, ymin), corner_color, corner_thickness)
            cv_line(img, (xmax, ymin), (xmax, ymin + corner_length), corner_color, corner_thickness)

            # Sol alt köşe
            cv_line(img, (xmin, ymax), (xmin + corner_length, ymax), corner_color, corner_thickness)
            cv_line(img, (xmin, ymax), (xmin, ymax - corner_length), corner_color, corner_thickness)

            # Sağ alt köşe
            cv_line(img, (xmax, ymax), (xmax - corner_length, ymax), corner_color, corner_thickness)
            cv_line(img, (xmax, ymax), (xmax, ymax - corner_length), corner_color, corner_thickness)
        
        return img

class Circle(Shape):
    def draw(self, img: ndarray, draw_corners: bool = False, corner_length: int = 2, corner_color: Color = (0, 0, 0), corner_thickness: int = 2) -> ndarray:
        """
        Daire çizer.

        :param img: Üzerine çizim yapılacak görüntü
        :param draw_corners: Köşelerin çizilip çizilmeyeceği
        :param corner_length: Köşe uzunluğu
        :param corner_color: Köşe renkleri
        :param corner_thickness: Köşe çizgi kalınlığı
        :return: Güncellenmiş görüntü
        """
        center: Point = (self.point[0] + self.size[0] // 2, self.point[1] + self.size[1] // 2)
        radius: int = min(self.size) // 2
        cv_circle(img, center, radius, self.color, CV_FILLED)
        
        if draw_corners:
            cv_circle(img, center, radius + corner_length, corner_color, corner_thickness)
        return img

class Triangle(Shape):
    def draw(self, img: ndarray, draw_corners: bool = False, corner_color: Color = (0, 255, 0)) -> ndarray:
        """
        Üçgen çizer.

        :param img: Üzerine çizim yapılacak görüntü
        :param draw_corners: Köşelerin çizilip çizilmeyeceği
        :param corner_color: Köşe renkleri
        :return: Güncellenmiş görüntü
        """
        xmin, ymin = self.point
        xmax, ymax = xmin + self.size[0], ymin + self.size[1]
        center: Point = (xmin + self.size[0] // 2, ymin)
        pts: ndarray = np_array([center, (xmin, ymax), (xmax, ymax)], np_int32)
        pts = pts.reshape((-1, 1, 2))
        cv_fillPoly(img, [pts], self.color)
        
        if draw_corners:
            thickness: int = 2

            for i in range(3):
                start: Point = tuple(pts[i][0])
                end: Point = tuple(pts[(i + 1) % 3][0])
                
                # Kenar vektörünü hesapla
                dx: int = end[0] - start[0]
                dy: int = end[1] - start[1]
                
                # Kenarın ilk 1/4'ü
                first_quarter: Point = (int(start[0] + dx / 4), int(start[1] + dy / 4))
                cv_line(img, start, first_quarter, corner_color, thickness)
                
                # Kenarın son 1/4'ü
                last_quarter: Point = (int(end[0] - dx / 4), int(end[1] - dy / 4))
                cv_line(img, end, last_quarter, corner_color, thickness)
        
        return img
